/*
 * Copyright 2026 Igalia S.L.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <utility>
#include <memory>
#include <algorithm>

#if !defined(_WIN32)
#include <unistd.h>
#include <sys/wait.h>
#else
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#define dup    _dup
#define dup2   _dup2
#define close  _close
#endif

#include "VkCodecUtils/VulkanDeviceContext.h"
#include "VkVideoCore/VkVideoCoreProfile.h"
#include "VkVideoCore/VulkanVideoCapabilities.h"
#include "VkVSCommon.h"

namespace {

struct FlagName {
    uint32_t bit;
    const char* name;
};

// Decode a bitmask into a "NAME|NAME|..." string using the supplied bit->name table.
std::string decodeFlags(uint32_t flags, const FlagName* names, size_t count)
{
    std::string decoded;
    for (size_t i = 0; i < count; ++i) {
        if (flags & names[i].bit) {
            decoded += decoded.empty() ? "" : "|";
            decoded += names[i].name;
        }
    }
    return decoded;
}

// Output backend. Markdown is the default; --json selects the JSON backend.
// Sections nest via heading()/endSection(); typed row() helpers add key/value pairs.
class Emitter {
public:
    virtual ~Emitter() = default;

    // Start a section. level 1 = '#', 2 = '##', 3 = '###' (Markdown) / nested object (JSON).
    // key is the JSON object key (machine name); title is the human Markdown heading.
    // A base level (see setBaseLevel) is added so a whole report can be nested deeper, e.g.
    // as one element of a top-level devices list.
    virtual void heading(int level, const std::string& key, const std::string& title) = 0;
    virtual void endSection(int level) = 0;

    // Shift all subsequent heading levels by `base` (Markdown only; JSON nests structurally).
    void setBaseLevel(int base) { m_baseLevel = base; }
    int baseLevel() const { return m_baseLevel; }

    // Begin/end an array of like sections (JSON only cares; Markdown ignores).
    virtual void beginArray(const std::string& key) = 0;
    virtual void endArray() = 0;

    virtual void str(const char* field, const std::string& value) = 0;
    virtual void u64(const char* field, uint64_t value) = 0;
    virtual void i64(const char* field, int64_t value) = 0;
    virtual void boolean(const char* field, bool value) = 0;
    virtual void hex(const char* field, uint64_t value) = 0;
    virtual void extent(const char* field, const VkExtent2D& e) = 0;
    virtual void flags(const char* field, uint32_t value, const FlagName* names, size_t count) = 0;

    // A standalone note (e.g. "not supported") attached to the current section.
    virtual void note(const std::string& text) = 0;

    // Add a value to a single-column list with the given header (Markdown renders a one-column
    // table; JSON ignores it -- structured callers emit a proper array instead).
    virtual void listItem(const char* header, const std::string& value) = 0;

    // Set the column headers for the next two-column table (Markdown only; reset to the
    // defaults "Field"/"Value" after the table is flushed). No-op in JSON.
    virtual void tableHeaders(const char* col1, const char* col2) = 0;

    // True for structured backends (JSON) where arrays of objects make sense; false for
    // Markdown, which prefers a flat table.
    virtual bool structured() const = 0;

    virtual void finish() = 0;

protected:
    int m_baseLevel = 0;
};

Emitter* g_emit = nullptr;

#define EMIT_FLAGS(field, value, table) g_emit->flags(field, value, table, sizeof(table) / sizeof(table[0]))

// ----- Markdown backend -------------------------------------------------------------------

class MarkdownEmitter : public Emitter {
public:
    void heading(int level, const std::string&, const std::string& title) override
    {
        flushTable();
        std::cout << "\n" << std::string(m_baseLevel + level, '#') << " " << title << "\n";
    }
    void endSection(int) override { flushTable(); }
    void beginArray(const std::string&) override {}
    void endArray() override {}

    void str(const char* field, const std::string& value) override { addRow(field, value); }
    void u64(const char* field, uint64_t value) override { addRow(field, std::to_string(value)); }
    void i64(const char* field, int64_t value) override { addRow(field, std::to_string(value)); }
    void boolean(const char* field, bool value) override { addRow(field, value ? "true" : "false"); }
    void hex(const char* field, uint64_t value) override
    {
        char buf[24];
        snprintf(buf, sizeof(buf), "0x%llx", (unsigned long long)value);
        addRow(field, buf);
    }
    void extent(const char* field, const VkExtent2D& e) override
    {
        addRow(field, std::to_string(e.width) + " x " + std::to_string(e.height));
    }
    void flags(const char* field, uint32_t value, const FlagName* names, size_t count) override
    {
        char buf[24];
        snprintf(buf, sizeof(buf), "0x%x", value);
        std::string decoded = decodeFlags(value, names, count);
        addRow(field, decoded.empty() ? std::string(buf) : std::string(buf) + " [" + decoded + "]");
    }
    void note(const std::string& text) override { flushTable(); std::cout << "\n" << text << "\n"; }
    void listItem(const char* header, const std::string& value) override
    {
        m_listHeader = header;
        m_listItems.push_back(value);
    }
    void tableHeaders(const char* col1, const char* col2) override
    {
        m_fieldHdr = col1;
        m_valueHdr = col2;
    }
    bool structured() const override { return false; }
    void finish() override { flushTable(); }

private:
    void addRow(const char* field, const std::string& value) { m_rows.emplace_back(field, value); }

    static std::string pad(const std::string& s, size_t width)
    {
        return s + std::string(width - s.size(), ' ');
    }

    void flushTable()
    {
        flushList();
        if (m_rows.empty()) {
            return;
        }

        // Pad both columns to their widest cell so values line up under each other.
        size_t fieldWidth = m_fieldHdr.size();
        size_t valueWidth = m_valueHdr.size();
        for (const auto& row : m_rows) {
            fieldWidth = std::max(fieldWidth, row.first.size());
            valueWidth = std::max(valueWidth, row.second.size());
        }

        std::cout << "\n| " << pad(m_fieldHdr, fieldWidth) << " | " << pad(m_valueHdr, valueWidth) << " |\n";
        std::cout << "| " << std::string(fieldWidth, '-') << " | " << std::string(valueWidth, '-') << " |\n";
        for (const auto& row : m_rows) {
            std::cout << "| " << pad(row.first, fieldWidth) << " | " << pad(row.second, valueWidth) << " |\n";
        }
        m_rows.clear();
        m_fieldHdr = "Field"; // reset to defaults for the next table
        m_valueHdr = "Value";
    }

    void flushList()
    {
        if (m_listItems.empty()) {
            return;
        }
        size_t width = m_listHeader.size();
        for (const auto& item : m_listItems) {
            width = std::max(width, item.size());
        }
        std::cout << "\n| " << pad(m_listHeader, width) << " |\n";
        std::cout << "| " << std::string(width, '-') << " |\n";
        for (const auto& item : m_listItems) {
            std::cout << "| " << pad(item, width) << " |\n";
        }
        m_listItems.clear();
    }

    std::vector<std::pair<std::string, std::string>> m_rows;
    std::string m_fieldHdr = "Field";
    std::string m_valueHdr = "Value";
    std::string m_listHeader;
    std::vector<std::string> m_listItems;
};

// ----- JSON backend -----------------------------------------------------------------------

class JsonEmitter : public Emitter {
public:
    JsonEmitter()
    {
        std::cout << "{";
        m_needComma = false;
        m_inArray.push_back(false); // root object
    }

    void heading(int, const std::string& key, const std::string&) override
    {
        comma();
        if (inArray()) {
            // Array element: anonymous object. The caller emits its own identifying field.
            std::cout << "\n" << indent() << "{";
        } else {
            std::cout << "\n" << indent() << quote(key) << ": {";
        }
        m_needComma = false;
        m_inArray.push_back(false);
    }
    void endSection(int) override { closeBrace(); }

    void beginArray(const std::string& key) override
    {
        comma();
        std::cout << "\n" << indent() << quote(key) << ": [";
        m_needComma = false;
        m_inArray.push_back(true);
    }
    void endArray() override
    {
        m_inArray.pop_back();
        std::cout << "\n" << indent() << "]";
        m_needComma = true;
    }

    void str(const char* field, const std::string& value) override { field_(field); std::cout << quote(value); }
    void u64(const char* field, uint64_t value) override { field_(field); std::cout << value; }
    void i64(const char* field, int64_t value) override { field_(field); std::cout << value; }
    void boolean(const char* field, bool value) override { field_(field); std::cout << (value ? "true" : "false"); }
    void hex(const char* field, uint64_t value) override
    {
        char buf[24];
        snprintf(buf, sizeof(buf), "0x%llx", (unsigned long long)value);
        field_(field);
        std::cout << quote(buf);
    }
    void extent(const char* field, const VkExtent2D& e) override
    {
        field_(field);
        std::cout << "{ \"width\": " << e.width << ", \"height\": " << e.height << " }";
    }
    void flags(const char* field, uint32_t value, const FlagName* names, size_t count) override
    {
        field_(field);
        std::cout << "{ \"value\": " << value << ", \"names\": [";
        bool first = true;
        for (size_t i = 0; i < count; ++i) {
            if (value & names[i].bit) {
                std::cout << (first ? "" : ", ") << quote(names[i].name);
                first = false;
            }
        }
        std::cout << "] }";
    }
    void note(const std::string& text) override { field_("note"); std::cout << quote(text); }
    void listItem(const char*, const std::string&) override {} // structured callers emit arrays
    void tableHeaders(const char*, const char*) override {}     // JSON has no table headers
    bool structured() const override { return true; }

    void finish() override
    {
        while (m_inArray.size() > 1) {
            if (m_inArray.back()) {
                endArray();
            } else {
                closeBrace();
            }
        }
        std::cout << "\n}\n";
    }

private:
    bool inArray() const { return m_inArray.back(); }
    void closeBrace()
    {
        m_inArray.pop_back();
        std::cout << "\n" << indent() << "}";
        m_needComma = true;
    }
    void field_(const char* field)
    {
        comma();
        std::cout << "\n" << indent() << quote(field) << ": ";
        m_needComma = true;
    }
    void comma()
    {
        if (m_needComma) {
            std::cout << ",";
        }
    }
    std::string indent() const { return std::string(m_inArray.size() * 2, ' '); }
    static std::string quote(const std::string& s)
    {
        std::string out = "\"";
        for (char c : s) {
            if (c == '"' || c == '\\') {
                out += '\\';
            }
            out += c;
        }
        out += "\"";
        return out;
    }

    std::vector<bool> m_inArray;
    bool m_needComma = false;
};

// Open a codec section (## <codec>) and, for structured output, emit a "codec" id field so
// JSON array elements are self-describing. Markdown skips the field (the heading names it).
void beginCodec(const std::string& key)
{
    g_emit->heading(2, key, key);
    if (g_emit->structured()) {
        g_emit->str("codec", key);
    }
}

// Emit a codec section heading then a "not supported" note (used when a codec is absent).
void emitUnsupported(const std::string& key, const std::string& reason)
{
    beginCodec(key);
    g_emit->note(reason);
    g_emit->endSection(2);
}

// Emit a device-level section (## <name>) carrying only a skip note, for a device that
// could not be initialized. Used as an element of the top-level devices array.
void emitDeviceSkip(const std::string& name, const std::string& reason)
{
    g_emit->heading(1, "device", name);
    if (g_emit->structured()) {
        g_emit->str("deviceName", name);
    }
    g_emit->note(reason);
    g_emit->endSection(1);
}

const FlagName kRateControlModeNames[] = {
    { VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DEFAULT_KHR, "DEFAULT" },
    { VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR, "DISABLED" },
    { VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR, "CBR" },
    { VK_VIDEO_ENCODE_RATE_CONTROL_MODE_VBR_BIT_KHR, "VBR" },
};

const FlagName kEncodeCapNames[] = {
    { VK_VIDEO_ENCODE_CAPABILITY_PRECEDING_EXTERNALLY_ENCODED_BYTES_BIT_KHR, "PRECEDING_EXTERNALLY_ENCODED_BYTES" },
    { VK_VIDEO_ENCODE_CAPABILITY_INSUFFICIENT_BITSTREAM_BUFFER_RANGE_DETECTION_BIT_KHR, "INSUFFICIENT_BITSTREAM_BUFFER_RANGE_DETECTION" },
    { VK_VIDEO_ENCODE_CAPABILITY_QUANTIZATION_DELTA_MAP_BIT_KHR, "QUANTIZATION_DELTA_MAP" },
    { VK_VIDEO_ENCODE_CAPABILITY_EMPHASIS_MAP_BIT_KHR, "EMPHASIS_MAP" },
};

const FlagName kEncodeFeedbackNames[] = {
    { VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BUFFER_OFFSET_BIT_KHR, "BITSTREAM_BUFFER_OFFSET" },
    { VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BYTES_WRITTEN_BIT_KHR, "BITSTREAM_BYTES_WRITTEN" },
    { VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_HAS_OVERRIDES_BIT_KHR, "BITSTREAM_HAS_OVERRIDES" },
};

const FlagName kIntraRefreshModeNames[] = {
    { VK_VIDEO_ENCODE_INTRA_REFRESH_MODE_PER_PICTURE_PARTITION_BIT_KHR, "PER_PICTURE_PARTITION" },
    { VK_VIDEO_ENCODE_INTRA_REFRESH_MODE_BLOCK_BASED_BIT_KHR, "BLOCK_BASED" },
    { VK_VIDEO_ENCODE_INTRA_REFRESH_MODE_BLOCK_ROW_BASED_BIT_KHR, "BLOCK_ROW_BASED" },
    { VK_VIDEO_ENCODE_INTRA_REFRESH_MODE_BLOCK_COLUMN_BASED_BIT_KHR, "BLOCK_COLUMN_BASED" },
};

const char* formatName(VkFormat format)
{
    switch (format) {
    case VK_FORMAT_UNDEFINED:                            return "VK_FORMAT_UNDEFINED";
    case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:             return "VK_FORMAT_G8_B8R8_2PLANE_420_UNORM";
    case VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM:            return "VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM";
    case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16: return "VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16";
    case VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16: return "VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16";
    case VK_FORMAT_G16_B16R16_2PLANE_420_UNORM:          return "VK_FORMAT_G16_B16R16_2PLANE_420_UNORM";
    case VK_FORMAT_G8_B8R8_2PLANE_422_UNORM:             return "VK_FORMAT_G8_B8R8_2PLANE_422_UNORM";
    case VK_FORMAT_G8_B8R8_2PLANE_444_UNORM:             return "VK_FORMAT_G8_B8R8_2PLANE_444_UNORM";
    default:                                             return nullptr;
    }
}

void emitFormat(const char* field, VkFormat format)
{
    const char* name = formatName(format);
    if (name) {
        g_emit->str(field, name);
    } else {
        g_emit->i64(field, (int)format);
    }
}

const char* deviceTypeName(VkPhysicalDeviceType type)
{
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "INTEGRATED_GPU";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "DISCRETE_GPU";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return "VIRTUAL_GPU";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:            return "CPU";
    default:                                     return "OTHER";
    }
}

// True if the physical device advertises VK_KHR_video_queue. Checked quietly (no logging)
// so non-video devices like llvmpipe are filtered out before InitPhysicalDevice, which
// would otherwise log a missing-extension error to stderr.
bool deviceHasVideoQueue(const VulkanDeviceContext* vkDevCtx, VkPhysicalDevice phys)
{
    std::vector<VkExtensionProperties> exts;
    if (vk::enumerate(vkDevCtx, phys, nullptr, exts) != VK_SUCCESS) {
        return false;
    }
    for (const auto& ext : exts) {
        if (strcmp(ext.extensionName, VK_KHR_VIDEO_QUEUE_EXTENSION_NAME) == 0) {
            return true;
        }
    }
    return false;
}

// True if the device passes the optional deviceID (hex) and deviceUuid selection filters.
bool deviceMatchesFilter(const VulkanDeviceContext* vkDevCtx, VkPhysicalDevice phys,
                         int32_t deviceId, const vk::DeviceUuidUtils& deviceUuid)
{
    if (deviceId == -1 && !deviceUuid) {
        return true;
    }
    VkPhysicalDeviceVulkan11Properties v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &v11};
    vkDevCtx->GetPhysicalDeviceProperties2(phys, &p2);
    if (deviceId != -1 && p2.properties.deviceID != (uint32_t)deviceId) {
        return false;
    }
    if (deviceUuid && !deviceUuid.Compare(v11.deviceUUID)) {
        return false;
    }
    return true;
}

// Sorted list of the device's VK_KHR_video_* extension names.
std::vector<std::string> videoExtensions(const VulkanDeviceContext* vkDevCtx, VkPhysicalDevice phys)
{
    std::vector<std::string> result;
    std::vector<VkExtensionProperties> exts;
    if (vk::enumerate(vkDevCtx, phys, nullptr, exts) != VK_SUCCESS) {
        return result;
    }
    for (const auto& ext : exts) {
        if (strncmp(ext.extensionName, "VK_KHR_video", strlen("VK_KHR_video")) == 0) {
            result.emplace_back(ext.extensionName);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

// Common VkVideoCapabilitiesKHR fields shared by decode and encode (the "Generic" section).
void emitGenericCaps(const VkVideoCapabilitiesKHR& caps)
{
    g_emit->hex("flags", caps.flags);
    g_emit->extent("minCodedExtent", caps.minCodedExtent);
    g_emit->extent("maxCodedExtent", caps.maxCodedExtent);
    g_emit->extent("pictureAccessGranularity", caps.pictureAccessGranularity);
    g_emit->u64("minBitstreamBufferOffsetAlignment", caps.minBitstreamBufferOffsetAlignment);
    g_emit->u64("minBitstreamBufferSizeAlignment", caps.minBitstreamBufferSizeAlignment);
    g_emit->u64("maxDpbSlots", caps.maxDpbSlots);
    g_emit->u64("maxActiveReferencePictures", caps.maxActiveReferencePictures);
    g_emit->str("stdHeaderVersion.extensionName", caps.stdHeaderVersion.extensionName);
    g_emit->hex("stdHeaderVersion.specVersion", caps.stdHeaderVersion.specVersion);
}

// Build a profile for a codec operation with the given codec profile IDC, chroma
// subsampling and bit depth. Each codec needs its own factory; CreateDecodeProfile
// asserts on H265/VP9 only.
VkVideoCoreProfile makeProfileEx(VkVideoCodecOperationFlagBitsKHR codec, uint32_t profileIdc,
                                 VkVideoChromaSubsamplingFlagsKHR chroma,
                                 VkVideoComponentBitDepthFlagsKHR depth)
{
    switch (codec) {
    case VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR:
        return VkVideoCoreProfile::CreateDecodeH264Profile(chroma, depth, depth, profileIdc,
                   VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_PROGRESSIVE_KHR);
    case VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR:
    case VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR:
        return VkVideoCoreProfile::CreateDecodeProfile(codec, chroma, depth, depth, profileIdc);
    case VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR:
        return VkVideoCoreProfile::CreateDecodeAV1Profile(chroma, depth, depth, profileIdc,
                   /*filmGrainSupport*/ false);
    case VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR:
    case VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR:
    case VK_VIDEO_CODEC_OPERATION_ENCODE_AV1_BIT_KHR:
        return VkVideoCoreProfile::CreateEncodeProfile(codec, chroma, depth, depth, profileIdc,
                   VK_VIDEO_ENCODE_TUNING_MODE_DEFAULT_KHR);
    default:
        return VkVideoCoreProfile();
    }
}

// Representative 8-bit 4:2:0 Main profile used for the detailed capability dump.
VkVideoCoreProfile makeProfile(VkVideoCodecOperationFlagBitsKHR codec)
{
    uint32_t mainIdc = 0;
    switch (codec) {
    case VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR:
    case VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR:
        mainIdc = STD_VIDEO_H264_PROFILE_IDC_MAIN; break;
    case VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR:
    case VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR:
        mainIdc = STD_VIDEO_H265_PROFILE_IDC_MAIN; break;
    case VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR:
    case VK_VIDEO_CODEC_OPERATION_ENCODE_AV1_BIT_KHR:
        mainIdc = STD_VIDEO_AV1_PROFILE_MAIN; break;
    case VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR:
        mainIdc = STD_VIDEO_VP9_PROFILE_0; break;
    default:
        return VkVideoCoreProfile();
    }
    return makeProfileEx(codec, mainIdc, VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR,
                         VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR);
}

// --- Profile matrix probing ---------------------------------------------------------------

struct NamedValue {
    uint32_t value;
    const char* name;
};

const NamedValue kChromaValues[] = {
    { VK_VIDEO_CHROMA_SUBSAMPLING_MONOCHROME_BIT_KHR, "monochrome" },
    { VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR, "420" },
    { VK_VIDEO_CHROMA_SUBSAMPLING_422_BIT_KHR, "422" },
    { VK_VIDEO_CHROMA_SUBSAMPLING_444_BIT_KHR, "444" },
};

const NamedValue kDepthValues[] = {
    { VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR, "8" },
    { VK_VIDEO_COMPONENT_BIT_DEPTH_10_BIT_KHR, "10" },
    { VK_VIDEO_COMPONENT_BIT_DEPTH_12_BIT_KHR, "12" },
};

const NamedValue kH264Profiles[] = {
    { STD_VIDEO_H264_PROFILE_IDC_BASELINE, "BASELINE" },
    { STD_VIDEO_H264_PROFILE_IDC_MAIN, "MAIN" },
    { STD_VIDEO_H264_PROFILE_IDC_HIGH, "HIGH" },
    { STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE, "HIGH_444_PREDICTIVE" },
};

const NamedValue kH265Profiles[] = {
    { STD_VIDEO_H265_PROFILE_IDC_MAIN, "MAIN" },
    { STD_VIDEO_H265_PROFILE_IDC_MAIN_10, "MAIN_10" },
    { STD_VIDEO_H265_PROFILE_IDC_MAIN_STILL_PICTURE, "MAIN_STILL_PICTURE" },
    { STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS, "FORMAT_RANGE_EXTENSIONS" },
    { STD_VIDEO_H265_PROFILE_IDC_SCC_EXTENSIONS, "SCC_EXTENSIONS" },
};

const NamedValue kAV1Profiles[] = {
    { STD_VIDEO_AV1_PROFILE_MAIN, "MAIN" },
    { STD_VIDEO_AV1_PROFILE_HIGH, "HIGH" },
    { STD_VIDEO_AV1_PROFILE_PROFESSIONAL, "PROFESSIONAL" },
};

const NamedValue kVP9Profiles[] = {
    { STD_VIDEO_VP9_PROFILE_0, "0" },
    { STD_VIDEO_VP9_PROFILE_1, "1" },
    { STD_VIDEO_VP9_PROFILE_2, "2" },
    { STD_VIDEO_VP9_PROFILE_3, "3" },
};

// Return the profile-IDC name table for a codec (and its size via outCount).
const NamedValue* codecProfiles(VkVideoCodecOperationFlagBitsKHR codec, size_t& outCount)
{
    switch (codec) {
    case VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR:
    case VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR:
        outCount = sizeof(kH264Profiles) / sizeof(kH264Profiles[0]); return kH264Profiles;
    case VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR:
    case VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR:
        outCount = sizeof(kH265Profiles) / sizeof(kH265Profiles[0]); return kH265Profiles;
    case VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR:
    case VK_VIDEO_CODEC_OPERATION_ENCODE_AV1_BIT_KHR:
        outCount = sizeof(kAV1Profiles) / sizeof(kAV1Profiles[0]); return kAV1Profiles;
    case VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR:
        outCount = sizeof(kVP9Profiles) / sizeof(kVP9Profiles[0]); return kVP9Profiles;
    default:
        outCount = 0; return nullptr;
    }
}

bool profileSupported(const VulkanDeviceContext* vkDevCtx, VkVideoCodecOperationFlagBitsKHR codec,
                      uint32_t profileIdc, VkVideoChromaSubsamplingFlagsKHR chroma,
                      VkVideoComponentBitDepthFlagsKHR depth)
{
    VkVideoCoreProfile profile = makeProfileEx(codec, profileIdc, chroma, depth);
    if (!profile) {
        return false;
    }

    // The driver requires the codec-specific capabilities struct chained into pNext; without
    // it the query fails for every profile. Chain the matching struct per codec.
    VkVideoDecodeH264CapabilitiesKHR dec264{ VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_CAPABILITIES_KHR };
    VkVideoDecodeH265CapabilitiesKHR dec265{ VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_CAPABILITIES_KHR };
    VkVideoDecodeAV1CapabilitiesKHR  decAV1{ VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_CAPABILITIES_KHR };
    VkVideoDecodeVP9CapabilitiesKHR  decVP9{ VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_CAPABILITIES_KHR };
    VkVideoEncodeH264CapabilitiesKHR enc264{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_CAPABILITIES_KHR };
    VkVideoEncodeH265CapabilitiesKHR enc265{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_CAPABILITIES_KHR };
    VkVideoEncodeAV1CapabilitiesKHR  encAV1{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_CAPABILITIES_KHR };

    VkVideoDecodeCapabilitiesKHR decodeCaps{ VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR };
    VkVideoEncodeCapabilitiesKHR encodeCaps{ VK_STRUCTURE_TYPE_VIDEO_ENCODE_CAPABILITIES_KHR };

    void* codecChain = nullptr;
    void* topChain = nullptr;
    switch (codec) {
    case VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR: codecChain = &dec264; topChain = &decodeCaps; break;
    case VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR: codecChain = &dec265; topChain = &decodeCaps; break;
    case VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR:  codecChain = &decAV1; topChain = &decodeCaps; break;
    case VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR:  codecChain = &decVP9; topChain = &decodeCaps; break;
    case VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR: codecChain = &enc264; topChain = &encodeCaps; break;
    case VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR: codecChain = &enc265; topChain = &encodeCaps; break;
    case VK_VIDEO_CODEC_OPERATION_ENCODE_AV1_BIT_KHR:  codecChain = &encAV1; topChain = &encodeCaps; break;
    default: return false;
    }
    static_cast<VkBaseOutStructure*>(topChain)->pNext = static_cast<VkBaseOutStructure*>(codecChain);

    VkVideoCapabilitiesKHR caps{ VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR, topChain };
    return vkDevCtx->GetPhysicalDeviceVideoCapabilitiesKHR(vkDevCtx->getPhysicalDevice(),
                                                           profile.GetProfile(), &caps) == VK_SUCCESS;
}

// Emit a "Profiles" section listing every supported (profile, chroma, bitDepth) combination.
// Unsupported combinations are skipped entirely. In Markdown each supported combo is one
// table row (profile -> "chroma N-bit"); in JSON it is an object in the "supported" array.
void emitProfiles(const VulkanDeviceContext* vkDevCtx, VkVideoCodecOperationFlagBitsKHR codec)
{
    size_t profileCount = 0;
    const NamedValue* profiles = codecProfiles(codec, profileCount);
    const bool json = g_emit->structured();

    g_emit->heading(3, "profiles", "Profiles");
    if (json) {
        g_emit->beginArray("supported");
    } else {
        g_emit->tableHeaders("Profile", "ColorSpace");
    }
    for (size_t p = 0; p < profileCount; ++p) {
        for (const NamedValue& chroma : kChromaValues) {
            for (const NamedValue& depth : kDepthValues) {
                if (!profileSupported(vkDevCtx, codec, profiles[p].value, chroma.value, depth.value)) {
                    continue;
                }
                if (json) {
                    g_emit->heading(4, "profile", profiles[p].name);
                    g_emit->str("profile", profiles[p].name);
                    g_emit->str("chroma", chroma.name);
                    g_emit->str("bitDepth", depth.name);
                    g_emit->endSection(4);
                } else {
                    g_emit->str(profiles[p].name,
                                std::string(chroma.name) + " " + depth.name + "-bit");
                }
            }
        }
    }
    if (json) {
        g_emit->endArray();
    }
    g_emit->endSection(3);
}

// Short codec key for headings/JSON (e.g. "h.264"), derived from CodecToName ("decode h.264").
std::string codecKey(VkVideoCodecOperationFlagBitsKHR codec)
{
    std::string name = VkVideoCoreProfile::CodecToName(codec);
    size_t space = name.rfind(' ');
    return (space == std::string::npos) ? name : name.substr(space + 1);
}

void dumpDecodeCaps(const VulkanDeviceContext* vkDevCtx, VkVideoCodecOperationFlagBitsKHR codec)
{
    std::string key = codecKey(codec);

    int32_t queueFamily = vkDevCtx->GetVideoDecodeQueueFamilyIdx();
    VkVideoCodecOperationFlagsKHR supported =
        VulkanVideoCapabilities::GetSupportedCodecs(vkDevCtx, vkDevCtx->getPhysicalDevice(),
                                                    &queueFamily, VK_QUEUE_VIDEO_DECODE_BIT_KHR);
    if ((supported & codec) == 0) {
        emitUnsupported(key, "not supported (no decode queue family)");
        return;
    }

    VkVideoCoreProfile profile = makeProfile(codec);
    VkVideoCapabilitiesKHR videoCaps{};
    VkVideoDecodeCapabilitiesKHR decodeCaps{};
    VkResult result = VulkanVideoCapabilities::GetVideoDecodeCapabilities(vkDevCtx, profile,
                                                                          videoCaps, decodeCaps);
    if (result != VK_SUCCESS) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%s (0x%x)",
                 IsVideoUnsupportedResult(result) ? "unsupported, skipped" : "query failed", result);
        emitUnsupported(key, buf);
        return;
    }

    beginCodec(key);

    emitProfiles(vkDevCtx, codec);

    g_emit->heading(3, "generic", "Generic");
    emitGenericCaps(videoCaps);
    g_emit->endSection(3);

    g_emit->heading(3, "decode", "Decode");
    g_emit->hex("flags", decodeCaps.flags);
    VkFormat pictureFormat = VK_FORMAT_UNDEFINED;
    VkFormat referenceFormat = VK_FORMAT_UNDEFINED;
    if (VulkanVideoCapabilities::GetSupportedVideoFormats(vkDevCtx, profile, decodeCaps.flags,
                                                          pictureFormat, referenceFormat) == VK_SUCCESS) {
        emitFormat("pictureFormat", pictureFormat);
        emitFormat("referencePicturesFormat", referenceFormat);
    }
    g_emit->endSection(3);

    g_emit->endSection(2);
}

// "### Encode" section: common VkVideoEncodeCapabilitiesKHR fields + quantization map.
void emitEncodeCommon(const VkVideoEncodeCapabilitiesKHR& encCaps,
                      const VkVideoEncodeQuantizationMapCapabilitiesKHR& qmapCaps)
{
    EMIT_FLAGS("flags", encCaps.flags, kEncodeCapNames);
    EMIT_FLAGS("rateControlModes", encCaps.rateControlModes, kRateControlModeNames);
    g_emit->u64("maxRateControlLayers", encCaps.maxRateControlLayers);
    g_emit->u64("maxBitrate", encCaps.maxBitrate);
    g_emit->u64("maxQualityLevels", encCaps.maxQualityLevels);
    g_emit->extent("encodeInputPictureGranularity", encCaps.encodeInputPictureGranularity);
    EMIT_FLAGS("supportedEncodeFeedbackFlags", encCaps.supportedEncodeFeedbackFlags, kEncodeFeedbackNames);
    g_emit->extent("maxQuantizationMapExtent", qmapCaps.maxQuantizationMapExtent);
}

// "### Intra-refresh" section.
void emitIntraRefresh(const VkVideoEncodeIntraRefreshCapabilitiesKHR& c)
{
    EMIT_FLAGS("supportedModes", c.intraRefreshModes, kIntraRefreshModeNames);
    g_emit->u64("maxIntraRefreshCycleDuration", c.maxIntraRefreshCycleDuration);
    g_emit->u64("maxIntraRefreshActiveReferencePictures", c.maxIntraRefreshActiveReferencePictures);
    g_emit->boolean("partitionIndependentIntraRefreshRegions", c.partitionIndependentIntraRefreshRegions);
    g_emit->boolean("nonRectangularIntraRefreshRegions", c.nonRectangularIntraRefreshRegions);
}

// Per-codec "### <codec>" capability sections (VkVideoEncodeXXXCapabilitiesKHR).
void emitCodecCaps(const VkVideoEncodeH264CapabilitiesKHR& c)
{
    g_emit->hex("flags", c.flags);
    g_emit->u64("maxLevelIdc", c.maxLevelIdc);
    g_emit->u64("maxSliceCount", c.maxSliceCount);
    g_emit->u64("maxPPictureL0ReferenceCount", c.maxPPictureL0ReferenceCount);
    g_emit->u64("maxBPictureL0ReferenceCount", c.maxBPictureL0ReferenceCount);
    g_emit->u64("maxL1ReferenceCount", c.maxL1ReferenceCount);
    g_emit->u64("maxTemporalLayerCount", c.maxTemporalLayerCount);
    g_emit->boolean("expectDyadicTemporalLayerPattern", c.expectDyadicTemporalLayerPattern);
    g_emit->i64("minQp", c.minQp);
    g_emit->i64("maxQp", c.maxQp);
    g_emit->boolean("prefersGopRemainingFrames", c.prefersGopRemainingFrames);
    g_emit->boolean("requiresGopRemainingFrames", c.requiresGopRemainingFrames);
    g_emit->hex("stdSyntaxFlags", c.stdSyntaxFlags);
}

void emitCodecCaps(const VkVideoEncodeH265CapabilitiesKHR& c)
{
    g_emit->hex("flags", c.flags);
    g_emit->u64("maxLevelIdc", c.maxLevelIdc);
    g_emit->u64("maxSliceSegmentCount", c.maxSliceSegmentCount);
    g_emit->extent("maxTiles", c.maxTiles);
    g_emit->hex("ctbSizes", c.ctbSizes);
    g_emit->hex("transformBlockSizes", c.transformBlockSizes);
    g_emit->u64("maxPPictureL0ReferenceCount", c.maxPPictureL0ReferenceCount);
    g_emit->u64("maxBPictureL0ReferenceCount", c.maxBPictureL0ReferenceCount);
    g_emit->u64("maxL1ReferenceCount", c.maxL1ReferenceCount);
    g_emit->u64("maxSubLayerCount", c.maxSubLayerCount);
    g_emit->boolean("expectDyadicTemporalSubLayerPattern", c.expectDyadicTemporalSubLayerPattern);
    g_emit->i64("minQp", c.minQp);
    g_emit->i64("maxQp", c.maxQp);
    g_emit->boolean("prefersGopRemainingFrames", c.prefersGopRemainingFrames);
    g_emit->boolean("requiresGopRemainingFrames", c.requiresGopRemainingFrames);
    g_emit->hex("stdSyntaxFlags", c.stdSyntaxFlags);
}

void emitCodecCaps(const VkVideoEncodeAV1CapabilitiesKHR& c)
{
    g_emit->hex("flags", c.flags);
    g_emit->u64("maxLevel", c.maxLevel);
    g_emit->extent("codedPictureAlignment", c.codedPictureAlignment);
    g_emit->extent("maxTiles", c.maxTiles);
    g_emit->extent("minTileSize", c.minTileSize);
    g_emit->extent("maxTileSize", c.maxTileSize);
    g_emit->hex("superblockSizes", c.superblockSizes);
    g_emit->u64("maxSingleReferenceCount", c.maxSingleReferenceCount);
    g_emit->hex("singleReferenceNameMask", c.singleReferenceNameMask);
    g_emit->u64("maxUnidirectionalCompoundReferenceCount", c.maxUnidirectionalCompoundReferenceCount);
    g_emit->u64("maxUnidirectionalCompoundGroup1ReferenceCount", c.maxUnidirectionalCompoundGroup1ReferenceCount);
    g_emit->hex("unidirectionalCompoundReferenceNameMask", c.unidirectionalCompoundReferenceNameMask);
    g_emit->u64("maxBidirectionalCompoundReferenceCount", c.maxBidirectionalCompoundReferenceCount);
    g_emit->u64("maxBidirectionalCompoundGroup1ReferenceCount", c.maxBidirectionalCompoundGroup1ReferenceCount);
    g_emit->u64("maxBidirectionalCompoundGroup2ReferenceCount", c.maxBidirectionalCompoundGroup2ReferenceCount);
    g_emit->hex("bidirectionalCompoundReferenceNameMask", c.bidirectionalCompoundReferenceNameMask);
    g_emit->u64("maxTemporalLayerCount", c.maxTemporalLayerCount);
    g_emit->u64("maxSpatialLayerCount", c.maxSpatialLayerCount);
    g_emit->u64("maxOperatingPoints", c.maxOperatingPoints);
    g_emit->u64("minQIndex", c.minQIndex);
    g_emit->u64("maxQIndex", c.maxQIndex);
    g_emit->boolean("prefersGopRemainingFrames", c.prefersGopRemainingFrames);
    g_emit->boolean("requiresGopRemainingFrames", c.requiresGopRemainingFrames);
    g_emit->hex("stdSyntaxFlags", c.stdSyntaxFlags);
}

// "### Quality level" section (videoEncodeXXXQualityLevelPropertiesKHR) at level 0.
void emitQualityLevel(const VkVideoEncodeQualityLevelPropertiesKHR& q,
                      const VkVideoEncodeH264QualityLevelPropertiesKHR& c)
{
    EMIT_FLAGS("preferredRateControlMode", q.preferredRateControlMode, kRateControlModeNames);
    g_emit->u64("preferredRateControlLayerCount", q.preferredRateControlLayerCount);
    g_emit->hex("preferredRateControlFlags", c.preferredRateControlFlags);
    g_emit->u64("preferredGopFrameCount", c.preferredGopFrameCount);
    g_emit->u64("preferredIdrPeriod", c.preferredIdrPeriod);
    g_emit->u64("preferredConsecutiveBFrameCount", c.preferredConsecutiveBFrameCount);
    g_emit->u64("preferredTemporalLayerCount", c.preferredTemporalLayerCount);
    g_emit->i64("preferredConstantQp.qpI", c.preferredConstantQp.qpI);
    g_emit->i64("preferredConstantQp.qpP", c.preferredConstantQp.qpP);
    g_emit->i64("preferredConstantQp.qpB", c.preferredConstantQp.qpB);
    g_emit->u64("preferredMaxL0ReferenceCount", c.preferredMaxL0ReferenceCount);
    g_emit->u64("preferredMaxL1ReferenceCount", c.preferredMaxL1ReferenceCount);
    g_emit->boolean("preferredStdEntropyCodingModeFlag", c.preferredStdEntropyCodingModeFlag);
}

void emitQualityLevel(const VkVideoEncodeQualityLevelPropertiesKHR& q,
                      const VkVideoEncodeH265QualityLevelPropertiesKHR& c)
{
    EMIT_FLAGS("preferredRateControlMode", q.preferredRateControlMode, kRateControlModeNames);
    g_emit->u64("preferredRateControlLayerCount", q.preferredRateControlLayerCount);
    g_emit->hex("preferredRateControlFlags", c.preferredRateControlFlags);
    g_emit->u64("preferredGopFrameCount", c.preferredGopFrameCount);
    g_emit->u64("preferredIdrPeriod", c.preferredIdrPeriod);
    g_emit->u64("preferredConsecutiveBFrameCount", c.preferredConsecutiveBFrameCount);
    g_emit->u64("preferredSubLayerCount", c.preferredSubLayerCount);
    g_emit->i64("preferredConstantQp.qpI", c.preferredConstantQp.qpI);
    g_emit->i64("preferredConstantQp.qpP", c.preferredConstantQp.qpP);
    g_emit->i64("preferredConstantQp.qpB", c.preferredConstantQp.qpB);
    g_emit->u64("preferredMaxL0ReferenceCount", c.preferredMaxL0ReferenceCount);
    g_emit->u64("preferredMaxL1ReferenceCount", c.preferredMaxL1ReferenceCount);
}

void emitQualityLevel(const VkVideoEncodeQualityLevelPropertiesKHR& q,
                      const VkVideoEncodeAV1QualityLevelPropertiesKHR& c)
{
    EMIT_FLAGS("preferredRateControlMode", q.preferredRateControlMode, kRateControlModeNames);
    g_emit->u64("preferredRateControlLayerCount", q.preferredRateControlLayerCount);
    g_emit->hex("preferredRateControlFlags", c.preferredRateControlFlags);
    g_emit->u64("preferredGopFrameCount", c.preferredGopFrameCount);
    g_emit->u64("preferredKeyFramePeriod", c.preferredKeyFramePeriod);
    g_emit->u64("preferredConsecutiveBipredictiveFrameCount", c.preferredConsecutiveBipredictiveFrameCount);
    g_emit->u64("preferredTemporalLayerCount", c.preferredTemporalLayerCount);
    g_emit->u64("preferredConstantQIndex.intraQIndex", c.preferredConstantQIndex.intraQIndex);
    g_emit->u64("preferredConstantQIndex.predictiveQIndex", c.preferredConstantQIndex.predictiveQIndex);
    g_emit->u64("preferredConstantQIndex.bipredictiveQIndex", c.preferredConstantQIndex.bipredictiveQIndex);
    g_emit->u64("preferredMaxSingleReferenceCount", c.preferredMaxSingleReferenceCount);
    g_emit->hex("preferredSingleReferenceNameMask", c.preferredSingleReferenceNameMask);
    g_emit->u64("preferredMaxUnidirectionalCompoundReferenceCount", c.preferredMaxUnidirectionalCompoundReferenceCount);
    g_emit->u64("preferredMaxBidirectionalCompoundReferenceCount", c.preferredMaxBidirectionalCompoundReferenceCount);
}

// Each encode codec carries distinct codec-capabilities and quality-level structs, so the
// templated queries must be instantiated per codec. The per-codec dumps are selected by
// overload resolution on printCodecCaps()/printQualityLevel().
template <class CodecCaps, VkStructureType CodecCapsSType,
          class CodecQMapCaps, VkStructureType CodecQMapCapsSType,
          class CodecQualityLevel, VkStructureType CodecQualityLevelSType>
bool queryAndDumpEncode(const VulkanDeviceContext* vkDevCtx, VkVideoCodecOperationFlagBitsKHR codec,
                        const VkVideoCoreProfile& profile, const std::string& key)
{
    VkVideoCapabilitiesKHR videoCaps{};
    VkVideoEncodeCapabilitiesKHR encCaps{};
    CodecCaps codecCaps{};
    VkVideoEncodeQuantizationMapCapabilitiesKHR qmapCaps{};
    CodecQMapCaps codecQMapCaps{};
    VkVideoEncodeIntraRefreshCapabilitiesKHR intraRefreshCaps{};

    VkResult result = VulkanVideoCapabilities::GetVideoEncodeCapabilities<
        CodecCaps, CodecCapsSType, CodecQMapCaps, CodecQMapCapsSType>(
            vkDevCtx, profile, videoCaps, encCaps, codecCaps,
            qmapCaps, codecQMapCaps, intraRefreshCaps);
    if (result != VK_SUCCESS) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%s (0x%x)",
                 IsVideoUnsupportedResult(result) ? "unsupported, skipped" : "query failed", result);
        emitUnsupported(key, buf);
        return false;
    }

    beginCodec(key);

    emitProfiles(vkDevCtx, codec);

    g_emit->heading(3, "generic", "Generic");
    emitGenericCaps(videoCaps);
    g_emit->endSection(3);

    g_emit->heading(3, "encode", "Encode");
    emitEncodeCommon(encCaps, qmapCaps);
    g_emit->endSection(3);

    g_emit->heading(3, "intraRefresh", "Intra-refresh");
    emitIntraRefresh(intraRefreshCaps);
    g_emit->endSection(3);

    g_emit->heading(3, key, key);
    emitCodecCaps(codecCaps);
    g_emit->endSection(3);

    VkVideoEncodeQualityLevelPropertiesKHR qualityLevel{};
    CodecQualityLevel codecQualityLevel{};
    if (VulkanVideoCapabilities::GetPhysicalDeviceVideoEncodeQualityLevelProperties<
            CodecQualityLevel, CodecQualityLevelSType>(
                vkDevCtx, profile, /*qualityLevel*/ 0, qualityLevel, codecQualityLevel) == VK_SUCCESS) {
        g_emit->heading(3, "qualityLevel", "Quality level");
        emitQualityLevel(qualityLevel, codecQualityLevel);
        g_emit->endSection(3);
    }

    g_emit->endSection(2);
    return true;
}

void dumpEncodeCaps(const VulkanDeviceContext* vkDevCtx, VkVideoCodecOperationFlagBitsKHR codec)
{
    std::string key = codecKey(codec);

    int32_t queueFamily = vkDevCtx->GetVideoEncodeQueueFamilyIdx();
    VkVideoCodecOperationFlagsKHR supported =
        VulkanVideoCapabilities::GetSupportedCodecs(vkDevCtx, vkDevCtx->getPhysicalDevice(),
                                                    &queueFamily, VK_QUEUE_VIDEO_ENCODE_BIT_KHR);
    if ((supported & codec) == 0) {
        emitUnsupported(key, "not supported (no encode queue family)");
        return;
    }

    VkVideoCoreProfile profile = makeProfile(codec);
    switch (codec) {
    case VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR:
        queryAndDumpEncode<VkVideoEncodeH264CapabilitiesKHR,
                           VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_CAPABILITIES_KHR,
                           VkVideoEncodeH264QuantizationMapCapabilitiesKHR,
                           VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_QUANTIZATION_MAP_CAPABILITIES_KHR,
                           VkVideoEncodeH264QualityLevelPropertiesKHR,
                           VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_QUALITY_LEVEL_PROPERTIES_KHR>(
                               vkDevCtx, codec, profile, key);
        break;
    case VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR:
        queryAndDumpEncode<VkVideoEncodeH265CapabilitiesKHR,
                           VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_CAPABILITIES_KHR,
                           VkVideoEncodeH265QuantizationMapCapabilitiesKHR,
                           VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_QUANTIZATION_MAP_CAPABILITIES_KHR,
                           VkVideoEncodeH265QualityLevelPropertiesKHR,
                           VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_QUALITY_LEVEL_PROPERTIES_KHR>(
                               vkDevCtx, codec, profile, key);
        break;
    case VK_VIDEO_CODEC_OPERATION_ENCODE_AV1_BIT_KHR:
        queryAndDumpEncode<VkVideoEncodeAV1CapabilitiesKHR,
                           VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_CAPABILITIES_KHR,
                           VkVideoEncodeAV1QuantizationMapCapabilitiesKHR,
                           VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_QUANTIZATION_MAP_CAPABILITIES_KHR,
                           VkVideoEncodeAV1QualityLevelPropertiesKHR,
                           VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_QUALITY_LEVEL_PROPERTIES_KHR>(
                               vkDevCtx, codec, profile, key);
        break;
    default:
        break;
    }
}

// Page our stdout through a pager, modeled on gst-inspect: page only when stdout is a
// console, use $PAGER, and fall back to "less" (POSIX) / "more" (Windows). Implemented
// portably with popen()/_popen(): the pager's stdin pipe replaces our STDOUT_FILENO, so
// everything written to std::cout / printf flows into the pager.
#if defined(_WIN32)
#define POPEN  _popen
#define PCLOSE _pclose
#define DEFAULT_PAGER "more"
#else
#define POPEN  popen
#define PCLOSE pclose
#define DEFAULT_PAGER "less"
#endif

FILE* g_pager = nullptr;
int g_savedStdout = -1;

bool startPager()
{
    if (!isatty(fileno(stdout))) {
        return false; // output is redirected; don't interpose a pager
    }

    const char* pagerEnv = getenv("PAGER");
    const char* pager = (pagerEnv && pagerEnv[0]) ? pagerEnv : DEFAULT_PAGER;

    g_pager = POPEN(pager, "w");
    if (!g_pager) {
        return false;
    }

    // Redirect STDOUT_FILENO to the pager pipe, keeping the original to restore later.
    fflush(stdout);
    g_savedStdout = dup(fileno(stdout));
    dup2(fileno(g_pager), fileno(stdout));
    return true;
}

void finishPager()
{
    if (!g_pager) {
        return;
    }
    std::cout.flush();
    fflush(stdout);

    // Restore the real stdout, then close the pager pipe so it drains and exits.
    if (g_savedStdout >= 0) {
        dup2(g_savedStdout, fileno(stdout));
        close(g_savedStdout);
        g_savedStdout = -1;
    }
    PCLOSE(g_pager);
    g_pager = nullptr;
}

// Emit the full report for one already-initialized device: Device/Features plus the
// Decode/Encode codec sections. Heading levels are relative; the caller sets the base level
// (1 for a single device, 2 when nested as an element of a top-level devices array).
void dumpDevice(const VulkanDeviceContext* vkDevCtx, bool decodeOnly, bool encodeOnly)
{
    VkPhysicalDeviceProperties props{};
    vkDevCtx->GetPhysicalDeviceProperties(vkDevCtx->getPhysicalDevice(), &props);

    VkPhysicalDeviceDriverProperties driverProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driverProps};
    vkDevCtx->GetPhysicalDeviceProperties2(vkDevCtx->getPhysicalDevice(), &props2);

    char buf[64];

    g_emit->heading(1, "device", props.deviceName);
    if (g_emit->structured()) {
        g_emit->str("deviceName", props.deviceName);
    }
    g_emit->str("deviceType", deviceTypeName(props.deviceType));
    g_emit->str("driverName", driverProps.driverName);
    g_emit->str("driverInfo", driverProps.driverInfo);
    snprintf(buf, sizeof(buf), "0x%04x", props.vendorID);
    g_emit->str("vendorID", buf);
    snprintf(buf, sizeof(buf), "0x%04x", props.deviceID);
    g_emit->str("deviceID", buf);
    snprintf(buf, sizeof(buf), "%u.%u.%u",
             VK_API_VERSION_MAJOR(props.apiVersion),
             VK_API_VERSION_MINOR(props.apiVersion),
             VK_API_VERSION_PATCH(props.apiVersion));
    g_emit->str("apiVersion", buf);
    snprintf(buf, sizeof(buf), "0x%08x", props.driverVersion);
    g_emit->str("driverVersion", buf);

    static const VkVideoCodecOperationFlagBitsKHR decodeCodecs[] = {
        VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR,
        VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR,
        VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR,
        VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR,
    };
    static const VkVideoCodecOperationFlagBitsKHR encodeCodecs[] = {
        VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR,
        VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR,
        VK_VIDEO_CODEC_OPERATION_ENCODE_AV1_BIT_KHR,
    };

    // The codec dump helpers hardcode heading levels 2 (codec) / 3 (sub-sections). Nested
    // under a device they must shift one deeper, so bump the Markdown base level around them
    // (additively, so this composes with any base level dumpDevice itself is called under).
    const int base = g_emit->baseLevel();
    if (!encodeOnly && (vkDevCtx->GetVideoDecodeQueueFamilyIdx() >= 0)) {
        g_emit->heading(2, "decode", "Decode");
        g_emit->beginArray("codecs");
        g_emit->setBaseLevel(base + 1);
        for (VkVideoCodecOperationFlagBitsKHR codec : decodeCodecs) {
            dumpDecodeCaps(vkDevCtx, codec);
        }
        g_emit->setBaseLevel(base);
        g_emit->endArray();
        g_emit->endSection(2);
    }

    if (!decodeOnly && (vkDevCtx->GetVideoEncodeQueueFamilyIdx() >= 0)) {
        g_emit->heading(2, "encode", "Encode");
        g_emit->beginArray("codecs");
        g_emit->setBaseLevel(base + 1);
        for (VkVideoCodecOperationFlagBitsKHR codec : encodeCodecs) {
            dumpEncodeCaps(vkDevCtx, codec);
        }
        g_emit->setBaseLevel(base);
        g_emit->endArray();
        g_emit->endSection(2);
    }

    g_emit->endSection(1);
}

// Emit the top "Summary" section: one entry per video-capable device (after filtering) with
// its name, driver, and the VK_KHR_video_* extensions it advertises. JSON emits an
// "extensions" array per device; Markdown lists one device per row with the extensions joined.
void emitSummary(const VulkanDeviceContext* primary,
                 const std::vector<VkPhysicalDevice>& physicalDevices,
                 int32_t deviceId, const vk::DeviceUuidUtils& deviceUuid)
{
    const bool json = g_emit->structured();

    g_emit->heading(1, "summary", "Summary");
    g_emit->beginArray("devices");

    for (VkPhysicalDevice phys : physicalDevices) {
        if (!deviceHasVideoQueue(primary, phys) ||
            !deviceMatchesFilter(primary, phys, deviceId, deviceUuid)) {
            continue;
        }

        VkPhysicalDeviceDriverProperties driverProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driverProps};
        primary->GetPhysicalDeviceProperties2(phys, &p2);

        std::vector<std::string> exts = videoExtensions(primary, phys);

        if (json) {
            g_emit->heading(2, "device", p2.properties.deviceName);
            g_emit->str("deviceName", p2.properties.deviceName);
            g_emit->str("driverName", driverProps.driverName);
            g_emit->beginArray("extensions");
            for (const std::string& ext : exts) {
                g_emit->heading(3, "extension", ext);
                g_emit->str("name", ext);
                g_emit->endSection(3);
            }
            g_emit->endArray();
            g_emit->endSection(2);
        } else {
            // One sub-section per device with its video extensions as a single-column list.
            std::string title = std::string(p2.properties.deviceName) +
                                " (" + driverProps.driverName + ")";
            g_emit->heading(2, "device", title);
            for (const std::string& ext : exts) {
                g_emit->listItem("Video extensions", ext);
            }
            g_emit->endSection(2);
        }
    }

    g_emit->endArray();
    g_emit->endSection(1);
}

void printUsage(const char* prog)
{
    std::cout << "Usage: " << prog << " [options]\n"
              << "  Enumerate and print Vulkan Video encode/decode capabilities for a GPU.\n\n"
              << "Options:\n"
              << "  --deviceID, -deviceID <hex>      Hex ID of the device to be used\n"
              << "  --deviceUuid, -deviceUuid <uuid> UUID HEX string of the device to be used\n"
              << "  --decode-only                    Only query decode capabilities\n"
              << "  --encode-only                    Only query encode capabilities\n"
              << "  --verbose                        Verbose device initialization\n"
              << "  --json                           Emit JSON instead of Markdown (default: Markdown)\n"
              << "  --no-pager                       Do not page output (default: page through $PAGER on a console)\n"
              << "  -h, --help                       Show this help and exit\n";
}

} // namespace

int main(int argc, const char** argv)
{
    int32_t deviceId = -1;            // PCI device ID (hex) to match; -1 = any
    vk::DeviceUuidUtils deviceUuid;   // unset = no UUID filter
    bool decodeOnly = false;
    bool encodeOnly = false;
    bool verbose = false;
    bool usePager = true; // page by default; startPager() no-ops when stdout isn't a TTY
    bool jsonOutput = false;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        } else if (!strcmp(arg, "--deviceID") || !strcmp(arg, "-deviceID")) {
            if ((i + 1 >= argc) || (sscanf(argv[++i], "%x", &deviceId) != 1)) {
                fprintf(stderr, "Error: %s requires a hex device ID\n", arg);
                return EXIT_FAILURE;
            }
        } else if (!strcmp(arg, "--deviceUuid") || !strcmp(arg, "-deviceUuid")) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires a UUID string\n", arg);
                return EXIT_FAILURE;
            }
            if (deviceUuid.StringToUUID(argv[++i]) != VK_UUID_SIZE) {
                fprintf(stderr, "Error: invalid deviceUuid '%s'; "
                                "must be 16 hex values (32 digits, e.g. "
                                "12345678-1234-1234-1234-123456789abc)\n", argv[i]);
                return EXIT_FAILURE;
            }
        } else if (!strcmp(arg, "--decode-only")) {
            decodeOnly = true;
        } else if (!strcmp(arg, "--encode-only")) {
            encodeOnly = true;
        } else if (!strcmp(arg, "--verbose")) {
            verbose = true;
        } else if (!strcmp(arg, "--json")) {
            jsonOutput = true;
        } else if (!strcmp(arg, "--no-pager")) {
            usePager = false;
        } else {
            fprintf(stderr, "Error: unknown argument '%s'\n", arg);
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (decodeOnly && encodeOnly) {
        fprintf(stderr, "Error: --decode-only and --encode-only are mutually exclusive\n");
        return EXIT_FAILURE;
    }

    // Primary context: creates the Vulkan instance and enumerates physical devices. The
    // instance is shared with the per-device contexts below so it is created only once. It
    // needs no device extensions itself (it never creates a logical/physical device); each
    // per-device context requests exactly the video extensions that device exposes.
    VulkanDeviceContext primary;

    VkResult result = primary.InitVulkanDevice("vk-video-caps", VK_NULL_HANDLE, verbose);
    if (result != VK_SUCCESS) {
        if (IsVideoUnsupportedResult(result)) {
            fprintf(stderr, "Could not initialize the Vulkan device: incompatible driver\n");
            return VVS_EXIT_UNSUPPORTED;
        }
        fprintf(stderr, "Could not initialize the Vulkan device (0x%x)\n", result);
        return EXIT_FAILURE;
    }

    std::vector<VkPhysicalDevice> physicalDevices;
    if (vk::enumerate(&primary, primary.getInstance(), physicalDevices) != VK_SUCCESS ||
        physicalDevices.empty()) {
        fprintf(stderr, "No Vulkan physical devices found\n");
        return VVS_EXIT_UNSUPPORTED;
    }

    if (usePager) {
        startPager();
    }

    MarkdownEmitter markdown;
    // JsonEmitter writes "{" on construction, so only build it when JSON is requested.
    std::unique_ptr<JsonEmitter> json;
    if (jsonOutput) {
        json.reset(new JsonEmitter());
        g_emit = json.get();
    } else {
        g_emit = &markdown;
    }

    emitSummary(&primary, physicalDevices, deviceId, deviceUuid);

    // "# Devices" umbrella so each device is "## <name>" with "### Features" etc. underneath.
    // In JSON the beginArray below already names the "devices" key, so the heading is
    // Markdown-only; setBaseLevel(1) shifts each device's report one level deeper.
    if (!g_emit->structured()) {
        g_emit->heading(1, "devices", "Devices");
    }
    g_emit->beginArray("devices");
    g_emit->setBaseLevel(1);

    int dumped = 0;
    for (VkPhysicalDevice phys : physicalDevices) {
        // Skip non-video devices (e.g. llvmpipe) quietly, before InitPhysicalDevice would
        // log a missing-extension error. Apply the deviceID / deviceUuid selection filters
        // up front too, so non-matching devices are skipped silently.
        if (!deviceHasVideoQueue(&primary, phys) ||
            !deviceMatchesFilter(&primary, phys, deviceId, deviceUuid)) {
            continue;
        }

        // Detect which video queues this device actually has. InitPhysicalDevice requires
        // *every* bit in requestQueueTypes to be present (and logs an error otherwise), so we
        // request only the video queue families the device really exposes. This lets a
        // decode-only or encode-only device through instead of demanding both.
        int32_t qf = -1;
        bool hasDecode = VulkanVideoCapabilities::GetSupportedCodecs(
            &primary, phys, &qf, VK_QUEUE_VIDEO_DECODE_BIT_KHR) != VK_VIDEO_CODEC_OPERATION_NONE_KHR;
        qf = -1;
        bool hasEncode = VulkanVideoCapabilities::GetSupportedCodecs(
            &primary, phys, &qf, VK_QUEUE_VIDEO_ENCODE_BIT_KHR) != VK_VIDEO_CODEC_OPERATION_NONE_KHR;

        if (!hasDecode && !hasEncode) {
            continue; // video_queue extension present but no decode/encode queue family
        }

        VkQueueFlags requestQueueTypes = 0;
        if (hasDecode) {
            requestQueueTypes |= VK_QUEUE_VIDEO_DECODE_BIT_KHR;
        }
        if (hasEncode) {
            requestQueueTypes |= VK_QUEUE_VIDEO_ENCODE_BIT_KHR;
        }

        // Require only the queue extensions the device actually has: an encode-only device
        // (e.g. lavapipe) lacks VK_KHR_video_decode_queue, so requiring it unconditionally
        // would make InitPhysicalDevice reject the device and log an error.
        std::vector<const char*> deviceExtensions = { VK_KHR_VIDEO_QUEUE_EXTENSION_NAME };
        if (hasDecode) {
            deviceExtensions.push_back(VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME);
        }
        if (hasEncode) {
            deviceExtensions.push_back(VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME);
        }
        deviceExtensions.push_back(nullptr);

        // Fresh context per device (reusing the shared instance) so queue-family state never
        // leaks between devices.
        VulkanDeviceContext devCtx;
        devCtx.AddReqDeviceExtensions(deviceExtensions.data());
        if (devCtx.InitVulkanDevice("vk-video-caps", primary.getInstance(), verbose) != VK_SUCCESS) {
            continue;
        }
        VkResult devResult = devCtx.InitPhysicalDevice(
            deviceId, deviceUuid,
            requestQueueTypes,
            nullptr /* headless, no display shell */,
            VK_QUEUE_VIDEO_DECODE_BIT_KHR, VulkanDeviceContext::VIDEO_CODEC_OPERATIONS_DECODE,
            VK_QUEUE_VIDEO_ENCODE_BIT_KHR, VulkanDeviceContext::VIDEO_CODEC_OPERATIONS_ENCODE,
            phys, verbose, /*noDeviceFallback*/ true);

        if (devResult != VK_SUCCESS) {
            VkPhysicalDeviceProperties props{};
            primary.GetPhysicalDeviceProperties(phys, &props);
            emitDeviceSkip(props.deviceName, "skipped: device initialization failed");
            continue;
        }

        dumpDevice(&devCtx, decodeOnly, encodeOnly);
        ++dumped;
    }

    g_emit->setBaseLevel(0);
    g_emit->endArray();
    g_emit->finish();
    finishPager();

    if (dumped == 0) {
        fprintf(stderr, "No matching video-capable device found\n");
        return VVS_EXIT_UNSUPPORTED;
    }
    return EXIT_SUCCESS;
}
