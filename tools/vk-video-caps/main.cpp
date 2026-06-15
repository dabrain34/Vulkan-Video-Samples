/*
 * Copyright 2026 NVIDIA Corporation.
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
    virtual void heading(int level, const std::string& key, const std::string& title) = 0;
    virtual void endSection(int level) = 0;

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

    virtual void finish() = 0;
};

Emitter* g_emit = nullptr;

#define EMIT_FLAGS(field, value, table) g_emit->flags(field, value, table, sizeof(table) / sizeof(table[0]))

// ----- Markdown backend -------------------------------------------------------------------

class MarkdownEmitter : public Emitter {
public:
    void heading(int level, const std::string&, const std::string& title) override
    {
        flushTable();
        std::cout << "\n" << std::string(level, '#') << " " << title << "\n";
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
    void finish() override { flushTable(); }

private:
    void addRow(const char* field, const std::string& value) { m_rows.emplace_back(field, value); }

    void flushTable()
    {
        if (m_rows.empty()) {
            return;
        }

        // Pad both columns to their widest cell so values line up under each other.
        std::string fieldHdr = "Field";
        std::string valueHdr = "Value";
        size_t fieldWidth = fieldHdr.size();
        size_t valueWidth = valueHdr.size();
        for (const auto& row : m_rows) {
            fieldWidth = std::max(fieldWidth, row.first.size());
            valueWidth = std::max(valueWidth, row.second.size());
        }

        auto pad = [](const std::string& s, size_t width) {
            return s + std::string(width - s.size(), ' ');
        };

        std::cout << "\n| " << pad(fieldHdr, fieldWidth) << " | " << pad(valueHdr, valueWidth) << " |\n";
        std::cout << "| " << std::string(fieldWidth, '-') << " | " << std::string(valueWidth, '-') << " |\n";
        for (const auto& row : m_rows) {
            std::cout << "| " << pad(row.first, fieldWidth) << " | " << pad(row.second, valueWidth) << " |\n";
        }
        m_rows.clear();
    }

    std::vector<std::pair<std::string, std::string>> m_rows;
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
            // Array element: anonymous object carrying the codec name as a field.
            std::cout << "\n" << indent() << "{";
            m_needComma = false;
            m_inArray.push_back(false);
            field_("codec");
            std::cout << quote(key);
        } else {
            std::cout << "\n" << indent() << quote(key) << ": {";
            m_needComma = false;
            m_inArray.push_back(false);
        }
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

// Emit the codec section heading then a "not supported" note (used when a codec is absent).
void emitUnsupported(int level, const std::string& key, const std::string& title,
                     const std::string& reason)
{
    g_emit->heading(level, key, title);
    g_emit->note(reason);
    g_emit->endSection(level);
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

bool videoMaintenance2Supported(const VulkanDeviceContext* vkDevCtx)
{
    VkPhysicalDeviceVideoMaintenance2FeaturesKHR maint2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_MAINTENANCE_2_FEATURES_KHR, nullptr, VK_FALSE};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &maint2};
    vkDevCtx->GetPhysicalDeviceFeatures2(vkDevCtx->getPhysicalDevice(), &features);
    return maint2.videoMaintenance2 == VK_TRUE;
}

bool videoEncodeQuantizationMapSupported(const VulkanDeviceContext* vkDevCtx)
{
    VkPhysicalDeviceVideoEncodeQuantizationMapFeaturesKHR qpMap{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_ENCODE_QUANTIZATION_MAP_FEATURES_KHR, nullptr, VK_FALSE};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &qpMap};
    vkDevCtx->GetPhysicalDeviceFeatures2(vkDevCtx->getPhysicalDevice(), &features);
    return qpMap.videoEncodeQuantizationMap == VK_TRUE;
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

// Build the representative 8-bit 4:2:0 profile for a codec operation.
// Each codec needs its own factory; CreateDecodeProfile asserts on H265/VP9 only.
VkVideoCoreProfile makeProfile(VkVideoCodecOperationFlagBitsKHR codec)
{
    const VkVideoChromaSubsamplingFlagsKHR chroma = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
    const VkVideoComponentBitDepthFlagsKHR depth8 = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;

    switch (codec) {
    case VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR:
        return VkVideoCoreProfile::CreateDecodeH264Profile(chroma, depth8, depth8,
                   STD_VIDEO_H264_PROFILE_IDC_MAIN,
                   VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_PROGRESSIVE_KHR);
    case VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR:
        return VkVideoCoreProfile::CreateDecodeProfile(codec, chroma, depth8, depth8,
                   STD_VIDEO_H265_PROFILE_IDC_MAIN);
    case VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR:
        return VkVideoCoreProfile::CreateDecodeAV1Profile(chroma, depth8, depth8,
                   STD_VIDEO_AV1_PROFILE_MAIN, /*filmGrainSupport*/ false);
    case VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR:
        return VkVideoCoreProfile::CreateDecodeProfile(codec, chroma, depth8, depth8,
                   STD_VIDEO_VP9_PROFILE_0);
    case VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR:
        return VkVideoCoreProfile::CreateEncodeProfile(codec, chroma, depth8, depth8,
                   STD_VIDEO_H264_PROFILE_IDC_MAIN, VK_VIDEO_ENCODE_TUNING_MODE_DEFAULT_KHR);
    case VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR:
        return VkVideoCoreProfile::CreateEncodeProfile(codec, chroma, depth8, depth8,
                   STD_VIDEO_H265_PROFILE_IDC_MAIN, VK_VIDEO_ENCODE_TUNING_MODE_DEFAULT_KHR);
    case VK_VIDEO_CODEC_OPERATION_ENCODE_AV1_BIT_KHR:
        return VkVideoCoreProfile::CreateEncodeProfile(codec, chroma, depth8, depth8,
                   STD_VIDEO_AV1_PROFILE_MAIN, VK_VIDEO_ENCODE_TUNING_MODE_DEFAULT_KHR);
    default:
        return VkVideoCoreProfile();
    }
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
        emitUnsupported(2, key, key, "not supported (no decode queue family)");
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
        emitUnsupported(2, key, key, buf);
        return;
    }

    g_emit->heading(2, key, key);

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
bool queryAndDumpEncode(const VulkanDeviceContext* vkDevCtx, const VkVideoCoreProfile& profile,
                        const std::string& key)
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
        emitUnsupported(2, key, key, buf);
        return false;
    }

    g_emit->heading(2, key, key);

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
        emitUnsupported(2, key, key, "not supported (no encode queue family)");
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
                               vkDevCtx, profile, key);
        break;
    case VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR:
        queryAndDumpEncode<VkVideoEncodeH265CapabilitiesKHR,
                           VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_CAPABILITIES_KHR,
                           VkVideoEncodeH265QuantizationMapCapabilitiesKHR,
                           VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_QUANTIZATION_MAP_CAPABILITIES_KHR,
                           VkVideoEncodeH265QualityLevelPropertiesKHR,
                           VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_QUALITY_LEVEL_PROPERTIES_KHR>(
                               vkDevCtx, profile, key);
        break;
    case VK_VIDEO_CODEC_OPERATION_ENCODE_AV1_BIT_KHR:
        queryAndDumpEncode<VkVideoEncodeAV1CapabilitiesKHR,
                           VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_CAPABILITIES_KHR,
                           VkVideoEncodeAV1QuantizationMapCapabilitiesKHR,
                           VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_QUANTIZATION_MAP_CAPABILITIES_KHR,
                           VkVideoEncodeAV1QualityLevelPropertiesKHR,
                           VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_QUALITY_LEVEL_PROPERTIES_KHR>(
                               vkDevCtx, profile, key);
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

void printUsage(const char* prog)
{
    std::cout << "Usage: " << prog << " [options]\n"
              << "  Enumerate and print Vulkan Video encode/decode capabilities for a GPU.\n\n"
              << "Options:\n"
              << "  -d, --device <id>   Select physical device by index (default: auto)\n"
              << "      --decode-only   Only query decode capabilities\n"
              << "      --encode-only   Only query encode capabilities\n"
              << "      --verbose       Verbose device initialization\n"
              << "      --json          Emit JSON instead of Markdown (default: Markdown)\n"
              << "      --no-pager      Do not page output (default: page through $PAGER on a console)\n"
              << "  -h, --help          Show this help and exit\n";
}

} // namespace

int main(int argc, const char** argv)
{
    int32_t deviceId = -1;
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
        } else if (!strcmp(arg, "-d") || !strcmp(arg, "--device")) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg);
                return EXIT_FAILURE;
            }
            deviceId = atoi(argv[++i]);
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

    static const char* const requiredDeviceExtensions[] = {
        VK_KHR_VIDEO_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME,
        nullptr
    };

    VulkanDeviceContext vkDevCtx;
    vkDevCtx.AddReqDeviceExtensions(requiredDeviceExtensions);

    VkResult result = vkDevCtx.InitVulkanDevice("vk-video-caps", VK_NULL_HANDLE, verbose);
    if (result != VK_SUCCESS) {
        if (IsVideoUnsupportedResult(result)) {
            fprintf(stderr, "Could not initialize the Vulkan device: incompatible driver\n");
            return VVS_EXIT_UNSUPPORTED;
        }
        fprintf(stderr, "Could not initialize the Vulkan device (0x%x)\n", result);
        return EXIT_FAILURE;
    }

    result = vkDevCtx.InitPhysicalDevice(deviceId, vk::DeviceUuidUtils(),
                                         (VK_QUEUE_VIDEO_DECODE_BIT_KHR | VK_QUEUE_VIDEO_ENCODE_BIT_KHR),
                                         nullptr /* headless, no display shell */,
                                         VK_QUEUE_VIDEO_DECODE_BIT_KHR,
                                         VulkanDeviceContext::VIDEO_CODEC_OPERATIONS_DECODE,
                                         VK_QUEUE_VIDEO_ENCODE_BIT_KHR,
                                         VulkanDeviceContext::VIDEO_CODEC_OPERATIONS_ENCODE,
                                         VK_NULL_HANDLE, verbose, /*noDeviceFallback*/ false);
    if (result != VK_SUCCESS) {
        if (IsVideoUnsupportedResult(result)) {
            fprintf(stderr, "No physical device with video queue support found\n");
            return VVS_EXIT_UNSUPPORTED;
        }
        fprintf(stderr, "Could not initialize the Vulkan physical device (0x%x)\n", result);
        return EXIT_FAILURE;
    }

    VkPhysicalDeviceProperties props{};
    vkDevCtx.GetPhysicalDeviceProperties(vkDevCtx.getPhysicalDevice(), &props);

    VkPhysicalDeviceDriverProperties driverProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driverProps};
    vkDevCtx.GetPhysicalDeviceProperties2(vkDevCtx.getPhysicalDevice(), &props2);

    char buf[64];

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

    g_emit->heading(1, "device", "Device");
    g_emit->str("deviceName", props.deviceName);
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

    g_emit->heading(2, "features", "Features");
    g_emit->boolean("videoMaintenance1",
                    VulkanVideoCapabilities::GetVideoMaintenance1FeatureSupported(&vkDevCtx));
    g_emit->boolean("videoMaintenance2", videoMaintenance2Supported(&vkDevCtx));
    g_emit->boolean("videoEncodeIntraRefresh",
                    VulkanVideoCapabilities::IsVideoEncodeIntraRefreshSupported(&vkDevCtx));
    g_emit->boolean("videoEncodeQuantizationMap", videoEncodeQuantizationMapSupported(&vkDevCtx));
    g_emit->endSection(2);

    g_emit->endSection(1);

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

    if (!encodeOnly && (vkDevCtx.GetVideoDecodeQueueFamilyIdx() >= 0)) {
        g_emit->heading(1, "decode", "Decode");
        g_emit->beginArray("codecs");
        for (VkVideoCodecOperationFlagBitsKHR codec : decodeCodecs) {
            dumpDecodeCaps(&vkDevCtx, codec);
        }
        g_emit->endArray();
        g_emit->endSection(1);
    }

    if (!decodeOnly && (vkDevCtx.GetVideoEncodeQueueFamilyIdx() >= 0)) {
        g_emit->heading(1, "encode", "Encode");
        g_emit->beginArray("codecs");
        for (VkVideoCodecOperationFlagBitsKHR codec : encodeCodecs) {
            dumpEncodeCaps(&vkDevCtx, codec);
        }
        g_emit->endArray();
        g_emit->endSection(1);
    }

    g_emit->finish();
    finishPager();
    return EXIT_SUCCESS;
}
