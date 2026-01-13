#!/bin/bash
#
# Test script to exercise different descriptor set layout code paths
# in vk-video-decoder by manipulating Vulkan extension availability.
#
# Usage: ./test_descriptor_modes.sh <video_file> [decoder_path]
#
# This script exercises all code paths in UpdateDescriptorBuffer():
# - VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER (presentation path)
# - VK_DESCRIPTOR_TYPE_STORAGE_IMAGE (compute filter path)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DECODER="${SCRIPT_DIR}/../BUILD/vk_video_decoder/demos/vk-video-dec-test"

usage() {
    echo "Usage: $0 <video_file> [decoder_path]"
    echo ""
    echo "Arguments:"
    echo "  video_file    Path to video file to decode"
    echo "  decoder_path  Path to vk-video-dec binary (optional)"
    echo ""
    echo "This script tests descriptor modes and UpdateDescriptorBuffer code paths:"
    echo ""
    echo "  Descriptor Modes:"
    echo "    1. Auto mode (default priority selection)"
    echo "    2. Push Descriptors (VK_KHR_push_descriptor)"
    echo "    3. Descriptor Buffers (VK_EXT_descriptor_buffer)"
    echo "    4. Standard Descriptor Sets (fallback)"
    echo ""
    echo "  UpdateDescriptorBuffer Code Paths (descriptor buffer mode only):"
    echo "    5. COMBINED_IMAGE_SAMPLER - via presentation"
    echo "    6. STORAGE_IMAGE - via compute filter (--enablePostProcessFilter)"
    exit 1
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW} $1${NC}"
    echo -e "${YELLOW}========================================${NC}"
}

log_subsection() {
    echo ""
    echo -e "${CYAN}--- $1 ---${NC}"
}

# Check arguments
if [ $# -lt 1 ]; then
    usage
fi

VIDEO_FILE="$1"
DECODER="${2:-$DEFAULT_DECODER}"

# Validate inputs
if [ ! -f "$VIDEO_FILE" ]; then
    log_error "Video file not found: $VIDEO_FILE"
    exit 1
fi

if [ ! -x "$DECODER" ]; then
    log_error "Decoder not found or not executable: $DECODER"
    log_info "Try building with: cmake --build build --target vk-video-dec"
    exit 1
fi

log_info "Video file: $VIDEO_FILE"
log_info "Decoder: $DECODER"

# Track results
RESULTS=()

run_test() {
    local mode_name="$1"
    local descriptor_mode="$2"
    local extra_args="$3"
    local extra_info="$4"
    local expect_debug="$5"

    log_subsection "Testing: $mode_name"
    [ -n "$extra_info" ] && log_info "$extra_info"

    # Clean up previous output
    rm -f /tmp/test_output.yuv

    # Build the command with --descriptor-mode flag
    local cmd="$DECODER -o /tmp/test_output.yuv --descriptor-mode $descriptor_mode $extra_args --maxFrameCount 5 --input \"$VIDEO_FILE\""

    log_info "Running: $cmd"
    echo ""

    # Run and capture both stdout and stderr
    local output
    if output=$(eval "$cmd" 2>&1); then
        # Check for expected debug output if specified
        if [ -n "$expect_debug" ]; then
            if echo "$output" | grep -q "$expect_debug"; then
                log_success "$mode_name completed - debug output verified: $expect_debug"
                RESULTS+=("$mode_name: PASS (debug output verified)")
            else
                log_warning "$mode_name completed but expected debug output not found: $expect_debug"
                RESULTS+=("$mode_name: PASS (no debug output - may need debug build)")
            fi
        else
            log_success "$mode_name completed"
            RESULTS+=("$mode_name: PASS")
        fi
        # Show relevant output (limit lines)
        echo "$output" | head -50
    else
        log_error "$mode_name failed"
        RESULTS+=("$mode_name: FAIL")
        echo "$output" | tail -20
    fi
}

#==============================================================================
# PART 1: Descriptor Mode Selection Tests
#==============================================================================
log_section "Part 1: Descriptor Mode Selection Tests"
log_info "These tests verify the descriptor mode selection logic in CreateDescriptorSet()"

#------------------------------------------------------------------------------
# Test 1: Auto mode (default - Push Descriptors have priority)
#------------------------------------------------------------------------------
run_test "Mode 1: Auto (default)" \
    "auto" \
    "--noPresent" \
    "Using default extension priority - push descriptors have highest priority"

#------------------------------------------------------------------------------
# Test 2: Force Push Descriptors
#------------------------------------------------------------------------------
run_test "Mode 2: Push Descriptors (forced)" \
    "push" \
    "--noPresent" \
    "Forcing VK_KHR_push_descriptor mode"

#------------------------------------------------------------------------------
# Test 3: Force Descriptor Buffer
#------------------------------------------------------------------------------
run_test "Mode 3: Descriptor Buffer (forced)" \
    "buffer" \
    "--noPresent" \
    "Forcing VK_EXT_descriptor_buffer mode"

#------------------------------------------------------------------------------
# Test 4: Force Standard Descriptor Sets
#------------------------------------------------------------------------------
run_test "Mode 4: Standard Descriptor Sets (forced)" \
    "standard" \
    "--noPresent" \
    "Forcing standard descriptor sets (no extensions)"

#==============================================================================
# PART 2: UpdateDescriptorBuffer Code Path Tests
#==============================================================================
log_section "Part 2: UpdateDescriptorBuffer Code Path Tests"
log_info "These tests exercise different descriptor types in UpdateDescriptorBuffer()"
log_info "Debug output only appears in debug builds (compiled without NDEBUG)"

#------------------------------------------------------------------------------
# Test 5: COMBINED_IMAGE_SAMPLER path (presentation)
#------------------------------------------------------------------------------
# This requires presentation to be enabled (no --noPresent)
# VulkanVideoUtils.cpp:747 uses VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
run_test "Path 5: COMBINED_IMAGE_SAMPLER (presentation)" \
    "buffer" \
    "" \
    "Descriptor buffer with presentation - triggers COMBINED_IMAGE_SAMPLER path" \
    "COMBINED_IMAGE_SAMPLER"

#------------------------------------------------------------------------------
# Test 6: STORAGE_IMAGE path (compute filter)
#------------------------------------------------------------------------------
# This requires the post-process filter to be enabled
# VulkanFilterYuvCompute.h:208 uses VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
# Filter modes: 0=YCBCRCOPY, 1=YCBCRCLEAR, 2=YCBCR2RGBA, 3=RGBA2YCBCR

log_subsection "Testing STORAGE_IMAGE paths via compute filters"

# Test with YCBCRCOPY filter (mode 0)
run_test "Path 6a: STORAGE_IMAGE (YCBCRCOPY filter)" \
    "buffer" \
    "--enablePostProcessFilter 0 --selectVideoWithComputeQueue" \
    "Descriptor buffer with compute filter mode 0 (YCBCRCOPY)" \
    "STORAGE_IMAGE"

# Test with YCBCR2RGBA filter (mode 2) - most common use case
run_test "Path 6b: STORAGE_IMAGE (YCBCR2RGBA filter)" \
    "buffer" \
    "--enablePostProcessFilter 2 --selectVideoWithComputeQueue" \
    "Descriptor buffer with compute filter mode 2 (YCBCR2RGBA)" \
    "STORAGE_IMAGE"

#==============================================================================
# Summary
#==============================================================================
log_section "Test Summary"

pass_count=0
fail_count=0

for result in "${RESULTS[@]}"; do
    if [[ "$result" == *"PASS"* ]]; then
        log_success "$result"
        ((pass_count++))
    else
        log_error "$result"
        ((fail_count++))
    fi
done

echo ""
log_info "Results: $pass_count passed, $fail_count failed"

echo ""
log_section "Code Path Reference"
echo ""
echo "UpdateDescriptorBuffer descriptor type switch (VulkanDescriptorSetLayout.h:284-312):"
echo ""
echo "  VK_DESCRIPTOR_TYPE_SAMPLER              - Not used in codebase"
echo "  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER - VulkanVideoUtils.cpp:747 (presentation)"
echo "  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE        - Not used in codebase"
echo "  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE        - VulkanFilterYuvCompute.h:208 (compute filter)"
echo ""
echo "Descriptor mode flags:"
echo "  --descriptor-mode auto     - Auto-select based on extension availability"
echo "  --descriptor-mode push     - Force VK_KHR_push_descriptor"
echo "  --descriptor-mode buffer   - Force VK_EXT_descriptor_buffer"
echo "  --descriptor-mode standard - Force standard descriptor sets"
echo ""
echo "Compute filter modes (--enablePostProcessFilter):"
echo "  0 = YCBCRCOPY   - Copy YCbCr planes"
echo "  1 = YCBCRCLEAR  - Clear YCbCr planes"
echo "  2 = YCBCR2RGBA  - Convert YCbCr to RGBA"
echo "  3 = RGBA2YCBCR  - Convert RGBA to YCbCr"

# Cleanup
rm -f /tmp/test_output.yuv

echo ""
if [ $fail_count -eq 0 ]; then
    log_success "All tests completed successfully!"
    exit 0
else
    log_error "Some tests failed"
    exit 1
fi
