"""
Unit tests for test suite filtering functionality.

Tests filter_test_suite() method and pattern matching behavior.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.libs.video_test_config_base import (
    CodecType,
    DenyFilter,
    DenyRule,
)
from tests.libs.video_test_framework_base import VulkanVideoTestFrameworkBase


@dataclass
class MockSample:
    """Mock sample for testing filter_test_suite"""
    name: str
    codec: CodecType
    _display_name: Optional[str] = None

    @property
    def display_name(self) -> str:
        """Return display name with prefix if not explicitly set"""
        if self._display_name:
            return self._display_name
        return f"decode_{self.name}"


class MockFramework(VulkanVideoTestFrameworkBase):
    """Mock framework for testing filter_test_suite without file dependencies"""

    def __init__(self, deny_rules=None):
        # Skip parent __init__ to avoid file system dependencies
        self._deny_rules = deny_rules or []
        self._options = {
            'deny_filter': DenyFilter.ENABLED,
        }

    @property
    def deny_filter(self) -> DenyFilter:
        return self._options['deny_filter']

    def set_deny_filter(self, mode: DenyFilter):
        self._options['deny_filter'] = mode

    def check_resources(self, auto_download=True, test_configs=None):
        return True

    def create_test_suite(self, codec_filter=None, test_pattern=None):
        return []

    def run_single_test(self, config):
        pass


class TestFilterByCodec:
    """Tests for codec filtering"""

    def test_filter_single_codec(self):
        """Test filtering by single codec"""
        framework = MockFramework()
        samples = [
            MockSample(name="test1", codec=CodecType.H264),
            MockSample(name="test2", codec=CodecType.H265),
            MockSample(name="test3", codec=CodecType.H264),
            MockSample(name="test4", codec=CodecType.AV1),
        ]

        result = framework.filter_test_suite(samples, codec_filter="h264")
        assert len(result) == 2
        assert all(s.codec == CodecType.H264 for s in result)

    def test_filter_no_codec_returns_all(self):
        """Test that no codec filter returns all samples"""
        framework = MockFramework()
        samples = [
            MockSample(name="test1", codec=CodecType.H264),
            MockSample(name="test2", codec=CodecType.H265),
        ]

        result = framework.filter_test_suite(samples, codec_filter=None)
        assert len(result) == 2


class TestFilterByPattern:
    """Tests for pattern matching"""

    def test_filter_by_display_name_prefix(self):
        """Test filtering by display name with prefix"""
        framework = MockFramework()
        samples = [
            MockSample(name="h264_test", codec=CodecType.H264),
            MockSample(name="h265_test", codec=CodecType.H265),
        ]

        result = framework.filter_test_suite(samples, test_pattern="decode_h264_*")
        assert len(result) == 1
        assert result[0].name == "h264_test"

    def test_filter_by_base_name(self):
        """Test filtering by base name without prefix"""
        framework = MockFramework()
        samples = [
            MockSample(name="h264_test", codec=CodecType.H264),
            MockSample(name="h265_test", codec=CodecType.H265),
        ]

        result = framework.filter_test_suite(samples, test_pattern="h264_*")
        assert len(result) == 1
        assert result[0].name == "h264_test"

    def test_filter_exact_match_base_name(self):
        """Test exact match by base name"""
        framework = MockFramework()
        samples = [
            MockSample(name="h264_test", codec=CodecType.H264),
            MockSample(name="h264_test_extended", codec=CodecType.H264),
        ]

        result = framework.filter_test_suite(samples, test_pattern="h264_test")
        assert len(result) == 1
        assert result[0].name == "h264_test"

    def test_filter_exact_match_display_name(self):
        """Test exact match by display name"""
        framework = MockFramework()
        samples = [
            MockSample(name="h264_test", codec=CodecType.H264),
        ]

        result = framework.filter_test_suite(samples, test_pattern="decode_h264_test")
        assert len(result) == 1

    def test_filter_wildcard_matches_multiple(self):
        """Test wildcard pattern matching multiple samples"""
        framework = MockFramework()
        samples = [
            MockSample(name="av1_basic_8bit", codec=CodecType.AV1),
            MockSample(name="av1_basic_10bit", codec=CodecType.AV1),
            MockSample(name="av1_advanced_8bit", codec=CodecType.AV1),
            MockSample(name="h264_basic", codec=CodecType.H264),
        ]

        result = framework.filter_test_suite(samples, test_pattern="av1_*")
        assert len(result) == 3
        assert all(s.codec == CodecType.AV1 for s in result)


class TestFilterByDenyList:
    """Tests for deny list filtering"""

    def test_deny_mode_enabled_skips_denied(self):
        """Test ENABLED mode skips denied tests"""
        deny_rules = [
            DenyRule(name="denied_test", test_type="decode", format="vvs")
        ]
        framework = MockFramework(deny_rules=deny_rules)
        framework.set_deny_filter(DenyFilter.ENABLED)

        samples = [
            MockSample(name="denied_test", codec=CodecType.H264),
            MockSample(name="allowed_test", codec=CodecType.H264),
        ]

        result = framework.filter_test_suite(
            samples, deny_filter=DenyFilter.ENABLED,
            test_format="vvs", test_type="decode"
        )
        assert len(result) == 1
        assert result[0].name == "allowed_test"

    def test_deny_mode_denied_only_denied(self):
        """Test DENIED mode runs only denied tests"""
        deny_rules = [
            DenyRule(name="denied_test", test_type="decode", format="vvs")
        ]
        framework = MockFramework(deny_rules=deny_rules)

        samples = [
            MockSample(name="denied_test", codec=CodecType.H264),
            MockSample(name="allowed_test", codec=CodecType.H264),
        ]

        result = framework.filter_test_suite(
            samples, deny_filter=DenyFilter.DENIED,
            test_format="vvs", test_type="decode"
        )
        assert len(result) == 1
        assert result[0].name == "denied_test"

    def test_deny_mode_all_includes_everything(self):
        """Test ALL mode includes both denied and non-denied"""
        deny_rules = [
            DenyRule(name="denied_test", test_type="decode", format="vvs")
        ]
        framework = MockFramework(deny_rules=deny_rules)

        samples = [
            MockSample(name="denied_test", codec=CodecType.H264),
            MockSample(name="allowed_test", codec=CodecType.H264),
        ]

        result = framework.filter_test_suite(
            samples, deny_filter=DenyFilter.ALL,
            test_format="vvs", test_type="decode"
        )
        assert len(result) == 2

    def test_exact_match_overrides_deny_in_enabled_mode(self):
        """Test that exact match can run denied test in ENABLED mode"""
        deny_rules = [
            DenyRule(name="denied_test", test_type="decode", format="vvs")
        ]
        framework = MockFramework(deny_rules=deny_rules)

        samples = [
            MockSample(name="denied_test", codec=CodecType.H264),
            MockSample(name="allowed_test", codec=CodecType.H264),
        ]

        # Exact match by base name should override deny
        result = framework.filter_test_suite(
            samples, test_pattern="denied_test",
            deny_filter=DenyFilter.ENABLED,
            test_format="vvs", test_type="decode"
        )
        assert len(result) == 1
        assert result[0].name == "denied_test"


class TestFilterCombined:
    """Tests for combined filtering"""

    def test_codec_and_pattern_combined(self):
        """Test codec and pattern filters combined"""
        framework = MockFramework()
        samples = [
            MockSample(name="h264_basic", codec=CodecType.H264),
            MockSample(name="h264_advanced", codec=CodecType.H264),
            MockSample(name="h265_basic", codec=CodecType.H265),
        ]

        result = framework.filter_test_suite(
            samples, codec_filter="h264", test_pattern="*_basic"
        )
        assert len(result) == 1
        assert result[0].name == "h264_basic"

    def test_empty_samples_list(self):
        """Test filtering empty samples list"""
        framework = MockFramework()
        result = framework.filter_test_suite([])
        assert result == []

    def test_no_matches_returns_empty(self):
        """Test that no matches returns empty list"""
        framework = MockFramework()
        samples = [
            MockSample(name="test1", codec=CodecType.H264),
        ]

        result = framework.filter_test_suite(
            samples, test_pattern="nonexistent_*"
        )
        assert result == []
