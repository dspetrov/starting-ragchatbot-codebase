"""
Tests for configuration validation

These tests ensure all configuration values are properly set and valid.
This will catch the MAX_RESULTS=0 bug immediately.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config


class TestConfigValidation:
    """Test suite for configuration validation"""

    def test_max_results_is_positive(self):
        """
        CRITICAL TEST: Verify MAX_RESULTS is greater than 0.
        This test will FAIL with the current config (MAX_RESULTS=0)
        """
        assert config.MAX_RESULTS > 0, (
            f"MAX_RESULTS must be > 0, got {config.MAX_RESULTS}. "
            f"A value of 0 causes vector store to return no search results!"
        )

    def test_max_results_is_reasonable(self):
        """Verify MAX_RESULTS is in a reasonable range"""
        assert config.MAX_RESULTS >= 1, "MAX_RESULTS should be at least 1"
        assert config.MAX_RESULTS <= 20, "MAX_RESULTS should not exceed 20 (performance consideration)"

    def test_anthropic_api_key_is_set(self):
        """Verify Anthropic API key is configured"""
        assert config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY must be set in .env file"
        assert len(config.ANTHROPIC_API_KEY) > 0, "ANTHROPIC_API_KEY cannot be empty"
        # Basic format check (Anthropic keys start with 'sk-ant-')
        if not config.ANTHROPIC_API_KEY.startswith("test_"):  # Allow test keys in testing
            assert config.ANTHROPIC_API_KEY.startswith("sk-ant-"), (
                "ANTHROPIC_API_KEY should start with 'sk-ant-'"
            )

    def test_chunk_size_is_valid(self):
        """Verify chunk size is appropriate"""
        assert config.CHUNK_SIZE > 0, "CHUNK_SIZE must be positive"
        assert config.CHUNK_SIZE >= 100, "CHUNK_SIZE should be at least 100 characters"
        assert config.CHUNK_SIZE <= 2000, "CHUNK_SIZE should not exceed 2000 characters"

    def test_chunk_overlap_is_valid(self):
        """Verify chunk overlap is valid"""
        assert config.CHUNK_OVERLAP >= 0, "CHUNK_OVERLAP must be non-negative"
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE, (
            "CHUNK_OVERLAP must be less than CHUNK_SIZE"
        )

    def test_max_history_is_valid(self):
        """Verify conversation history limit is reasonable"""
        assert config.MAX_HISTORY >= 0, "MAX_HISTORY must be non-negative"
        assert config.MAX_HISTORY <= 10, "MAX_HISTORY should not exceed 10 (token consideration)"

    def test_embedding_model_is_set(self):
        """Verify embedding model is configured"""
        assert config.EMBEDDING_MODEL, "EMBEDDING_MODEL must be set"
        assert isinstance(config.EMBEDDING_MODEL, str), "EMBEDDING_MODEL must be a string"

    def test_anthropic_model_is_set(self):
        """Verify Anthropic model is configured"""
        assert config.ANTHROPIC_MODEL, "ANTHROPIC_MODEL must be set"
        assert isinstance(config.ANTHROPIC_MODEL, str), "ANTHROPIC_MODEL must be a string"

    def test_chroma_path_is_set(self):
        """Verify ChromaDB path is configured"""
        assert config.CHROMA_PATH, "CHROMA_PATH must be set"
        assert isinstance(config.CHROMA_PATH, str), "CHROMA_PATH must be a string"


class TestConfigRecommendations:
    """Tests for recommended configuration values"""

    def test_recommended_max_results(self):
        """Test that MAX_RESULTS is set to a recommended value"""
        recommended_range = range(3, 11)  # 3-10 is a good range
        assert config.MAX_RESULTS in recommended_range, (
            f"Recommended MAX_RESULTS is between 3-10, got {config.MAX_RESULTS}"
        )

    def test_recommended_chunk_size(self):
        """Test that CHUNK_SIZE is set to a recommended value"""
        recommended_range = range(500, 1501)  # 500-1500 is typical
        assert config.CHUNK_SIZE in recommended_range, (
            f"Recommended CHUNK_SIZE is between 500-1500, got {config.CHUNK_SIZE}"
        )

    def test_recommended_chunk_overlap(self):
        """Test that CHUNK_OVERLAP is a reasonable percentage of CHUNK_SIZE"""
        overlap_percentage = (config.CHUNK_OVERLAP / config.CHUNK_SIZE) * 100
        assert 5 <= overlap_percentage <= 25, (
            f"CHUNK_OVERLAP should be 5-25% of CHUNK_SIZE, got {overlap_percentage:.1f}%"
        )
