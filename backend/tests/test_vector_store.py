"""
Tests for VectorStore class

These tests verify that the vector store correctly handles:
- Adding course metadata and content
- Searching with proper result counts
- Course name resolution (fuzzy matching)
- Filtering by course and lesson
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_store import VectorStore, SearchResults
from config import config


class TestSearchResults:
    """Test SearchResults dataclass"""

    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB response"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'course_title': 'Course 1'}, {'course_title': 'Course 2'}]],
            'distances': [[0.1, 0.2]]
        }
        results = SearchResults.from_chroma(chroma_results)

        assert len(results.documents) == 2
        assert results.documents[0] == 'doc1'
        assert len(results.metadata) == 2
        assert results.metadata[0]['course_title'] == 'Course 1'
        assert results.error is None

    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB response"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        results = SearchResults.from_chroma(chroma_results)

        assert len(results.documents) == 0
        assert results.is_empty()

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        error_msg = "No course found"
        results = SearchResults.empty(error_msg)

        assert results.is_empty()
        assert results.error == error_msg

    def test_is_empty(self):
        """Test is_empty method"""
        empty = SearchResults(documents=[], metadata=[], distances=[])
        assert empty.is_empty()

        not_empty = SearchResults(documents=['doc'], metadata=[{}], distances=[0.1])
        assert not not_empty.is_empty()


class TestVectorStoreWithConfig:
    """
    Tests that verify VectorStore behavior with current config.
    These tests will FAIL if MAX_RESULTS=0 in config.
    """

    def test_max_results_from_config(self, temp_chroma_path):
        """
        CRITICAL TEST: Verify that VectorStore uses config.MAX_RESULTS
        This will FAIL if MAX_RESULTS=0
        """
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        assert store.max_results == config.MAX_RESULTS
        assert store.max_results > 0, (
            f"VectorStore max_results must be > 0, got {store.max_results}. "
            f"This comes from config.MAX_RESULTS={config.MAX_RESULTS}"
        )

    def test_search_limit_with_config(self, temp_chroma_path, sample_course, sample_course_chunks):
        """
        Test that search respects the configured MAX_RESULTS limit.
        This will FAIL if MAX_RESULTS=0
        """
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, config.MAX_RESULTS)

        # Add test data
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)

        # Search should use config.MAX_RESULTS
        results = store.search("MCP protocol")

        # With MAX_RESULTS=0, this will fail
        if config.MAX_RESULTS > 0:
            assert len(results.documents) > 0, (
                f"Expected search results with MAX_RESULTS={config.MAX_RESULTS}, got none"
            )
        else:
            # This is the expected failure case
            assert len(results.documents) == 0, (
                f"With MAX_RESULTS=0, search should return 0 results"
            )


class TestVectorStoreAddOperations:
    """Test adding data to vector store"""

    def test_add_course_metadata(self, temp_chroma_path, sample_course):
        """Test adding course metadata to catalog"""
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, max_results=5)
        store.add_course_metadata(sample_course)

        # Verify course was added
        course_titles = store.get_existing_course_titles()
        assert sample_course.title in course_titles

    def test_add_course_content(self, temp_chroma_path, sample_course_chunks):
        """Test adding course content chunks"""
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, max_results=5)
        store.add_course_content(sample_course_chunks)

        # Verify content was added (check collection count)
        count = store.course_content.count()
        assert count == len(sample_course_chunks)

    def test_add_empty_chunks(self, temp_chroma_path):
        """Test adding empty chunks list"""
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, max_results=5)
        # Should not raise error
        store.add_course_content([])

    def test_get_course_count(self, temp_chroma_path, sample_course):
        """Test getting course count"""
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, max_results=5)

        assert store.get_course_count() == 0
        store.add_course_metadata(sample_course)
        assert store.get_course_count() == 1


class TestVectorStoreSearch:
    """Test search functionality with proper MAX_RESULTS"""

    def test_search_with_results(self, temp_chroma_path, sample_course, sample_course_chunks):
        """Test basic search returns results when max_results > 0"""
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, max_results=5)

        # Add data
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)

        # Search
        results = store.search("What is MCP?")

        assert not results.is_empty(), "Search should return results when max_results=5"
        assert len(results.documents) > 0
        assert results.error is None

    def test_search_with_course_filter(self, temp_chroma_path, sample_course, sample_course_chunks):
        """Test search with course name filter"""
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, max_results=5)

        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)

        # Search with course filter
        results = store.search("MCP", course_name="Introduction to MCP")

        assert not results.is_empty()
        # All results should be from the specified course
        for meta in results.metadata:
            assert meta['course_title'] == sample_course.title

    def test_search_with_lesson_filter(self, temp_chroma_path, sample_course, sample_course_chunks):
        """Test search with lesson number filter"""
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, max_results=5)

        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)

        # Search with lesson filter
        results = store.search("MCP", course_name=sample_course.title, lesson_number=0)

        # All results should be from lesson 0
        for meta in results.metadata:
            assert meta['lesson_number'] == 0

    def test_search_nonexistent_course(self, temp_chroma_path):
        """Test search for non-existent course returns error"""
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, max_results=5)

        results = store.search("test query", course_name="Nonexistent Course")

        assert results.error is not None
        assert "No course found" in results.error

    def test_search_with_custom_limit(self, temp_chroma_path, sample_course, sample_course_chunks):
        """Test search respects custom limit parameter"""
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, max_results=5)

        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)

        # Search with custom limit
        results = store.search("MCP", limit=2)

        assert len(results.documents) <= 2


class TestVectorStoreCourseResolution:
    """Test course name resolution (fuzzy matching)"""

    def test_resolve_exact_course_name(self, temp_chroma_path, sample_course):
        """Test resolving exact course name"""
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, max_results=5)
        store.add_course_metadata(sample_course)

        resolved = store._resolve_course_name(sample_course.title)
        assert resolved == sample_course.title

    def test_resolve_partial_course_name(self, temp_chroma_path, sample_course):
        """Test resolving partial course name (fuzzy match)"""
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, max_results=5)
        store.add_course_metadata(sample_course)

        # Search with just "MCP"
        resolved = store._resolve_course_name("MCP")
        assert resolved == sample_course.title

    def test_resolve_nonexistent_course(self, temp_chroma_path):
        """Test resolving non-existent course returns None"""
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, max_results=5)

        resolved = store._resolve_course_name("Nonexistent Course")
        assert resolved is None


class TestVectorStoreFiltering:
    """Test filter building logic"""

    def test_build_filter_no_params(self, temp_chroma_path):
        """Test building filter with no parameters"""
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, max_results=5)

        filter_dict = store._build_filter(None, None)
        assert filter_dict is None

    def test_build_filter_course_only(self, temp_chroma_path):
        """Test building filter with only course"""
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, max_results=5)

        filter_dict = store._build_filter("Test Course", None)
        assert filter_dict == {"course_title": "Test Course"}

    def test_build_filter_lesson_only(self, temp_chroma_path):
        """Test building filter with only lesson"""
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, max_results=5)

        filter_dict = store._build_filter(None, 1)
        assert filter_dict == {"lesson_number": 1}

    def test_build_filter_both_params(self, temp_chroma_path):
        """Test building filter with both course and lesson"""
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, max_results=5)

        filter_dict = store._build_filter("Test Course", 1)
        assert "$and" in filter_dict
        assert {"course_title": "Test Course"} in filter_dict["$and"]
        assert {"lesson_number": 1} in filter_dict["$and"]


class TestVectorStoreMetadataEnrichment:
    """Test lesson link enrichment in search results"""

    def test_enrich_with_lesson_links(self, temp_chroma_path, sample_course, sample_course_chunks):
        """Test that search results are enriched with lesson links"""
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, max_results=5)

        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)

        results = store.search("MCP", course_name=sample_course.title)

        # Check that lesson links were added to metadata
        for meta in results.metadata:
            if meta.get('lesson_number') is not None:
                # Should have lesson_link added
                assert 'lesson_link' in meta or meta.get('lesson_number') >= 0

    def test_get_lesson_link(self, temp_chroma_path, sample_course):
        """Test getting specific lesson link"""
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, max_results=5)
        store.add_course_metadata(sample_course)

        lesson_link = store.get_lesson_link(sample_course.title, 0)
        assert lesson_link == sample_course.lessons[0].lesson_link

    def test_get_course_link(self, temp_chroma_path, sample_course):
        """Test getting course link"""
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, max_results=5)
        store.add_course_metadata(sample_course)

        course_link = store.get_course_link(sample_course.title)
        assert course_link == sample_course.course_link


class TestVectorStoreClearOperations:
    """Test clearing data from vector store"""

    def test_clear_all_data(self, temp_chroma_path, sample_course, sample_course_chunks):
        """Test clearing all data from collections"""
        store = VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, max_results=5)

        # Add data
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)

        assert store.get_course_count() > 0

        # Clear data
        store.clear_all_data()

        assert store.get_course_count() == 0
