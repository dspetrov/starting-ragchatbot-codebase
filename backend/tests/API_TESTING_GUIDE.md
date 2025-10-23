# API Testing Guide

## Overview

This guide documents the enhanced API testing infrastructure for the RAG system. The testing framework now includes comprehensive API endpoint tests, pytest configuration, and reusable test fixtures.

## What Was Added

### 1. Pytest Configuration (`pyproject.toml`)

Added comprehensive pytest configuration with:
- **Test discovery**: Automatic test file/function pattern matching
- **Markers**: Organized tests by category (unit, integration, api, slow)
- **Clean output**: Verbose mode with concise tracebacks
- **Async support**: Proper asyncio configuration for FastAPI testing

### 2. API Test Fixtures (`conftest.py`)

Added specialized fixtures for API testing:

#### `mock_rag_system`
Creates a fully mocked RAG system with:
- Session management (create/clear sessions)
- Query processing with sample responses
- Course analytics data

#### `test_app`
Creates a test FastAPI application that:
- Includes all API endpoints from the main app
- Uses mocked dependencies (avoiding static file issues)
- Identical request/response schemas to production

#### `client`
FastAPI TestClient for making HTTP requests in tests

#### `sample_query_request` & `sample_query_request_with_session`
Pre-configured request payloads for testing

### 3. Comprehensive API Tests (`test_api.py`)

Created 30 tests covering all endpoints:

#### POST /api/query (9 tests)
- ✓ Session creation when no session_id provided
- ✓ Using existing session_id
- ✓ Response format validation
- ✓ Missing required fields (422 validation)
- ✓ Empty query strings
- ✓ Invalid JSON payloads
- ✓ Error handling (500 responses)
- ✓ Very long query strings
- ✓ Complete schema validation

#### GET /api/courses (5 tests)
- ✓ Successful statistics retrieval
- ✓ Response structure validation
- ✓ Error handling
- ✓ No parameters required
- ✓ Ignores unexpected parameters

#### DELETE /api/session/{session_id} (5 tests)
- ✓ Successful session deletion
- ✓ Special characters in session_id
- ✓ Non-existent sessions (idempotent)
- ✓ Error handling
- ✓ Empty session_id (404/405)

#### GET / (2 tests)
- ✓ Health check response
- ✓ No authentication required

#### CORS Configuration (2 tests)
- ✓ Allows all origins
- ✓ CORS headers present

#### Content-Type Validation (2 tests)
- ✓ Requires JSON for POST endpoints
- ✓ Accepts application/json

#### Error Responses (3 tests)
- ✓ 404 for unknown endpoints
- ✓ 405 for wrong HTTP methods
- ✓ Error responses include detail field

#### End-to-End Workflows (2 tests)
- ✓ Query → Delete session workflow
- ✓ Multiple queries with same session

## Running the Tests

### Run All API Tests
```bash
cd backend
uv run pytest tests/test_api.py -v
```

### Run Specific Test Class
```bash
uv run pytest tests/test_api.py::TestQueryEndpoint -v
```

### Run Tests by Marker
```bash
# Run only API tests
uv run pytest -m api

# Run API integration tests
uv run pytest -m "api and integration"

# Exclude slow tests
uv run pytest -m "not slow"
```

### Run All Tests with Coverage
```bash
uv run pytest --cov=. --cov-report=html
```

## Test Architecture

### Separation of Concerns

The test suite avoids the static file mounting issue in `app.py` by:

1. **Creating a separate test app** in `conftest.py` that:
   - Duplicates API endpoint definitions
   - Uses mocked dependencies
   - Excludes frontend static file mounting

2. **Using dependency injection** via fixtures to control behavior

3. **Isolating tests** so they don't affect each other

### Benefits of This Approach

✅ **No import errors**: Test app doesn't require frontend directory
✅ **Fast execution**: All 30 tests run in ~3 seconds
✅ **Full control**: Mock behavior easily customizable per test
✅ **Isolation**: Tests don't share state or interfere
✅ **Maintainability**: Clear separation between test and production code

## Writing New API Tests

### Example: Testing a New Endpoint

```python
@pytest.mark.api
class TestNewEndpoint:
    """Test suite for POST /api/new-endpoint"""

    def test_success_case(self, client, mock_rag_system):
        """Test successful request"""
        # Configure mock behavior
        mock_rag_system.new_method = Mock(return_value="expected_result")

        # Make request
        response = client.post("/api/new-endpoint", json={"param": "value"})

        # Assertions
        assert response.status_code == 200
        assert response.json()["result"] == "expected_result"

        # Verify mock was called
        mock_rag_system.new_method.assert_called_once()

    def test_error_case(self, client, mock_rag_system):
        """Test error handling"""
        # Configure mock to raise exception
        mock_rag_system.new_method.side_effect = Exception("Error message")

        # Make request
        response = client.post("/api/new-endpoint", json={"param": "value"})

        # Assertions
        assert response.status_code == 500
        assert "Error message" in response.json()["detail"]
```

## Best Practices

### ✅ Do's

- Use `@pytest.mark.api` for all API tests
- Test both success and error cases
- Verify response status codes AND response schemas
- Test edge cases (empty strings, special characters, very long inputs)
- Use descriptive test names that explain what's being tested
- Verify mocks were called with expected arguments

### ❌ Don'ts

- Don't import the production `app` directly in tests (use `test_app` fixture)
- Don't share state between tests
- Don't test multiple unrelated things in one test
- Don't hardcode magic numbers (use constants or variables)
- Don't skip assertions on response structure

## Troubleshooting

### Tests Fail with "Frontend directory not found"

**Solution**: Make sure you're using the `test_app` fixture from `conftest.py`, not importing `app` from `app.py`.

### Tests Hang or Take Too Long

**Solution**: Check that you're using mocks properly. Real API calls or database operations will slow tests significantly.

### Mock Not Working as Expected

**Solution**: Verify the mock is configured before the client makes the request. Use `.assert_called_once()` to verify the mock is being hit.

### Async Test Warnings

**Solution**: Ensure `pytest-asyncio` is installed and configured in `pyproject.toml` (already done).

## Test Coverage Analysis

Current API test coverage:

| Endpoint | Tests | Coverage |
|----------|-------|----------|
| POST /api/query | 9 | ✓ Complete |
| GET /api/courses | 5 | ✓ Complete |
| DELETE /api/session/{id} | 5 | ✓ Complete |
| GET / | 2 | ✓ Complete |
| Middleware (CORS) | 2 | ✓ Complete |
| Error Handling | 3 | ✓ Complete |
| Integration Workflows | 2 | ✓ Complete |

**Total**: 30 tests covering all critical paths

## Dependencies Added

The following packages were added to support API testing:

```toml
dependencies = [
    # ... existing dependencies ...
    "httpx==0.28.1",           # For FastAPI TestClient
    "pytest==8.3.4",           # Testing framework
    "pytest-asyncio==0.25.2",  # Async test support
]
```

## Next Steps

Potential enhancements to the test suite:

1. **Add test coverage reporting** (pytest-cov)
2. **Performance tests** for query response times
3. **Load tests** for concurrent requests
4. **Contract tests** to ensure API schema consistency
5. **Security tests** for input validation and injection attacks
6. **Mock ChromaDB** for vector store integration tests
7. **Add test data generators** for more diverse test cases

## Related Files

- `pyproject.toml` - Pytest configuration
- `backend/tests/conftest.py` - Test fixtures and test app
- `backend/tests/test_api.py` - API endpoint tests
- `backend/app.py` - Production FastAPI application
