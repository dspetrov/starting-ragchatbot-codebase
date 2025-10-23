# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Retrieval-Augmented Generation (RAG) system** for querying course materials. It combines semantic search (ChromaDB), AI generation (Claude), and a FastAPI backend with vanilla JS frontend to create an intelligent Q&A assistant for educational content.

**Key Architecture Pattern**: Tool-based RAG where Claude uses a search tool to retrieve relevant course content before generating responses, rather than having search results automatically injected.

## Development Commands

### Running the Application

**Quick Start** (Git Bash required on Windows):
```bash
./run.sh
```

**Manual Start**:
```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

The application runs at:
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs (FastAPI auto-generated Swagger UI)

### Dependency Management

```bash
# Install dependencies
uv sync

# Install dependencies with dev tools (includes black, flake8, isort, mypy)
uv sync --extra dev

# Add new dependency
# Edit pyproject.toml, then run:
uv sync
```

### Code Quality Tools

The project uses several code quality tools to maintain consistent formatting and code standards:

**Formatting:**
```bash
# Format code with black and isort
./format.sh

# Check formatting without modifying files
./format-check.sh
```

**Linting:**
```bash
# Run flake8 linter
./lint.sh
```

**Complete Quality Check:**
```bash
# Run all quality checks (formatting + linting)
./quality-check.sh
```

**Optional Type Checking:**
```bash
# Run mypy type checker (optional, for gradual typing adoption)
./type-check.sh
```

**Configuration:**
- Black: Line length 100, configured in `pyproject.toml` under `[tool.black]`
- isort: Black-compatible profile, configured in `pyproject.toml` under `[tool.isort]`
- flake8: Line length 100, E203/W503 ignored, configured in `.flake8`
- mypy: Relaxed configuration for gradual typing, configured in `pyproject.toml` under `[tool.mypy]`

**Pre-commit Workflow:**
1. Make code changes
2. Run `./format.sh` to auto-format
3. Run `./quality-check.sh` to verify all checks pass
4. Commit your changes

### Environment Setup

Create `.env` file in root directory:
```bash
ANTHROPIC_API_KEY=your_api_key_here
```

## High-Level Architecture

### Request Flow (User Query → AI Response)

```
1. User Query → Frontend (script.js)
   ↓
2. POST /api/query → FastAPI (app.py)
   ↓
3. RAGSystem.query() → rag_system.py
   ├─ Retrieves conversation history from SessionManager
   ├─ Constructs prompt with tool definitions
   └─ Calls AIGenerator.generate_response()
       ↓
4. Claude receives prompt + search tool definition
   ├─ If Claude decides to search:
   │   ├─ Outputs tool_use block with search parameters
   │   ├─ ToolManager.execute_tool() calls CourseSearchTool
   │   │   ├─ VectorStore._resolve_course_name() (fuzzy course matching)
   │   │   ├─ VectorStore.search() with filters
   │   │   └─ Returns SearchResults with metadata
   │   ├─ Tool results appended to conversation
   │   └─ Claude called again with search results
   └─ Claude generates final answer
       ↓
5. Response returned with answer + sources + session_id
   ↓
6. Frontend displays answer (markdown rendered) + collapsible sources
```

### Two-Collection ChromaDB Strategy

**Why two collections?** Separating course metadata from content enables:
- Fast fuzzy course name resolution ("MCP" → "MCP: Build Rich-Context AI Apps")
- Filtered semantic search within specific courses/lessons
- Better metadata organization

```
ChromaDB Client (./chroma_db)
├─ course_catalog Collection
│  ├─ Documents: Course titles (for semantic matching)
│  ├─ IDs: course.title
│  └─ Metadata: instructor, course_link, lessons_json
│
└─ course_content Collection
   ├─ Documents: Text chunks (800 chars, 100 char overlap)
   ├─ IDs: f"{course_title}_{chunk_index}"
   └─ Metadata: course_title, lesson_number, chunk_index
```

**Embedding Model**: `all-MiniLM-L6-v2` (384-dim, optimized for semantic similarity)

### Core Backend Components

| Module | Purpose | Key Responsibilities |
|--------|---------|---------------------|
| `app.py` | FastAPI entry point | REST endpoints (`/api/query`, `/api/courses`), CORS, static file serving |
| `rag_system.py` | Main orchestrator | Coordinates document processing, search, AI generation, sessions |
| `vector_store.py` | ChromaDB wrapper | Embedding, semantic search, course name resolution, metadata management |
| `ai_generator.py` | Claude API client | Tool-based prompting, streaming support, response generation |
| `document_processor.py` | Document parser | Extracts course metadata, chunks text with sentence-aware splitting |
| `search_tools.py` | Tool definitions | CourseSearchTool schema, ToolManager for execution |
| `session_manager.py` | Conversation state | Maintains per-session history (limited to MAX_HISTORY=2 turns) |
| `config.py` | Configuration | Env vars, constants (chunk size, model name, etc.) |
| `models.py` | Data models | Pydantic schemas for Course, Lesson, CourseChunk |

### Key Configuration Constants (config.py)

```python
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"  # Latest Sonnet
CHUNK_SIZE = 800           # Characters per chunk
CHUNK_OVERLAP = 100        # Overlap for context continuity
MAX_RESULTS = 5            # Search results returned
MAX_HISTORY = 2            # Conversation turns remembered (token optimization)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_db"
```

## Adding Course Documents

### Document Format

Plain text files in `docs/` folder with specific structure:

```
Course Title: Full Course Name
Course Link: https://...
Course Instructor: Instructor Name

Lesson 0: Introduction
Lesson Link: https://...
[Lesson content...]

Lesson 1: Next Lesson
Lesson Link: https://...
[Lesson content...]
```

### Automatic Loading

Documents in `docs/` are automatically loaded on startup (see `app.py:startup_event()`). The system:
- Checks for existing courses to avoid duplicates
- Chunks text with sentence-aware splitting
- Stores in both ChromaDB collections

### Manual Loading

```python
# In Python console or script:
from rag_system import RAGSystem
from config import config

rag = RAGSystem(config)

# Add single document
course, chunks = rag.add_course_document("path/to/document.txt")

# Add all documents from folder (clear_existing=True wipes DB first)
total_courses, total_chunks = rag.add_course_folder("docs/", clear_existing=False)
```

## Important Implementation Details

### Tool-Based Search Pattern

Unlike traditional RAG (auto-inject search results), this system uses Claude's tool use:
- **Advantage**: Claude decides when to search vs. answer from knowledge
- **Advantage**: Can handle general questions without unnecessary searches
- **Constraint**: System prompt instructs "one search per query maximum"

### Session Management

- Sessions stored **in-memory** (not persisted across restarts)
- Session IDs generated client-side and maintained in browser
- History limited to `MAX_HISTORY=2` turns to control token costs
- **Production consideration**: Replace SessionManager with Redis/DB for scalability

### Prompt Caching Strategy

Claude API prompt caching is NOT currently implemented but would benefit:
- System prompts (repeated across requests)
- Tool definitions (repeated across requests)
- Long conversation histories

### Chunking Algorithm (document_processor.py)

- Splits on sentence boundaries using regex (handles abbreviations like "Dr.", "Mr.")
- Target size: 800 characters with 100 character overlap
- First chunk of each lesson prefixed with `"Lesson X content:"` for context
- **Why overlap?** Prevents context loss at chunk boundaries

## Frontend Architecture

**Stack**: Vanilla HTML/CSS/JavaScript (no frameworks, no build process)

**Key Files**:
- `frontend/index.html` - DOM structure
- `frontend/script.js` - Client logic, API calls, markdown rendering
- `frontend/style.css` - Dark theme styling

**External Dependencies** (CDN):
- `marked.js` - Markdown rendering

**API Integration**:
```javascript
// Query endpoint
POST /api/query
Request: { query: string, session_id: string | null }
Response: { answer: string, sources: string[], session_id: string }

// Course stats endpoint
GET /api/courses
Response: { total_courses: number, course_titles: string[] }
```

## Common Development Patterns

### Adding New Tools

1. Define tool schema in `search_tools.py` (extend `Tool` base class)
2. Implement `execute()` method with actual functionality
3. Register with `ToolManager` in `rag_system.py:__init__()`
4. Tool automatically available to Claude in all queries

### Modifying System Prompt

Edit `ai_generator.py:_build_system_prompt()`. Current prompt emphasizes:
- Use search tool only for course-specific questions
- One search per query maximum
- No meta-commentary
- Synthesize search results into direct answers

### Adjusting Search Behavior

- **Course name matching**: `vector_store.py:_resolve_course_name()` uses semantic search on course_catalog collection
- **Result ranking**: ChromaDB's default cosine similarity
- **Max results**: Change `config.MAX_RESULTS` (currently 5)
- **Search filters**: Modify `search_tools.py:CourseSearchTool.execute()` to build ChromaDB `where` filters

### Conversation History Management

- **Token cost optimization**: `config.MAX_HISTORY=2` limits remembered turns
- **Tradeoff**: Lower values save tokens but reduce context
- **Session cleanup**: Currently no automatic cleanup (in-memory storage)

## Testing & Validation

**No automated tests currently exist.** Manual testing approaches:

1. **API Testing**: Use FastAPI Swagger UI at `/docs`
2. **Course Loading**: Check startup logs for successful document processing
3. **Search Quality**: Query via frontend, inspect sources returned
4. **Conversation Context**: Test multi-turn conversations to verify history

## Known Limitations & Scaling Considerations

1. **Session Persistence**: In-memory only (lost on restart)
2. **Concurrent Users**: Limited by single-process uvicorn (use gunicorn + workers for production)
3. **Vector Search Scale**: ChromaDB suitable for ~10k documents; larger datasets may need Pinecone/Weaviate
4. **API Latency**: Two Claude calls per tool-based search (~1-2s additional latency)
5. **Conversation History**: MAX_HISTORY=2 trades context for cost (adjust based on use case)
6. **Windows Compatibility**: Use Git Bash to run `run.sh` (PowerShell/CMD not supported)

## Troubleshooting

**"No courses loaded" on startup:**
- Check `docs/` folder exists with properly formatted `.txt` files
- Verify document format matches expected structure (see "Document Format")
- Check startup logs for parsing errors

**Empty search results:**
- Verify ChromaDB persistence (`./chroma_db` should exist)
- Try exact course title instead of fuzzy match
- Check if documents were successfully chunked (startup logs)

**API key errors:**
- Ensure `.env` file exists in root directory
- Verify `ANTHROPIC_API_KEY=sk-ant-...` format (no quotes)
- Restart server after adding `.env`

**Port 8000 already in use:**
- Change port: `uv run uvicorn app:app --reload --port XXXX`
- Update frontend API calls in `script.js` if using non-8000 port
