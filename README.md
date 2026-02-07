# AgentAnalyze — Local Agentic RAG System

A locally-run AI agent that ingests personal documents and answers questions using multi-step reasoning. Everything runs on your machine — no data leaves your computer.

## What It Does

Drop files into a folder. Ask questions. The agent reads your documents, searches for relevant information, and generates answers — all using a local LLM running directly in Python with no external server.

**Two modes:**
- **Simple mode** — type a question for fast single-pass retrieval and answer
- **Agent mode** — prefix with `agent:` for multi-step reasoning where the model decides which tools to use, gathers information across documents, and synthesizes an answer

## Architecture

```
Documents (PDF, DOCX, TXT, CSV)
        ↓
   Text Extraction → Chunking (300-word segments)
        ↓
   Embedding → ChromaDB Vector Store
        ↓
   User Query → Semantic Search → Retrieved Chunks
        ↓
   LLM (Mistral 7B GGUF, loaded directly via llama-cpp-python)
        ↓
   [Simple Mode] → Single-pass answer
   [Agent Mode]  → ReAct loop: reason → pick tool → execute → repeat → answer
```

### Agent Tools
| Tool | Description |
|------|-------------|
| `SEARCH_DOCS(query)` | Semantic search across all indexed documents |
| `READ_FILE(filename)` | Read full content of a specific file |
| `LIST_FILES()` | List all indexed documents |
| `ANSWER(response)` | Deliver final answer to the user |

## Tech Stack

- **LLM**: Mistral 7B Instruct (Q4_K_M GGUF) via `llama-cpp-python`
- **Vector Store**: ChromaDB with model-generated embeddings
- **Document Parsing**: PyMuPDF (PDF), python-docx (DOCX), pandas (CSV)
- **Agent Pattern**: ReAct (Reason + Act) loop with tool selection
- **Hardware**: Runs entirely on CPU — no GPU required

## Setup

### 1. Install dependencies
```bash
pip install llama-cpp-python chromadb PyMuPDF python-docx pandas huggingface-hub
```

### 2. Download the model (~4.4 GB)
```bash
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir models
```

### 3. Add documents
Drop `.txt`, `.pdf`, `.docx`, or `.csv` files into the project folder.

### 4. Run
```bash
python agent_v3.py
```

## Usage

```
🔍 You: summarize my resume
🤖 (thinking...)
William Conner Barry is an MBA graduate from the University of Florida...
📎 Sources: resume.txt (88.5s)

🔍 You: agent: which of these job titles am I most qualified for?
🤖 Agent thinking...
  🔍 Step 1: Searching "job titles list" (45.2s)
  🔍 Step 2: Searching "technical skills experience" (52.1s)
💡 Answer (38.4s):
Based on your documents...
```

## Design Decisions

- **Direct GGUF loading** instead of Ollama server — eliminates HTTP overhead, faster on constrained hardware
- **Dual-mode operation** — simple RAG for quick answers, agent mode for complex multi-document questions
- **Model-generated embeddings** — uses the same Mistral model for both embeddings and generation, no separate embedding model needed
- **Swappable LLM** — architecture works identically with API-based models (Claude, GPT-4) by changing the `call_llm` function

## Performance (HP Pavilion, 32GB RAM, No GPU)

| Metric | Simple Mode | Agent Mode |
|--------|-------------|------------|
| Model load | ~3s | ~3s |
| Response time | 60-90s | 2-5 min |
| RAM usage | ~6-8 GB | ~6-8 GB |

## Project Evolution

This project went through three iterations:
1. **v1** (`agent.py`) — Basic RAG pipeline using Ollama server
2. **v2** (`agent_v2.py`) — Added ReAct agent loop with tool use via Ollama
3. **v3** (`agent_v3.py`) — Direct GGUF loading, no server dependency, dual-mode operation
