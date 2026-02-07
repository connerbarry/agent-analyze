"""
AgentAnalyze v3 - Agentic RAG Agent (Direct GGUF, no Ollama)
Loads the model directly in Python via llama-cpp-python. No server needed.
"""

import os
import sys
import time
import hashlib
import json
import re
from pathlib import Path

# Document readers
import fitz  # PyMuPDF
from docx import Document as DocxDocument
import pandas as pd

# Vector store
import chromadb

# Local LLM
from llama_cpp import Llama

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

WATCH_FOLDER = r"C:\Users\conne\Desktop\AgentAnalyze"
CHROMA_DB_PATH = os.path.join(WATCH_FOLDER, ".vectorstore")
COLLECTION_NAME = "documents_v3"

# Path to your GGUF model file
MODEL_PATH = os.path.join(WATCH_FOLDER, "models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# Model settings
N_CTX = 4096          # context window
N_THREADS = 6         # CPU threads (leave a couple free for your OS)
MAX_TOKENS = 300      # max response length

CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".csv", ".md"}

MAX_AGENT_STEPS = 4


# ─────────────────────────────────────────────
# DOCUMENT READING
# ─────────────────────────────────────────────

def read_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(filepath: str) -> str:
    text_parts = []
    with fitz.open(filepath) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)

def read_docx(filepath: str) -> str:
    doc = DocxDocument(filepath)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def read_csv(filepath: str) -> str:
    df = pd.read_csv(filepath)
    lines = [f"Columns: {', '.join(df.columns.tolist())}"]
    for _, row in df.iterrows():
        lines.append(" | ".join([f"{col}: {val}" for col, val in row.items()]))
    return "\n".join(lines)

READERS = {
    ".txt": read_txt,
    ".md": read_txt,
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".csv": read_csv,
}

def read_file(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    reader = READERS.get(ext)
    if reader is None:
        return ""
    try:
        return reader(filepath)
    except Exception as e:
        print(f"  ⚠ Error reading {filepath}: {e}")
        return ""


# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────

def chunk_text(text: str, filename: str) -> list[dict]:
    words = text.split()
    chunks = []
    if len(words) == 0:
        return chunks
    start = 0
    chunk_id = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)
        chunks.append({
            "id": f"{filename}__chunk_{chunk_id}",
            "text": chunk_text_str,
            "metadata": {
                "source": filename,
                "chunk_index": chunk_id,
                "word_count": len(chunk_words),
            }
        })
        start += CHUNK_SIZE - CHUNK_OVERLAP
        chunk_id += 1
    return chunks


# ─────────────────────────────────────────────
# SIMPLE EMBEDDINGS (no Ollama dependency)
# ─────────────────────────────────────────────

class SimpleEmbeddingFunction:
    """
    Uses the loaded Llama model to generate embeddings.
    Falls back to a basic word-frequency approach if embedding isn't supported.
    """
    def __init__(self, llm: Llama):
        self.llm = llm
        self.use_llm_embed = True
        # Test if the model supports embeddings
        try:
            test = self.llm.embed("test")
            if test is None or len(test) == 0:
                self.use_llm_embed = False
        except Exception:
            self.use_llm_embed = False

        if self.use_llm_embed:
            print("  📐 Using model embeddings")
        else:
            print("  📐 Using lightweight text embeddings")

    def name(self) -> str:
        return "simple_embedding"

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    def __call__(self, input: list[str]) -> list[list[float]]:
        if self.use_llm_embed:
            results = []
            for text in input:
                embed = self.llm.embed(text)
                # Flatten if nested (some models return [[...]])
                if embed and isinstance(embed[0], list):
                    embed = embed[0]
                results.append(embed)
            return results
        else:
            return [self._simple_embed(text) for text in input]

    def _simple_embed(self, text: str) -> list[float]:
        """Basic bag-of-words style embedding as fallback."""
        words = text.lower().split()
        # Use a fixed vocabulary size for consistent dimensions
        dim = 384
        vector = [0.0] * dim
        for word in words:
            h = hash(word) % dim
            vector[h] += 1.0
        # Normalize
        magnitude = sum(v * v for v in vector) ** 0.5
        if magnitude > 0:
            vector = [v / magnitude for v in vector]
        return vector


# ─────────────────────────────────────────────
# INGESTION
# ─────────────────────────────────────────────

def scan_folder(folder: str) -> list[str]:
    files = []
    for item in os.listdir(folder):
        if item.startswith("."):
            continue
        full_path = os.path.join(folder, item)
        if os.path.isfile(full_path) and Path(item).suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(full_path)
    return files

def ingest_documents(collection) -> int:
    files = scan_folder(WATCH_FOLDER)
    if not files:
        print(f"\n📂 No supported files found in {WATCH_FOLDER}")
        return 0

    print(f"\n📂 Found {len(files)} file(s) in {WATCH_FOLDER}")

    all_ids = collection.get()["ids"]
    if all_ids:
        collection.delete(ids=all_ids)

    total_chunks = 0
    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"  📄 Reading: {filename}")
        text = read_file(filepath)
        if not text.strip():
            continue
        chunks = chunk_text(text, filename)
        if not chunks:
            continue
        collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )
        total_chunks += len(chunks)
        print(f"     → {len(chunks)} chunk(s) indexed")

    print(f"\n✅ Ingestion complete: {total_chunks} total chunks from {len(files)} file(s)")
    return total_chunks


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

def tool_search_docs(collection, query: str) -> str:
    results = collection.query(query_texts=[query], n_results=3)
    if not results["ids"][0]:
        return "No relevant results found."
    output = []
    for i in range(len(results["ids"][0])):
        source = results["metadatas"][0][i]["source"]
        text = results["documents"][0][i]
        output.append(f"[{source}]: {text[:500]}")
    return "\n\n".join(output)

def tool_list_files() -> str:
    files = scan_folder(WATCH_FOLDER)
    if not files:
        return "No files found."
    names = [os.path.basename(f) for f in files]
    return "Indexed files: " + ", ".join(names)

def tool_read_full_file(filename: str) -> str:
    filepath = os.path.join(WATCH_FOLDER, filename)
    if not os.path.exists(filepath):
        return f"File '{filename}' not found."
    text = read_file(filepath)
    words = text.split()
    if len(words) > 600:
        return " ".join(words[:600]) + f"\n\n[...truncated, {len(words)} total words]"
    return text


# ─────────────────────────────────────────────
# LLM — direct llama-cpp-python call
# ─────────────────────────────────────────────

def call_llm(llm: Llama, messages: list[dict]) -> str:
    """Call the local model directly — no server, no API."""
    try:
        # Build the prompt in Mistral Instruct format
        prompt = build_mistral_prompt(messages)

        output = llm(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.4,
            stop=["</s>", "[INST]"],
            echo=False,
        )

        return output["choices"][0]["text"].strip()

    except Exception as e:
        return f"TOOL: ANSWER\nINPUT: Error generating response: {e}"


def build_mistral_prompt(messages: list[dict]) -> str:
    """Convert chat messages into Mistral Instruct format."""
    prompt = ""
    system_msg = ""
    first_inst = True

    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg["content"]
        elif msg["role"] == "user":
            if first_inst:
                if system_msg:
                    prompt += f"<s>[INST] {system_msg}\n\n{msg['content']} [/INST]"
                else:
                    prompt += f"<s>[INST] {msg['content']} [/INST]"
                first_inst = False
                system_msg = ""
            else:
                prompt += f"[INST] {msg['content']} [/INST]"
        elif msg["role"] == "assistant":
            prompt += f" {msg['content']}</s>"

    return prompt


# ─────────────────────────────────────────────
# AGENT CORE — ReAct loop
# ─────────────────────────────────────────────

TOOL_DESCRIPTIONS = """You have these tools:

1. SEARCH_DOCS(query) - Search documents for information.
2. LIST_FILES() - List all indexed files.
3. READ_FILE(filename) - Read a specific file's content.
4. ANSWER(response) - Give your final answer. You MUST end with this.

Format:
TOOL: tool_name
INPUT: your input

Always SEARCH_DOCS or READ_FILE before using ANSWER."""


def parse_tool_call(response: str) -> tuple[str, str]:
    tool_match = re.search(r'TOOL:\s*(\w+)', response, re.IGNORECASE)
    input_match = re.search(r'INPUT:\s*(.*)', response, re.IGNORECASE | re.DOTALL)

    if tool_match:
        tool_name = tool_match.group(1).upper()
        tool_input = input_match.group(1).strip() if input_match else ""
        return tool_name, tool_input

    return "ANSWER", response


def run_agent(llm: Llama, collection, user_query: str):
    """Run the ReAct agent loop."""

    system_prompt = f"""You answer questions about the user's documents using tools. Always search before answering.

{TOOL_DESCRIPTIONS}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    print(f"\n🤖 Agent thinking...\n")

    for step in range(MAX_AGENT_STEPS):
        start_time = time.time()
        response = call_llm(llm, messages)
        elapsed = time.time() - start_time

        if not response.strip():
            print(f"  ⚠ Empty response at step {step + 1} ({elapsed:.1f}s), retrying...")
            continue

        tool_name, tool_input = parse_tool_call(response)

        if tool_name == "ANSWER":
            print(f"💡 Answer ({elapsed:.1f}s):\n")
            print(tool_input)
            return

        elif tool_name == "SEARCH_DOCS":
            print(f"  🔍 Step {step + 1}: Searching \"{tool_input}\" ({elapsed:.1f}s)")
            result = tool_search_docs(collection, tool_input)
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Result:\n{result}\n\nWhat next? Use another tool or ANSWER."})

        elif tool_name == "LIST_FILES":
            print(f"  📂 Step {step + 1}: Listing files ({elapsed:.1f}s)")
            result = tool_list_files()
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Result:\n{result}\n\nWhat next? Use another tool or ANSWER."})

        elif tool_name == "READ_FILE":
            print(f"  📖 Step {step + 1}: Reading \"{tool_input}\" ({elapsed:.1f}s)")
            result = tool_read_full_file(tool_input)
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Result:\n{result}\n\nWhat next? Use another tool or ANSWER."})

        else:
            print(f"  ⚠ Step {step + 1}: Unknown tool '{tool_name}', treating as answer")
            print(response)
            return

    # Max steps reached
    print(f"\n⚠ Max steps reached. Forcing answer...")
    messages.append({"role": "user", "content": "Give your ANSWER now based on what you found."})
    response = call_llm(llm, messages)
    _, final_answer = parse_tool_call(response)
    print(f"\n💡 Answer:\n")
    print(final_answer)


# ─────────────────────────────────────────────
# V1 MODE — simple RAG (faster, single call)
# ─────────────────────────────────────────────

def run_simple(llm: Llama, collection, user_query: str):
    """Simple single-pass RAG — faster than agent mode."""
    results = collection.query(query_texts=[user_query], n_results=3)

    if not results["ids"][0]:
        print("No relevant documents found.")
        return

    context_parts = []
    sources = set()
    for i in range(len(results["ids"][0])):
        source = results["metadatas"][0][i]["source"]
        text = results["documents"][0][i]
        context_parts.append(f"[{source}]: {text[:500]}")
        sources.add(source)

    context = "\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": "Answer concisely based on the context. Mention source files."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"},
    ]

    print("\n🤖 (thinking...)", flush=True)
    start = time.time()
    response = call_llm(llm, messages)
    elapsed = time.time() - start

    print(f"\n{response}")
    print(f"\n📎 Sources: {', '.join(sorted(sources))} ({elapsed:.1f}s)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AgentAnalyze v3 — Direct GGUF (No Server)")
    print("=" * 60)

    # Check model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model not found at: {MODEL_PATH}")
        print(f"   Download it with:")
        print(f"   huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir {os.path.join(WATCH_FOLDER, 'models')}")
        sys.exit(1)

    # Load model
    print(f"\n🧠 Loading model (this takes 30-60 seconds on first run)...")
    print(f"   File: {MODEL_PATH}")
    start = time.time()

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=0,      # no GPU
        embedding=True,       # enable embeddings
        verbose=False,
    )

    elapsed = time.time() - start
    print(f"   ✅ Model loaded in {elapsed:.1f}s")

    # Set up vector store
    print("\n🔧 Setting up vector store...")
    embed_fn = SimpleEmbeddingFunction(llm)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    # Clear old collection to avoid embedding conflicts
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )

    total = ingest_documents(collection)
    if total == 0:
        print(f"\n⚠ No documents found. Add files to {WATCH_FOLDER} and restart.")
        sys.exit(0)

    # Mode selection
    print("\n" + "=" * 60)
    print("  Ready! Two modes available:")
    print("  • Just type a question → fast single-pass RAG (v1 style)")
    print("  • Type 'agent: your question' → agentic multi-step (v2 style)")
    print("  • 'reload' to re-scan, 'files' to list, 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            query = input("\n🔍 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break
        if query.lower() == "reload":
            print("\n🔄 Re-scanning folder...")
            ingest_documents(collection)
            continue
        if query.lower() == "files":
            files = scan_folder(WATCH_FOLDER)
            print(f"\n📂 {len(files)} file(s) indexed:")
            for f in files:
                print(f"   • {os.path.basename(f)}")
            continue

        # Check if agent mode requested
        if query.lower().startswith("agent:"):
            actual_query = query[6:].strip()
            if actual_query:
                run_agent(llm, collection, actual_query)
            else:
                print("Usage: agent: your question here")
        else:
            run_simple(llm, collection, query)


if __name__ == "__main__":
    main()