from __future__ import annotations

import argparse
import json
import os
import queue
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

try:
    import chromadb
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
except ImportError as exc:
    chromadb = None  # type: ignore[assignment]
    Documents = list[str]  # type: ignore[misc,assignment]
    Embeddings = list[list[float]]  # type: ignore[misc,assignment]

    class EmbeddingFunction:  # type: ignore[no-redef]
        """Fallback type used when chromadb is not installed."""

        pass

    CHROMADB_IMPORT_ERROR = exc
else:
    CHROMADB_IMPORT_ERROR = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    SentenceTransformer = None  # type: ignore[assignment]
    SENTENCE_TRANSFORMERS_IMPORT_ERROR = exc
else:
    SENTENCE_TRANSFORMERS_IMPORT_ERROR = None

try:
    import google.generativeai as genai
except ImportError as exc:
    genai = None  # type: ignore[assignment]
    GEMINI_IMPORT_ERROR = exc
else:
    GEMINI_IMPORT_ERROR = None


load_dotenv()

DEFAULT_VAULT_PATH = Path(
    os.getenv("OBSIDIAN_VAULT_PATH", Path(__file__).resolve().parent)
).resolve()
DEFAULT_CHROMA_PATH = Path(
    os.getenv("CHROMA_DB_PATH", DEFAULT_VAULT_PATH / "memory" / "chroma_db")
).resolve()
DEFAULT_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "cortex_obsidian")
DEFAULT_EMBEDDING_PROVIDER = os.getenv(
    "CORTEX_EMBEDDING_PROVIDER",
    "sentence-transformers",
)
DEFAULT_ST_MODEL_NAME = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
DEFAULT_GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "models/gemini-embedding-001",
)
DEFAULT_GEMINI_GENERATION_MODEL = os.getenv(
    "GEMINI_GENERATION_MODEL",
    "models/gemini-2.5-flash",
)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EXCLUDED_DIRECTORIES = {
    ".obsidian",
    ".git",
    ".trash",
    "__pycache__",
    "node_modules",
}

RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"


def log_info(message: str) -> None:
    """Write an informational message with ANSI colors."""

    print(f"{BLUE}[KNOWLEDGE]{RESET} {message}")


def log_success(message: str) -> None:
    """Write a success message with ANSI colors."""

    print(f"{GREEN}[KNOWLEDGE]{RESET} {message}")


def log_warning(message: str) -> None:
    """Write a warning message with ANSI colors."""

    print(f"{YELLOW}[KNOWLEDGE]{RESET} {message}")


def log_error(message: str) -> None:
    """Write an error message with ANSI colors."""

    print(f"{RED}[KNOWLEDGE]{RESET} {message}")


def ensure_runtime_dependencies(provider: str) -> None:
    """Validate optional dependencies before touching Chroma or embedding models."""

    missing_packages: list[str] = []
    if CHROMADB_IMPORT_ERROR is not None:
        missing_packages.append("chromadb")
    if provider == "sentence-transformers" and SENTENCE_TRANSFORMERS_IMPORT_ERROR is not None:
        missing_packages.append("sentence-transformers")
    if provider == "gemini":
        if GEMINI_IMPORT_ERROR is not None:
            missing_packages.append("google-generativeai")
        if not GEMINI_API_KEY or "votre_cle" in GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY est absente ou invalide dans le fichier .env.")

    if missing_packages:
        packages = " ".join(missing_packages)
        raise RuntimeError(
            "Dependances manquantes: "
            f"{', '.join(missing_packages)}. "
            f"Installe-les avec `pip install {packages}`."
        )


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """Local embedding function backed by Sentence-Transformers."""

    def __init__(self, model_name: str) -> None:
        ensure_runtime_dependencies("sentence-transformers")
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(
            list(input),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Remote embedding function backed by Gemini embeddings."""

    def __init__(self, model_name: str) -> None:
        ensure_runtime_dependencies("gemini")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        result = genai.embed_content(
            model=self.model_name,
            content=list(input),
            task_type="retrieval_document",
        )
        embeddings = result.get("embedding", [])
        if embeddings and isinstance(embeddings[0], float):
            return [embeddings]
        return embeddings


def remove_frontmatter(text: str) -> str:
    """Remove YAML frontmatter so chunking focuses on note content."""

    return re.sub(r"^---\s*\n.*?\n---\s*\n?", "", text, flags=re.DOTALL)


def chunk_text(text: str, chunk_size_words: int = 500) -> list[dict[str, Any]]:
    """Split markdown into paragraph-aware chunks of roughly chunk_size_words."""

    cleaned_text = remove_frontmatter(text).strip()
    if not cleaned_text:
        return []

    paragraphs = [paragraph.strip() for paragraph in cleaned_text.split("\n\n")]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]

    chunks: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_word_count = 0
    start_word = 0

    def flush_chunk() -> None:
        nonlocal current_parts, current_word_count, start_word
        if not current_parts:
            return

        chunk_value = "\n\n".join(current_parts).strip()
        if chunk_value:
            chunks.append(
                {
                    "text": chunk_value,
                    "word_count": current_word_count,
                    "start_word": start_word,
                    "end_word": start_word + current_word_count,
                }
            )
            start_word += current_word_count

        current_parts = []
        current_word_count = 0

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            continue

        if len(words) > chunk_size_words:
            flush_chunk()
            for offset in range(0, len(words), chunk_size_words):
                paragraph_words = words[offset : offset + chunk_size_words]
                paragraph_chunk = " ".join(paragraph_words).strip()
                chunks.append(
                    {
                        "text": paragraph_chunk,
                        "word_count": len(paragraph_words),
                        "start_word": start_word,
                        "end_word": start_word + len(paragraph_words),
                    }
                )
                start_word += len(paragraph_words)
            continue

        if current_word_count + len(words) > chunk_size_words and current_parts:
            flush_chunk()

        current_parts.append(paragraph)
        current_word_count += len(words)

    flush_chunk()
    return chunks


def extract_tags(content: str) -> list[str]:
    """Extract unique Obsidian-style tags from markdown content."""

    tags = re.findall(r"(#[a-zA-Z0-9_À-ÿ\-]+)", content)
    return sorted(set(tags))


def iter_markdown_files(vault_path: Path) -> list[Path]:
    """Return all markdown files in the vault excluding system folders."""

    markdown_files: list[Path] = []
    for md_file in vault_path.rglob("*.md"):
        if any(part in EXCLUDED_DIRECTORIES for part in md_file.parts):
            continue
        markdown_files.append(md_file)
    return sorted(markdown_files)


class KnowledgeEngine:
    """Single owner of ChromaDB state, indexing, semantic search, and file watching."""

    def __init__(
        self,
        vault_path: Path | str = DEFAULT_VAULT_PATH,
        chroma_path: Path | str = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
        sentence_transformer_model: str = DEFAULT_ST_MODEL_NAME,
        gemini_embedding_model: str = DEFAULT_GEMINI_EMBEDDING_MODEL,
    ) -> None:
        self.vault_path = Path(vault_path).resolve()
        self.chroma_path = Path(chroma_path).resolve()
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider
        self.sentence_transformer_model = sentence_transformer_model
        self.gemini_embedding_model = gemini_embedding_model

        self._client = None
        self._embedding_function = None
        self._collection = None
        self._lock = threading.Lock()

    def should_ignore_path(self, path: Path | str) -> bool:
        """Return True when a path should be ignored by indexing."""

        candidate = Path(path)
        if candidate.suffix.lower() != ".md":
            return True

        try:
            relative_path = candidate.resolve().relative_to(self.vault_path)
        except Exception:
            try:
                relative_path = candidate.relative_to(self.vault_path)
            except Exception:
                return True

        return any(part in EXCLUDED_DIRECTORIES for part in relative_path.parts)

    def get_relative_path(self, path: Path | str) -> Path:
        """Convert an absolute file path into a vault-relative path."""

        return Path(path).resolve().relative_to(self.vault_path)

    def _read_markdown_file(self, path: Path) -> str:
        """Read a markdown file with a safe utf-8 fallback."""

        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="ignore")

    def _build_note_records(
        self,
        relative_path: Path,
        content: str,
        chunk_size_words: int,
    ) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        """Build chunk ids, documents, and metadata for a single note."""

        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []
        relative_path_str = relative_path.as_posix()

        for chunk_index, chunk in enumerate(
            chunk_text(content, chunk_size_words=chunk_size_words)
        ):
            documents.append(chunk["text"])
            metadatas.append(
                {
                    "source": relative_path_str,
                    "filename": relative_path.name,
                    "chunk_index": chunk_index,
                    "start_word": chunk["start_word"],
                    "end_word": chunk["end_word"],
                    "word_count": chunk["word_count"],
                }
            )
            ids.append(f"{relative_path_str}#chunk-{chunk_index}")

        return ids, documents, metadatas

    def _get_client(self):
        """Lazily initialize the persistent Chroma client once per process."""

        if self._client is None:
            ensure_runtime_dependencies(self.embedding_provider)
            self.chroma_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.chroma_path))
        return self._client

    def _get_embedding_function(self):
        """Lazily initialize the configured embedding function once per process."""

        if self._embedding_function is None:
            if self.embedding_provider == "sentence-transformers":
                self._embedding_function = SentenceTransformerEmbeddingFunction(
                    self.sentence_transformer_model
                )
            elif self.embedding_provider == "gemini":
                self._embedding_function = GeminiEmbeddingFunction(
                    self.gemini_embedding_model
                )
            else:
                raise ValueError(
                    "embedding_provider doit valoir 'sentence-transformers' ou 'gemini'."
                )
        return self._embedding_function

    def _get_collection(self):
        """Lazily initialize the Chroma collection once per process."""

        if self._collection is None:
            self._collection = self._get_client().get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._get_embedding_function(),
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def reset_collection(self) -> None:
        """Delete and recreate the active collection."""

        with self._lock:
            client = self._get_client()
            try:
                client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = None

    def delete_note(self, path: Path | str) -> int:
        """Remove every chunk associated with a single note."""

        candidate = Path(path).resolve()
        if self.should_ignore_path(candidate):
            return 0

        with self._lock:
            relative_path = self.get_relative_path(candidate).as_posix()
            collection = self._get_collection()
            existing = collection.get(where={"source": relative_path}, include=[])
            ids = existing.get("ids", []) if existing else []
            if ids:
                collection.delete(ids=ids)
            return len(ids)

    def update_index(
        self,
        note_path: Path | str | None = None,
        *,
        deleted: bool = False,
        reset: bool = False,
        chunk_size_words: int = 500,
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """Index the full vault or a single note and return a structured event payload."""

        if note_path is None:
            total_chunks = self.index_vault(
                chunk_size_words=chunk_size_words,
                batch_size=batch_size,
                reset_collection=reset or True,
            )
            return {
                "type": "knowledge",
                "action": "reindexed",
                "chunks": total_chunks,
                "path": None,
            }

        candidate = Path(note_path).resolve()
        if deleted or not candidate.exists():
            removed_chunks = self.delete_note(candidate)
            return {
                "type": "knowledge",
                "action": "deleted",
                "chunks": removed_chunks,
                "path": candidate.name if candidate.name else str(candidate),
                "source": self._safe_relative_path(candidate),
                "filename": candidate.name,
                "tags": [],
            }

        if self.should_ignore_path(candidate):
            return {
                "type": "knowledge",
                "action": "ignored",
                "chunks": 0,
                "path": candidate.name,
            }

        with self._lock:
            relative_path = self.get_relative_path(candidate)
            relative_path_str = relative_path.as_posix()
            collection = self._get_collection()
            existing = collection.get(where={"source": relative_path_str}, include=[])
            existing_ids = existing.get("ids", []) if existing else []
            content = self._read_markdown_file(candidate)
            ids, documents, metadatas = self._build_note_records(
                relative_path=relative_path,
                content=content,
                chunk_size_words=chunk_size_words,
            )

            if existing_ids:
                collection.delete(ids=existing_ids)
            if ids:
                collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        return {
            "type": "knowledge",
            "action": "modified" if existing_ids else "created",
            "chunks": len(ids),
            "path": relative_path_str,
            "source": relative_path_str,
            "filename": relative_path.name,
            "tags": extract_tags(content),
        }

    def index_vault(
        self,
        chunk_size_words: int = 500,
        batch_size: int = 32,
        reset_collection: bool = True,
    ) -> int:
        """Reindex the entire vault into ChromaDB."""

        if not self.vault_path.exists():
            raise FileNotFoundError(f"Dossier Obsidian introuvable: {self.vault_path}")

        with self._lock:
            if reset_collection:
                self.reset_collection()

            collection = self._get_collection()
            documents: list[str] = []
            metadatas: list[dict[str, Any]] = []
            ids: list[str] = []

            for md_file in iter_markdown_files(self.vault_path):
                try:
                    content = self._read_markdown_file(md_file)
                except OSError as exc:
                    log_warning(f"Impossible de lire {md_file}: {exc}")
                    continue

                relative_path = md_file.relative_to(self.vault_path)
                chunk_ids, chunk_docs, chunk_metadatas = self._build_note_records(
                    relative_path=relative_path,
                    content=content,
                    chunk_size_words=chunk_size_words,
                )
                ids.extend(chunk_ids)
                documents.extend(chunk_docs)
                metadatas.extend(chunk_metadatas)

            for offset in range(0, len(documents), batch_size):
                collection.upsert(
                    ids=ids[offset : offset + batch_size],
                    documents=documents[offset : offset + batch_size],
                    metadatas=metadatas[offset : offset + batch_size],
                )

        return len(documents)

    def query_semantic(self, text: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Run semantic search and return note-level ids with normalized similarity scores."""

        with self._lock:
            results = self._get_collection().query(query_texts=[text], n_results=n_results * 3)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        grouped: dict[str, dict[str, Any]] = defaultdict(dict)
        for document, metadata, distance, result_id in zip(
            documents, metadatas, distances, ids
        ):
            source = metadata.get("source")
            if not source:
                continue

            score = max(0.0, 1.0 - float(distance if distance is not None else 0.0))
            best = grouped.get(source)
            if not best or score > best["score"]:
                grouped[source] = {
                    "note_id": source,
                    "score": round(score, 4),
                    "chunk_id": result_id,
                    "filename": metadata.get("filename"),
                    "chunk_index": metadata.get("chunk_index"),
                    "text": document,
                }

        ranked = sorted(grouped.values(), key=lambda item: item["score"], reverse=True)
        return ranked[:n_results]

    def _safe_relative_path(self, path: Path) -> str:
        """Return a relative path when possible, or a string fallback when deletion already happened."""

        try:
            return self.get_relative_path(path).as_posix()
        except Exception:
            return path.name if path.name else str(path)


class _WatchdogEventHandler(FileSystemEventHandler):
    """Translate filesystem events into indexed update/delete operations."""

    def __init__(self, event_queue: queue.Queue[tuple[str, str]]) -> None:
        self.event_queue = event_queue

    def on_created(self, event: FileSystemEvent) -> None:
        self._enqueue("upsert", event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._enqueue("upsert", event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._enqueue("delete", event)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self.event_queue.put(("delete", event.src_path))
        self.event_queue.put(("upsert", event.dest_path))

    def _enqueue(self, action: str, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self.event_queue.put((action, event.src_path))


class KnowledgeWatchService:
    """Watch the vault and keep ChromaDB synchronized incrementally."""

    def __init__(
        self,
        engine: KnowledgeEngine,
        *,
        chunk_size_words: int = 500,
        debounce_seconds: float = 0.8,
        retry_attempts: int = 6,
        retry_delay_seconds: float = 0.35,
        initial_sync: bool = True,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.engine = engine
        self.chunk_size_words = chunk_size_words
        self.debounce_seconds = debounce_seconds
        self.retry_attempts = retry_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.initial_sync = initial_sync
        self.on_event = on_event

        self.event_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.stop_event = threading.Event()
        self.observer = Observer()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)

    def start(self) -> None:
        """Start the observer and the worker thread."""

        if self.initial_sync:
            try:
                log_info("Synchronisation initiale de la base ChromaDB...")
                total_chunks = self.engine.index_vault(
                    chunk_size_words=self.chunk_size_words,
                    reset_collection=True,
                )
                log_success(f"Synchronisation initiale terminée: {total_chunks} segments indexés.")
            except Exception as exc:
                log_error(f"Échec de la synchronisation initiale: {exc}")

        handler = _WatchdogEventHandler(self.event_queue)
        self.observer.schedule(handler, path=str(self.engine.vault_path), recursive=True)
        self.observer.start()
        self.worker_thread.start()
        log_success(f"Surveillance active sur {self.engine.vault_path}")

    def stop(self) -> None:
        """Stop the observer and worker thread."""

        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self.observer.stop()
        self.observer.join()
        self.worker_thread.join(timeout=2)
        log_info("Service Watchdog arrêté.")

    def _worker_loop(self) -> None:
        pending: dict[str, str] = {}
        while not self.stop_event.is_set():
            try:
                action, raw_path = self.event_queue.get(timeout=0.5)
                pending[raw_path] = action

                while True:
                    try:
                        next_action, next_path = self.event_queue.get(
                            timeout=self.debounce_seconds
                        )
                        pending[next_path] = next_action
                    except queue.Empty:
                        break

                for pending_path, pending_action in list(pending.items()):
                    self._process_path(pending_action, Path(pending_path))
                pending.clear()
            except queue.Empty:
                continue
            except Exception as exc:
                log_error(f"Boucle de traitement interrompue: {exc}")

    def _process_path(self, action: str, path: Path) -> None:
        if action == "delete":
            self._delete(path)
        else:
            self._upsert_with_retries(path)

    def _delete(self, path: Path) -> None:
        try:
            event = self.engine.update_index(path, deleted=True)
            if event.get("chunks"):
                log_success(f"Supprimé de ChromaDB: {path.name} ({event['chunks']} segments)")
            self._emit(event)
        except Exception as exc:
            log_error(f"Suppression impossible pour {path}: {exc}")

    def _upsert_with_retries(self, path: Path) -> None:
        if self.engine.should_ignore_path(path):
            return

        for attempt in range(1, self.retry_attempts + 1):
            if not path.exists():
                log_warning(f"Fichier introuvable au moment du sync: {path}")
                return

            try:
                event = self.engine.update_index(
                    path,
                    chunk_size_words=self.chunk_size_words,
                )
                log_success(f"Sync OK: {path.name} ({event['chunks']} segments)")
                self._emit(event)
                return
            except (OSError, PermissionError) as exc:
                if attempt == self.retry_attempts:
                    log_error(f"Fichier verrouillé ou illisible après retries: {path} ({exc})")
                    return
                log_warning(
                    f"Fichier temporairement indisponible, tentative {attempt}/{self.retry_attempts}: {path.name}"
                )
                time.sleep(self.retry_delay_seconds)
            except Exception as exc:
                log_error(f"Échec de synchronisation pour {path}: {exc}")
                return

    def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(event)


_DEFAULT_ENGINE: KnowledgeEngine | None = None


def get_default_knowledge_engine() -> KnowledgeEngine:
    """Return the singleton knowledge engine used by the current process."""

    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = KnowledgeEngine()
    return _DEFAULT_ENGINE


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for knowledge indexing and querying."""

    parser = argparse.ArgumentParser(
        description="Moteur de connaissance unifié de Cortex."
    )
    parser.add_argument("--vault-path", default=str(DEFAULT_VAULT_PATH))
    parser.add_argument("--chroma-path", default=str(DEFAULT_CHROMA_PATH))
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument(
        "--embedding-provider",
        default=DEFAULT_EMBEDDING_PROVIDER,
        choices=["sentence-transformers", "gemini"],
    )
    parser.add_argument("--model-name", default=DEFAULT_ST_MODEL_NAME)
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_EMBEDDING_MODEL)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main() -> None:
    """CLI entrypoint used for manual indexing, querying, or watch mode."""

    parser = build_argument_parser()
    args = parser.parse_args()

    engine = KnowledgeEngine(
        vault_path=args.vault_path,
        chroma_path=args.chroma_path,
        collection_name=args.collection_name,
        embedding_provider=args.embedding_provider,
        sentence_transformer_model=args.model_name,
        gemini_embedding_model=args.gemini_model,
    )

    if args.watch:
        watcher = KnowledgeWatchService(
            engine=engine,
            chunk_size_words=args.chunk_size,
            initial_sync=not args.skip_index,
        )
        watcher.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            watcher.stop()
        return

    if not args.skip_index:
        total_chunks = engine.index_vault(
            chunk_size_words=args.chunk_size,
            batch_size=args.batch_size,
            reset_collection=True,
        )
        log_success(f"Indexation terminée: {total_chunks} segments enregistrés dans ChromaDB.")

    if args.query:
        matches = engine.query_semantic(args.query)
        if args.json_output:
            print(json.dumps(matches, ensure_ascii=False, indent=2))
        else:
            for index, match in enumerate(matches, start=1):
                print(f"\nRésultat {index}")
                print(f"Note: {match['note_id']}")
                print(f"Score: {match['score']:.4f}")
                print(match["text"])


if __name__ == "__main__":
    main()
