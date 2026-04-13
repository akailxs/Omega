from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    import chromadb
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
except ImportError as exc:
    chromadb = None  # type: ignore[assignment]
    Documents = list[str]  # type: ignore[misc,assignment]
    Embeddings = list[list[float]]  # type: ignore[misc,assignment]

    class EmbeddingFunction:  # type: ignore[no-redef]
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
DEFAULT_EMBEDDING_PROVIDER = os.getenv("CORTEX_EMBEDDING_PROVIDER", "sentence-transformers")
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


def ensure_runtime_dependencies(provider: str) -> None:
    missing_packages: list[str] = []
    if CHROMADB_IMPORT_ERROR is not None:
        missing_packages.append("chromadb")
    if provider == "sentence-transformers" and SENTENCE_TRANSFORMERS_IMPORT_ERROR is not None:
        missing_packages.append("sentence-transformers")
    if provider == "gemini":
        if GEMINI_IMPORT_ERROR is not None:
            missing_packages.append("google-generativeai")
        if not GEMINI_API_KEY or "votre_cle" in GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY est absente ou invalide dans le fichier .env."
            )

    if missing_packages:
        raise RuntimeError(
            "Dependances manquantes: "
            f"{', '.join(missing_packages)}. "
            f"Installe-les avec `pip install {' '.join(missing_packages)}`."
        )


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
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
    return re.sub(r"^---\s*\n.*?\n---\s*\n?", "", text, flags=re.DOTALL)


def chunk_text(text: str, chunk_size_words: int = 500) -> list[dict[str, Any]]:
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


def iter_markdown_files(vault_path: Path) -> list[Path]:
    markdown_files: list[Path] = []
    for md_file in vault_path.rglob("*.md"):
        if any(part in EXCLUDED_DIRECTORIES for part in md_file.parts):
            continue
        markdown_files.append(md_file)
    return sorted(markdown_files)


class CortexKnowledgeManager:
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

    def should_ignore_path(self, path: Path | str) -> bool:
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
        candidate = Path(path).resolve()
        return candidate.relative_to(self.vault_path)

    def _read_markdown_file(self, path: Path) -> str:
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
        if self._client is None:
            ensure_runtime_dependencies(self.embedding_provider)
            self.chroma_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.chroma_path))
        return self._client

    def _get_embedding_function(self):
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
        if self._collection is None:
            self._collection = self._get_client().get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._get_embedding_function(),
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def reset_collection(self) -> None:
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None

    def delete_note(self, path: Path | str) -> int:
        candidate = Path(path).resolve()
        if self.should_ignore_path(candidate):
            return 0

        relative_path = self.get_relative_path(candidate).as_posix()
        collection = self._get_collection()
        existing = collection.get(
            where={"source": relative_path},
            include=[],
        )
        ids = existing.get("ids", []) if existing else []
        if ids:
            collection.delete(ids=ids)
        return len(ids)

    def upsert_note(self, path: Path | str, chunk_size_words: int = 500) -> int:
        candidate = Path(path).resolve()
        if self.should_ignore_path(candidate):
            return 0

        relative_path = self.get_relative_path(candidate)
        content = self._read_markdown_file(candidate)
        ids, documents, metadatas = self._build_note_records(
            relative_path=relative_path,
            content=content,
            chunk_size_words=chunk_size_words,
        )

        collection = self._get_collection()
        existing = collection.get(where={"source": relative_path.as_posix()}, include=[])
        existing_ids = existing.get("ids", []) if existing else []
        if existing_ids:
            collection.delete(ids=existing_ids)

        if ids:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)

    def index_vault(
        self,
        chunk_size_words: int = 500,
        batch_size: int = 32,
        reset_collection: bool = True,
    ) -> int:
        if not self.vault_path.exists():
            raise FileNotFoundError(f"Dossier Obsidian introuvable: {self.vault_path}")

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
                print(f"Impossible de lire {md_file}: {exc}")
                continue

            relative_path = md_file.relative_to(self.vault_path).as_posix()
            for chunk_index, chunk in enumerate(
                chunk_text(content, chunk_size_words=chunk_size_words)
            ):
                documents.append(chunk["text"])
                metadatas.append(
                    {
                        "source": relative_path,
                        "filename": md_file.name,
                        "chunk_index": chunk_index,
                        "start_word": chunk["start_word"],
                        "end_word": chunk["end_word"],
                        "word_count": chunk["word_count"],
                    }
                )
                ids.append(f"{relative_path}#chunk-{chunk_index}")

        for offset in range(0, len(documents), batch_size):
            collection.upsert(
                ids=ids[offset : offset + batch_size],
                documents=documents[offset : offset + batch_size],
                metadatas=metadatas[offset : offset + batch_size],
            )

        return len(documents)

    def search(self, query: str, n_results: int = 3) -> list[dict[str, Any]]:
        results = self._get_collection().query(query_texts=[query], n_results=n_results)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        matches: list[dict[str, Any]] = []
        for document, metadata, distance, result_id in zip(
            documents, metadatas, distances, ids
        ):
            matches.append(
                {
                    "id": result_id,
                    "source": metadata.get("source"),
                    "filename": metadata.get("filename"),
                    "chunk_index": metadata.get("chunk_index"),
                    "distance": distance,
                    "text": document,
                }
            )
        return matches


_DEFAULT_MANAGER: CortexKnowledgeManager | None = None


def get_default_manager() -> CortexKnowledgeManager:
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = CortexKnowledgeManager()
    return _DEFAULT_MANAGER


def find_relevant_notes(query: str, n_results: int = 3) -> list[dict[str, Any]]:
    return get_default_manager().search(query=query, n_results=n_results)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gestionnaire unifie de la base de connaissances Cortex."
    )
    parser.add_argument(
        "--vault-path",
        default=str(DEFAULT_VAULT_PATH),
        help="Chemin du dossier Obsidian.",
    )
    parser.add_argument(
        "--chroma-path",
        default=str(DEFAULT_CHROMA_PATH),
        help="Chemin de persistance ChromaDB.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Nom de la collection ChromaDB.",
    )
    parser.add_argument(
        "--embedding-provider",
        default=DEFAULT_EMBEDDING_PROVIDER,
        choices=["sentence-transformers", "gemini"],
        help="Provider d'embeddings a utiliser.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_ST_MODEL_NAME,
        help="Nom du modele Sentence-Transformers.",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_EMBEDDING_MODEL,
        help="Nom du modele d'embeddings Gemini.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Taille maximale d'un segment en nombre de mots.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Taille des lots d'upsert ChromaDB.",
    )
    parser.add_argument(
        "--query",
        help="Recherche semantique a executer.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Interroge uniquement la base existante.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    manager = CortexKnowledgeManager(
        vault_path=args.vault_path,
        chroma_path=args.chroma_path,
        collection_name=args.collection_name,
        embedding_provider=args.embedding_provider,
        sentence_transformer_model=args.model_name,
        gemini_embedding_model=args.gemini_model,
    )

    if not args.skip_index:
        total_chunks = manager.index_vault(
            chunk_size_words=args.chunk_size,
            batch_size=args.batch_size,
        )
        print(f"Indexation terminee: {total_chunks} segments enregistres dans ChromaDB.")

    if args.query:
        matches = manager.search(query=args.query)
        for index, match in enumerate(matches, start=1):
            print(f"\nResultat {index}")
            print(f"Source: {match['source']} (chunk {match['chunk_index']})")
            print(f"Distance: {match['distance']:.4f}")
            print(match["text"])


if __name__ == "__main__":
    main()
