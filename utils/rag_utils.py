"""
utils/rag_utils.py
───────────────────
RAG pipeline for MediQuery:
  1. parse_medquad_xml()     — Parse MedQuAD XML files into Q&A dicts
  2. chunk_text()            — Split long text into overlapping chunks
  3. build_vector_store()    — Index chunks into ChromaDB
  4. retrieve_context()      — Query ChromaDB for relevant chunks
  5. load_or_build_store()   — Smart loader (reuse if already built)
"""

import os
import logging
import xml.etree.ElementTree as ET
from typing import Optional
import chromadb
from chromadb.config import Settings
from models.embeddings import get_embedding_model
from config.config import (
    CHROMA_DB_PATH, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#  1. MedQuAD XML Parser

def parse_medquad_xml(file_path: str) -> list[dict]:

    qa_pairs = []
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        source = root.get("source", os.path.basename(file_path))
        focus  = root.get("focus", "General Medical")

        for qa_pair in root.findall(".//QAPair"):

            question_el = qa_pair.find("Question")
            answer_el   = qa_pair.find("Answer")

            if question_el is None or answer_el is None:
                continue

            question = (question_el.text or "").strip()
            answer   = (answer_el.text   or "").strip()
            qtype    = question_el.get("qtype", "general")

            if question and answer:
                qa_pairs.append({
                    "question": question,
                    "answer":   answer,
                    "source":   source,
                    "focus":    focus,
                    "qtype":    qtype,
                })

    except ET.ParseError as e:
        logger.error("XML parse error in %s: %s", file_path, e)
    except Exception as e:
        logger.error("Unexpected error parsing %s: %s", file_path, e)

    return qa_pairs


def parse_medquad_folder(folder_path: str, max_files: int = 50) -> list[dict]:

    all_qa = []
    file_count = 0

    try:
        for root_dir, _, files in os.walk(folder_path):
            for fname in files:
                if fname.endswith(".xml") and file_count < max_files:
                    full_path = os.path.join(root_dir, fname)
                    pairs = parse_medquad_xml(full_path)
                    all_qa.extend(pairs)
                    file_count += 1
                    logger.info("Parsed %s — %d pairs", fname, len(pairs))

        logger.info("Total Q&A pairs loaded: %d from %d files", len(all_qa), file_count)

    except Exception as e:
        logger.error("parse_medquad_folder error: %s", e)

    return all_qa


#  2. Text Chunking

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    
    chunks = []
    try:
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
    except Exception as e:
        logger.error("chunk_text error: %s", e)
    return chunks


def qa_pairs_to_chunks(qa_pairs: list[dict]) -> tuple[list[str], list[dict]]:
    """
    Convert Q&A pairs into chunks with metadata for ChromaDB.

    Each chunk = "Q: {question}\nA: {answer}" — so both sides are embedded together.

    Args:
        qa_pairs: List of Q&A dicts from parser.

    Returns:
        (chunks, metadatas) — parallel lists for ChromaDB upsert.
    """
    chunks    = []
    metadatas = []

    try:
        for pair in qa_pairs:
            combined = f"Q: {pair['question']}\nA: {pair['answer']}"
            text_chunks = chunk_text(combined)

            for i, chunk in enumerate(text_chunks):
                chunks.append(chunk)
                metadatas.append({
                    "source": pair.get("source", "unknown"),
                    "focus":  pair.get("focus",  "general"),
                    "qtype":  pair.get("qtype",  "general"),
                    "chunk_index": i,
                })

    except Exception as e:
        logger.error("qa_pairs_to_chunks error: %s", e)

    return chunks, metadatas


# ──────────────────────────────────────────────────────────
#  3. ChromaDB Vector Store
# ──────────────────────────────────────────────────────────

def get_chroma_client():
    """Return a persistent ChromaDB client."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        return client
    except Exception as e:
        logger.error("ChromaDB client error: %s", e)
        raise


def build_vector_store(qa_pairs: list[dict]) -> chromadb.Collection:
    """
    Build and persist a ChromaDB collection from Q&A pairs.

    Args:
        qa_pairs: Parsed MedQuAD Q&A pairs.

    Returns:
        ChromaDB Collection object.
    """
    try:
        embedding_model = get_embedding_model()
        client          = get_chroma_client()

        # Delete old collection if it exists
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("Old collection deleted.")
        except Exception:
            pass

        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        chunks, metadatas = qa_pairs_to_chunks(qa_pairs)

        if not chunks:
            logger.warning("No chunks to index!")
            return collection

        # Batch embed (ChromaDB upsert in batches of 500)
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch_chunks    = chunks[i : i + batch_size]
            batch_metas     = metadatas[i : i + batch_size]
            batch_ids       = [f"chunk_{i + j}" for j in range(len(batch_chunks))]
            batch_embeddings = embedding_model.embed_texts(batch_chunks)

            collection.upsert(
                ids=batch_ids,
                documents=batch_chunks,
                embeddings=batch_embeddings,
                metadatas=batch_metas,
            )
            logger.info("Indexed batch %d–%d", i, i + len(batch_chunks))

        logger.info("Vector store built with %d chunks.", len(chunks))
        return collection

    except Exception as e:
        logger.error("build_vector_store error: %s", e)
        raise


# ──────────────────────────────────────────────────────────
#  4. Retrieval
# ──────────────────────────────────────────────────────────

def retrieve_context(query: str, collection: chromadb.Collection, top_k: int = TOP_K_RESULTS) -> str:
    """
    Retrieve the most relevant medical context chunks for a query.

    Args:
        query      : User's question.
        collection : ChromaDB collection to search.
        top_k      : Number of chunks to retrieve.

    Returns:
        A formatted string of retrieved context passages.
    """
    try:
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return ""

        context_parts = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            relevance = round((1 - dist) * 100, 1)
            source    = meta.get("source", "NIH")
            focus     = meta.get("focus",  "")
            context_parts.append(
                f"[Source: {source} | Topic: {focus} | Relevance: {relevance}%]\n{doc}"
            )

        return "\n\n---\n\n".join(context_parts)

    except Exception as e:
        logger.error("retrieve_context error: %s", e)
        return ""


# ──────────────────────────────────────────────────────────
#  5. Smart Loader
# ──────────────────────────────────────────────────────────

def load_or_build_store(data_folder: Optional[str] = None) -> Optional[chromadb.Collection]:
    """
    Load existing ChromaDB collection if it exists,
    otherwise build it from the given data folder.

    Args:
        data_folder: Path to MedQuAD XML folder (needed only for first build).

    Returns:
        ChromaDB Collection or None if not ready.
    """
    try:
        client = get_chroma_client()
        existing = [c.name for c in client.list_collections()]

        if COLLECTION_NAME in existing:
            logger.info("Existing vector store found — loading.")
            return client.get_collection(COLLECTION_NAME)

        if not data_folder or not os.path.exists(data_folder):
            logger.warning("No data folder provided and no existing store found.")
            return None

        logger.info("Building vector store from: %s", data_folder)
        qa_pairs = parse_medquad_folder(data_folder)
        if not qa_pairs:
            logger.warning("No Q&A pairs parsed — check your data folder.")
            return None

        return build_vector_store(qa_pairs)

    except Exception as e:
        logger.error("load_or_build_store error: %s", e)
        return None
