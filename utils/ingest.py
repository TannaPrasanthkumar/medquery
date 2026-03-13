import os
import sys
import time
import logging
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import chromadb

#  Configuration 
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT) 

DATA_FOLDER     = os.path.join(PROJECT_ROOT, "data", "MedQuAD")
CHROMA_DB_PATH  = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "medquad_collection"
CHUNK_SIZE      = 400                     
CHUNK_OVERLAP   = 80                      
MAX_FILES       = 100                    
BATCH_SIZE      = 500               

#  Logging Setup

LOG_FILE = os.path.join(PROJECT_ROOT, "ingest.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingest.log", mode="w", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)


#  Step 1: Parse MedQuAD XML files

def parse_xml_file(file_path: str) -> list:
    qa_pairs = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        source = root.get("source", os.path.basename(file_path))
        focus  = root.get("focus",  "General Medical")

        for qa in root.findall(".//QAPair"):
            q_el = qa.find("Question")
            a_el = qa.find("Answer")

            if q_el is None or a_el is None:
                continue

            question = (q_el.text or "").strip()
            answer   = (a_el.text or "").strip()
            qtype    = q_el.get("qtype", "general")

            if question and answer:
                qa_pairs.append({
                    "question": question,
                    "answer":   answer,
                    "source":   source,
                    "focus":    focus,
                    "qtype":    qtype,
                })
    except ET.ParseError as e:
        logger.warning("XML parse error in %s: %s", file_path, e)
    except Exception as e:
        logger.warning("Skipping %s - error: %s", file_path, e)

    return qa_pairs


def load_all_qa_pairs(data_folder: str, max_files: int = None) -> list:

    all_pairs  = []
    file_count = 0

    if not os.path.exists(data_folder):
        logger.error("Data folder not found: %s", data_folder)
        logger.error("   Please check the DATA_FOLDER path at the top of ingest.py")
        sys.exit(1)

    for root_dir, _, files in os.walk(data_folder):
        for fname in sorted(files):
            if not fname.endswith(".xml"):
                continue
            if max_files and file_count >= max_files:
                break

            full_path = os.path.join(root_dir, fname)
            pairs     = parse_xml_file(full_path)

            if pairs:
                all_pairs.extend(pairs)
                file_count += 1
                logger.info("  [%3d] %-50s -> %d pairs", file_count, fname, len(pairs))

    logger.info("-" * 60)
    logger.info("Parsed %d files  |  %d total Q&A pairs", file_count, len(all_pairs))
    return all_pairs


#  Step 2: Chunk Q&A pairs

def chunk_text(text: str) -> list:
    """Split text into overlapping chunks."""
    chunks = []
    start  = 0
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_chunks(qa_pairs: list) -> tuple:
    """
    Convert Q&A pairs into chunks with metadata and IDs.

    Returns:
        (chunks, metadatas, ids) -- parallel lists for ChromaDB
    """
    chunks    = []
    metadatas = []
    ids       = []
    chunk_idx = 0

    for pair in qa_pairs:
        combined    = f"Q: {pair['question']}\nA: {pair['answer']}"
        text_chunks = chunk_text(combined)

        for i, chunk in enumerate(text_chunks):
            chunks.append(chunk)
            metadatas.append({
                "source": pair.get("source", "unknown"),
                "focus":  pair.get("focus",  "general"),
                "qtype":  pair.get("qtype",  "general"),
                "chunk_index": i,
            })
            ids.append(f"chunk_{chunk_idx}")
            chunk_idx += 1

    logger.info("Created %d chunks from %d Q&A pairs", len(chunks), len(qa_pairs))
    return chunks, metadatas, ids


#  Step 3: Generate embeddings

def load_embedding_model() -> SentenceTransformer:

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    logger.info("   (Downloads ~90MB on first run - cached after that)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded")
    return model


def generate_embeddings(model: SentenceTransformer, chunks: list) -> list:
    """Generate embeddings for a list of text chunks."""
    logger.info("Generating embeddings for %d chunks...", len(chunks))
    all_embeddings = []
    total_batches  = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(chunks), BATCH_SIZE):
        
        batch      = chunks[i : i + BATCH_SIZE]
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())

        batch_num = i // BATCH_SIZE + 1
        progress  = round((batch_num / total_batches) * 100)
        logger.info(
            "  Batch %d/%d  (%d%%)  --  chunks %d to %d",
            batch_num, total_batches, progress, i, i + len(batch)
        )

    logger.info("Embeddings generated: %d vectors of dim %d",
                len(all_embeddings), len(all_embeddings[0]))
    return all_embeddings


#  Step 4: Store in ChromaDB

def store_in_chromadb(
    chunks:     list,
    metadatas:  list,
    ids:        list,
    embeddings: list,
) -> None:
    """
    Persist all chunks + embeddings into a local ChromaDB collection.
    Deletes old collection first to ensure a clean build.
    """
    logger.info("Connecting to ChromaDB at: %s", CHROMA_DB_PATH)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete old collection if it exists
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        logger.info("  Old collection deleted - rebuilding fresh")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Created ChromaDB collection: %s", COLLECTION_NAME)

    # Upsert in batches
    logger.info("Storing %d chunks in ChromaDB...", len(chunks))
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        collection.upsert(
            ids        = ids[i : i + BATCH_SIZE],
            documents  = chunks[i : i + BATCH_SIZE],
            embeddings = embeddings[i : i + BATCH_SIZE],
            metadatas  = metadatas[i : i + BATCH_SIZE],
        )
        logger.info("  Stored batch %d/%d", batch_num, total_batches)

    # Verify
    count = collection.count()
    logger.info("ChromaDB store complete -- %d chunks indexed", count)


#  Main
def main():
    print("\n" + "=" * 60)
    print("  MediQuery - Knowledge Base Ingestion")
    print("=" * 60 + "\n")

    start_time = time.time()

    print("STEP 1: Parsing MedQuAD XML files...")
    print(f"   Folder: {os.path.abspath(DATA_FOLDER)}\n")
    qa_pairs = load_all_qa_pairs(DATA_FOLDER, max_files=MAX_FILES)

    if not qa_pairs:
        logger.error("No Q&A pairs found. Check your DATA_FOLDER path.")
        sys.exit(1)
    
    print(f"\nSTEP 2: Chunking {len(qa_pairs)} Q&A pairs...")
    chunks, metadatas, ids = build_chunks(qa_pairs)
    
    print(f"\nSTEP 3: Generating embeddings for {len(chunks)} chunks...")
    model      = load_embedding_model()
    embeddings = generate_embeddings(model, chunks)

    print(f"\nSTEP 4: Storing in ChromaDB...")
    store_in_chromadb(chunks, metadatas, ids, embeddings)

    elapsed = round(time.time() - start_time, 1)
    print("\n" + "=" * 60)
    print(f"  Ingestion complete in {elapsed}s")
    print(f"  ChromaDB saved to: {os.path.abspath(CHROMA_DB_PATH)}")
    print(f"  Total chunks indexed: {len(chunks)}")
    print(f"  Now run: streamlit run app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()