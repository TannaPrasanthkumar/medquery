import os

# ─────────────────────────────────────────────
#  API Keys
#  Reads from Streamlit Cloud secrets first,
#  then falls back to environment variables,
#  then falls back to empty string.
#  NEVER hardcode real keys here.
# ─────────────────────────────────────────────

def get_secret(key: str) -> str:
    """
    Fetch a secret from Streamlit Cloud secrets or environment variables.
    Works both locally and on Streamlit Cloud.
    """
    try:
        import streamlit as st
        return st.secrets.get(key, os.environ.get(key, ""))
    except Exception:
        return os.environ.get(key, "")


GROQ_API_KEY   = get_secret("GROQ_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")

# ─────────────────────────────────────────────
#  LLM Model Names
# ─────────────────────────────────────────────

GROQ_MODEL   = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-1.5-flash"

# ─────────────────────────────────────────────
#  Embedding & Vector Store Settings
# ─────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH  = "./chroma_db"
COLLECTION_NAME = "medquad_collection"

# ─────────────────────────────────────────────
#  RAG Chunking Settings
# ─────────────────────────────────────────────

CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80
TOP_K_RESULTS = 5

# ─────────────────────────────────────────────
#  Response Mode Prompts
# ─────────────────────────────────────────────

CONCISE_INSTRUCTION = (
    "Respond in 3-4 sentences maximum. Be direct and to the point. "
    "No bullet points unless absolutely necessary."
)

DETAILED_INSTRUCTION = (
    "Provide a thorough, well-structured response. Use headings and bullet points "
    "where helpful. Include explanations, causes, symptoms, treatments, and references "
    "when available. Be comprehensive."
)

# ─────────────────────────────────────────────
#  System Prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are MediQuery, an intelligent medical information assistant trained on
NIH-sourced data. You help users understand diseases, symptoms, treatments, drugs, and
medical conditions using verified medical knowledge.

IMPORTANT GUIDELINES:
- Always recommend consulting a licensed healthcare professional for personal medical decisions.
- Base your answers on the retrieved context when available.
- If you are unsure, say so clearly rather than guessing.
- Never diagnose — provide information only.
- Use plain language, but be medically accurate.
"""