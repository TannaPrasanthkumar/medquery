import os

#  API Keys 

GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
TAVILY_API_KEY  = os.environ.get("TAVILY_API_KEY", "")

# LLM Models

GROQ_MODEL   = "llama-3.3-70b-versatile"   
GEMINI_MODEL = "gemini-2.0-flash"         

#  Embedding model & Vector Store Settings

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH   = "./chroma_db"
COLLECTION_NAME  = "medquad_collection"

#  RAG Chunking Settings

CHUNK_SIZE    = 400   
CHUNK_OVERLAP = 80    
TOP_K_RESULTS = 5     

#  Response Mode Prompts

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

SYSTEM_PROMPT = """ 

You are MediQuery, an intelligent medical information assistant trained on 
NIH-sourced data. You help users understand diseases, symptoms, treatments, drugs, and 
medical conditions using verified medical knowledge.

IMPORTANT GUIDELINES :

- Always recommend consulting a licensed healthcare professional for personal medical decisions.
- Base your answers on the retrieved context when available.
- If you are unsure, say so clearly rather than guessing.
- Never diagnose — provide information only.
- Use plain language, but be medically accurate.
- If the question is outside the medical domain, use your general knowledge to help, but clarify that you are not a general assistant.
- Always cite your sources when possible, especially for medical claims. 
- If the user asks for symptoms, causes, treatments, or drug info, provide detailed explanations.
- If the user asks for a summary, provide a concise overview with key points.
- If the user asks about personal medical advice, remind them that you are not a doctor and recommend consulting a healthcare professional.
"""
