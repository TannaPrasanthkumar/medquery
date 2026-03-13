# 🏥 MediQuery — AI Medical Information Assistant

> Built for the NeoStats AI Engineer Case Study

MediQuery is an intelligent chatbot that helps users understand medical conditions, symptoms,
treatments, and medications using **47,457 NIH-sourced Q&A pairs** from MedQuAD.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📚 **RAG Integration** | Retrieves relevant passages from MedQuAD using ChromaDB + sentence-transformers |
| 🌐 **Live Web Search** | Real-time search via Tavily API on trusted medical sites (NIH, CDC, Mayo Clinic) |
| ⚡ **Concise Mode** | 3-4 sentence focused answers |
| 📖 **Detailed Mode** | Structured, comprehensive explanations |
| 🤖 **Dual LLM** | Groq LLaMA 3.3 70B (primary) + Gemini 1.5 Flash (fallback) |
| 📎 **Source Citations** | Every web search result shows its source URL |

---

## 🗂️ Project Structure

```
medquery/
├── config/
│   └── config.py           ← API keys, model names, prompts
├── models/
│   ├── llm.py              ← Groq + Gemini LLM classes
│   └── embeddings.py       ← sentence-transformers embedding model
├── utils/
│   ├── rag_utils.py        ← MedQuAD parser, ChromaDB, retrieval
│   ├── search_utils.py     ← Tavily web search
│   └── ingest.py           ← One-time script to build ChromaDB from MedQuAD
├── data/
│   └── README.md           ← MedQuAD download instructions
├── app.py                  ← Main Streamlit UI
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd medquery
pip install -r requirements.txt
```

### 2. Add API Keys

Edit `config/config.py` and set your keys, **OR** set environment variables:

```bash
export GROQ_API_KEY="your_groq_key"
export GEMINI_API_KEY="your_gemini_key"
export TAVILY_API_KEY="your_tavily_key"
```

| Service | Free Signup |
|---|---|
| Groq (LLaMA 3.3 70B) | https://console.groq.com |
| Google Gemini | https://aistudio.google.com |
| Tavily (Web Search) | https://tavily.com |

### 3. Get the Data

```bash
cd data/
git clone https://github.com/abachaa/MedQuAD.git
cd ..
```

### 4. Build the Knowledge Base (Run Once)

This parses MedQuAD, generates embeddings, and stores them in ChromaDB.
Run it **once** — after that the app loads instantly every time.

```bash
python utils/ingest.py
```

Expected output:
```
============================================================
  MediQuery - Knowledge Base Ingestion
============================================================

STEP 1: Parsing MedQuAD XML files...
STEP 2: Chunking Q&A pairs...
STEP 3: Generating embeddings...
STEP 4: Storing in ChromaDB...

  Ingestion complete in ~90s
  Now run: streamlit run app.py
```

### 5. Run the App

```bash
streamlit run app.py
```

The app will auto-load the ChromaDB store — no manual setup needed in the UI.

---

## 🌐 Deploy to Streamlit Cloud

1. Push your code to GitHub (without API keys)
2. Go to https://streamlit.io/cloud
3. Connect your GitHub repo and set `app.py` as the main file
4. Add secrets in the Streamlit Cloud dashboard under **Settings → Secrets**:
   ```toml
   GROQ_API_KEY = "your_key"
   GEMINI_API_KEY = "your_key"
   TAVILY_API_KEY = "your_key"
   ```
5. Deploy!

> **Note:** For Streamlit Cloud deployment, run `ingest.py` locally first and commit the
> generated `chroma_db/` folder to your repo — or rebuild it as part of a startup script.

---

## ⚠️ Disclaimer

MediQuery provides general medical information for educational purposes only.
It is **not** a substitute for professional medical advice, diagnosis, or treatment.
Always consult a qualified healthcare provider for medical decisions.

---

## 📖 Data Attribution

This project uses the [MedQuAD Dataset](https://github.com/abachaa/MedQuAD) by Asma Ben Abacha
and Dina Demner-Fushman, licensed under CC BY 4.0.