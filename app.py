"""
app.py — MediQuery: AI-Powered Medical Information Assistant
─────────────────────────────────────────────────────────────
Built on MedQuAD (47,457 NIH Q&A pairs) with:
  ✅ RAG Integration (ChromaDB + sentence-transformers)
  ✅ Live Web Search (Tavily API)
  ✅ Concise / Detailed response modes
  ✅ Groq LLaMA 3.3 70B + Gemini 1.5 Flash
  ✅ Persistent chat history
  ✅ Source citations
"""

import os
import streamlit as st
from models.llm import get_llm
from models.embeddings import get_embedding_model
from utils.rag_utils import (
    load_or_build_store,
    build_vector_store,
    parse_medquad_folder,
    retrieve_context,
)
from utils.search_utils import web_search, format_search_results, build_search_citations


# ──────────────────────────────────────────────────────────
#  Page Config
# ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "MediQuery — AI Medical Assistant",
    page_icon   = "🏥",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# ──────────────────────────────────────────────────────────
#  Custom CSS
# ──────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a6b3a 0%, #0d4a5c 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }

    /* Chat bubbles */
    .chat-user {
        background: #e8f4f8;
        border-left: 4px solid #0d6efd;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .chat-assistant {
        background: #f0faf4;
        border-left: 4px solid #1a6b3a;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    /* Source badges */
    .source-badge {
        display: inline-block;
        background: #e9ecef;
        color: #495057;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 2px;
        text-decoration: none;
    }

    /* Status pills */
    .pill-rag    { background:#d4edda; color:#155724; padding:3px 10px; border-radius:12px; font-size:0.8rem; }
    .pill-search { background:#cce5ff; color:#004085; padding:3px 10px; border-radius:12px; font-size:0.8rem; }
    .pill-mode   { background:#fff3cd; color:#856404; padding:3px 10px; border-radius:12px; font-size:0.8rem; }

    /* Warning box */
    .medical-disclaimer {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.82rem;
        color: #856404;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
#  Session State Initialisation
# ──────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "messages":       [],       # chat history
        "collection":     None,     # ChromaDB collection
        "rag_ready":      False,    # RAG vector store status
        "llm_instance":   None,     # cached LLM
        "llm_provider":   "Groq (LLaMA 3.3)",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ──────────────────────────────────────────────────────────
#  Sidebar
# ──────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/caduceus.png", width=60)
    st.title("⚙️ MediQuery Settings")
    st.divider()

    # ── LLM Provider ──
    st.subheader("🤖 LLM Provider")
    llm_provider = st.selectbox(
        "Choose your model:",
        ["Groq (LLaMA 3.3)", "Gemini (Flash)"],
        index=0,
        help="Groq is faster. Gemini handles larger contexts."
    )

    if llm_provider != st.session_state["llm_provider"]:
        st.session_state["llm_provider"]  = llm_provider
        st.session_state["llm_instance"]  = None   # reset on change

    st.divider()

    # ── Response Mode ──
    st.subheader("📝 Response Mode")
    response_mode = st.radio(
        "Select mode:",
        ["Concise", "Detailed"],
        index=0,
        horizontal=True,
        help="Concise = 3-4 sentences. Detailed = full explanation with structure.",
    )

    st.divider()

    # ── RAG Settings ──
    st.subheader("📚 RAG Knowledge Base")

    use_rag = st.toggle("Enable RAG (MedQuAD)", value=True)

    if not st.session_state["rag_ready"]:
        st.info("💡 To enable RAG, provide your MedQuAD data folder path below, or upload XML files.")

        data_folder = st.text_input(
            "MedQuAD folder path:",
            value="./data",
            help="Path to the cloned MedQuAD repo folder",
        )

        uploaded_files = st.file_uploader(
            "Or upload MedQuAD XML files:",
            type=["xml"],
            accept_multiple_files=True,
        )

        if st.button("🔨 Build Knowledge Base", type="primary"):
            with st.spinner("Parsing MedQuAD and building vector store... (this takes ~1-2 min first time)"):
                try:
                    # Handle uploaded files
                    if uploaded_files:
                        import tempfile, shutil
                        tmp_dir = tempfile.mkdtemp()
                        for uf in uploaded_files:
                            with open(os.path.join(tmp_dir, uf.name), "wb") as f:
                                f.write(uf.read())
                        data_folder = tmp_dir

                    from utils.rag_utils import parse_medquad_folder, build_vector_store
                    qa_pairs = parse_medquad_folder(data_folder, max_files=60)

                    if not qa_pairs:
                        st.error("❌ No Q&A pairs found. Check your folder path.")
                    else:
                        collection = build_vector_store(qa_pairs)
                        st.session_state["collection"] = collection
                        st.session_state["rag_ready"]  = True
                        st.success(f"✅ Indexed {len(qa_pairs)} Q&A pairs into vector store!")
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Error building knowledge base: {e}")

        # Try auto-loading existing store
        if not st.session_state["rag_ready"]:
            try:
                collection = load_or_build_store()
                if collection:
                    st.session_state["collection"] = collection
                    st.session_state["rag_ready"]  = True
                    st.success("✅ Existing knowledge base loaded!")
                    st.rerun()
            except Exception:
                pass

    else:
        st.success("✅ Knowledge base ready!")
        if st.button("🔄 Rebuild Knowledge Base"):
            st.session_state["rag_ready"]  = False
            st.session_state["collection"] = None
            st.rerun()

    st.divider()

    # ── Web Search ──
    st.subheader("🌐 Live Web Search")
    use_web_search = st.toggle(
        "Enable web search",
        value=False,
        help="Searches trusted medical sites (NIH, Mayo Clinic, CDC) in real time.",
    )

    st.divider()

    # ── Clear Chat ──
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    # ── About ──
    st.divider()
    st.caption("**MediQuery v1.0**")
    st.caption("Data: MedQuAD (47,457 NIH Q&A pairs)")
    st.caption("Models: LLaMA 3.3 70B via Groq | Gemini 1.5 Flash")
    st.caption("⚠️ For informational use only. Not medical advice.")


# ──────────────────────────────────────────────────────────
#  Main Header
# ──────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
  <h1>🏥 MediQuery</h1>
  <p>AI-powered medical information assistant built on 47,457 NIH Q&A pairs.<br>
  Ask about diseases, symptoms, treatments, medications, and more.</p>
</div>
""", unsafe_allow_html=True)

# Medical disclaimer
st.markdown("""
<div class="medical-disclaimer">
  ⚠️ <strong>Disclaimer:</strong> MediQuery provides general medical information only.
  It is not a substitute for professional medical advice, diagnosis, or treatment.
  Always consult a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)

# Active feature pills
cols = st.columns([1, 1, 1, 5])
with cols[0]:
    if use_rag and st.session_state["rag_ready"]:
        st.markdown('<span class="pill-rag">✅ RAG Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:grey;font-size:0.8rem">○ RAG Off</span>', unsafe_allow_html=True)
with cols[1]:
    if use_web_search:
        st.markdown('<span class="pill-search">🌐 Web Search</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:grey;font-size:0.8rem">○ Search Off</span>', unsafe_allow_html=True)
with cols[2]:
    mode_label = "⚡ Concise" if response_mode == "Concise" else "📖 Detailed"
    st.markdown(f'<span class="pill-mode">{mode_label}</span>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
#  Chat History Display
# ──────────────────────────────────────────────────────────

chat_container = st.container()

with chat_container:
    if not st.session_state["messages"]:
        st.markdown("""
        ### 👋 Welcome to MediQuery!

        Here are some questions you can ask:

        - *What are the symptoms of Type 2 Diabetes?*
        - *How is hypertension treated?*
        - *What medications are used for Alzheimer's disease?*
        - *What causes chronic kidney disease?*
        - *What is the difference between Type 1 and Type 2 Diabetes?*
        """)
    else:
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🏥"):
                    st.markdown(msg["content"])
                    # Show citations if any
                    if msg.get("citations"):
                        st.markdown("**📎 Sources:**")
                        citation_html = " ".join([
                            f'<a class="source-badge" href="{c["url"]}" target="_blank">🔗 {c["title"][:40]}</a>'
                            for c in msg["citations"]
                        ])
                        st.markdown(citation_html, unsafe_allow_html=True)
                    # Show RAG tag
                    if msg.get("used_rag"):
                        st.caption("📚 Answer enhanced with MedQuAD knowledge base")
                    if msg.get("used_search"):
                        st.caption("🌐 Answer enhanced with live web search")


# ──────────────────────────────────────────────────────────
#  Chat Input
# ──────────────────────────────────────────────────────────

user_input = st.chat_input("Ask a medical question (e.g. What are symptoms of diabetes?)")

if user_input:
    # Display user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Build context from RAG + Web Search
    rag_context    = ""
    search_context = ""
    citations      = []
    used_rag       = False
    used_search    = False

    with st.status("🔍 Thinking...", expanded=True) as status:

        # Step 1: RAG retrieval
        if use_rag and st.session_state["rag_ready"] and st.session_state["collection"]:
            st.write("📚 Searching MedQuAD knowledge base...")
            try:
                rag_context = retrieve_context(
                    query=user_input,
                    collection=st.session_state["collection"],
                )
                if rag_context:
                    used_rag = True
                    st.write(f"✅ Found {rag_context.count('---') + 1} relevant passages")
            except Exception as e:
                st.write(f"⚠️ RAG retrieval error: {e}")

        # Step 2: Web search
        if use_web_search:
            st.write("🌐 Searching trusted medical websites...")
            try:
                search_data    = web_search(user_input)
                search_context = format_search_results(search_data)
                citations      = build_search_citations(search_data)
                if search_context and not search_data.get("error"):
                    used_search = True
                    st.write(f"✅ Found {len(citations)} web sources")
                elif search_data.get("error"):
                    st.write(f"⚠️ Search error: {search_data['error']}")
            except Exception as e:
                st.write(f"⚠️ Web search error: {e}")

        # Step 3: Build combined context
        combined_context = ""
        if rag_context:
            combined_context += f"=== MedQuAD Knowledge Base ===\n{rag_context}\n\n"
        if search_context:
            combined_context += f"=== Live Web Search Results ===\n{search_context}"

        # Step 4: Generate LLM response
        st.write(f"🤖 Generating {response_mode.lower()} response with {llm_provider}...")
        try:
            if not st.session_state["llm_instance"]:
                st.session_state["llm_instance"] = get_llm(llm_provider)

            llm = st.session_state["llm_instance"]

            # Build chat history for context window (last 6 messages)
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state["messages"][:-1][-6:]
            ]

            response_text = llm.generate(
                user_message=user_input,
                context=combined_context,
                mode=response_mode.lower(),
                chat_history=history,
            )
            status.update(label="✅ Done!", state="complete", expanded=False)

        except Exception as e:
            response_text = f"⚠️ Error generating response: {str(e)}\n\nPlease check your API keys in config/config.py."
            status.update(label="❌ Error", state="error")

    # Display assistant response
    with st.chat_message("assistant", avatar="🏥"):
        st.markdown(response_text)
        if citations:
            st.markdown("**📎 Sources:**")
            citation_html = " ".join([
                f'<a class="source-badge" href="{c["url"]}" target="_blank">🔗 {c["title"][:40]}</a>'
                for c in citations
            ])
            st.markdown(citation_html, unsafe_allow_html=True)
        if used_rag:
            st.caption("📚 Answer enhanced with MedQuAD knowledge base")
        if used_search:
            st.caption("🌐 Answer enhanced with live web search")

    # Save to history
    st.session_state["messages"].append({
        "role":        "assistant",
        "content":     response_text,
        "citations":   citations,
        "used_rag":    used_rag,
        "used_search": used_search,
    })
