import os
os.environ.setdefault("USER_AGENT", "DocChat/1.0")

import concurrent.futures
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from core import (
    process_uploaded_file,
    process_url,
    add_documents,
    get_collection_count,
    clear_collection,
    delete_document_by_source,
    get_indexed_sources,
    get_source_chunk_counts,
    query_rag,
    generate_followup_questions,
    summarize_document,
)

st.set_page_config(
    page_title="DocChat",
    page_icon=None,
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

:root {
    --ground:        #EEF1F5;
    --surface:       #FFFFFF;
    --text:          #1A2332;
    --text-muted:    #5C6B7A;
    --accent:        #1B6CA8;
    --accent-light:  #EAF2FA;
    --panel:         #1A2332;
    --panel-surface: #232F40;
    --panel-border:  #2D3E50;
    --panel-text:    #D8E4EF;
    --panel-muted:   #7A90A8;
    --border:        #D0D7DF;
    --danger-dim:    #3D1515;
    --danger-text:   #E08585;
}

/* ── Global reset ──────────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
    background-color: var(--ground) !important;
    font-family: 'Inter', system-ui, sans-serif;
    color: var(--text);
}

/* ── Sidebar — dark instrument panel ───────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--panel) !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: var(--panel-text) !important;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--panel-muted) !important;
    margin-top: 0 !important;
    margin-bottom: 0.75rem !important;
}

[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid var(--panel-border) !important;
    margin: 1.25rem 0 !important;
}

/* Sidebar select */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background-color: var(--panel-surface) !important;
    border: 1px solid var(--panel-border) !important;
    color: var(--panel-text) !important;
    border-radius: 4px !important;
}

/* Sidebar labels */
[data-testid="stSidebar"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--panel-muted) !important;
}

/* Sidebar file uploader */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background-color: var(--panel-surface) !important;
    border: 1px dashed var(--panel-border) !important;
    border-radius: 4px !important;
}

/* File list expander — float over sidebar content */
[data-testid="stSidebar"] details[data-testid="stExpander"] {
    position: relative !important;
}
[data-testid="stSidebar"] details[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {
    position: absolute !important;
    z-index: 9999 !important;
    top: 100% !important;
    left: 0 !important;
    width: 100% !important;
    background-color: var(--panel-surface) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 4px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
    padding: 0.5rem !important;
}

/* File list rows — styled as plain list items */
[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] .file-list-btn > button {
    background-color: transparent !important;
    border: none !important;
    color: var(--panel-text) !important;
    font-size: 0.8rem !important;
    text-align: left !important;
    padding: 0.25rem 0.5rem !important;
    border-radius: 3px !important;
}
[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] .file-list-btn > button:hover {
    background-color: var(--panel-border) !important;
}

/* Hide native file chip list — replaced by custom file rows */
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
[data-testid="stSidebar"] [data-testid="stFileUploaderFileData"],
[data-testid="stSidebar"] [data-testid="stFileUploaderFileList"],
[data-testid="stSidebar"] [data-testid="stFileUploader"] ul,
[data-testid="stSidebar"] [data-testid="stFileUploader"] li {
    display: none !important;
}

/* Sidebar metric card */
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background-color: var(--panel-surface) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 4px !important;
    padding: 0.75rem 1rem !important;
}

[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--panel-muted) !important;
}

[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.75rem !important;
    font-weight: 600 !important;
    color: var(--panel-text) !important;
}

/* Sidebar primary button */
[data-testid="stSidebar"] .stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    background-color: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    padding: 0.55rem 1rem !important;
    transition: opacity 0.15s ease !important;
    width: 100% !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    opacity: 0.85 !important;
    background-color: var(--accent) !important;
}

/* Sidebar secondary (destructive) button */
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background-color: transparent !important;
    border: 1px solid var(--danger-dim) !important;
    color: var(--danger-text) !important;
}

[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    background-color: var(--danger-dim) !important;
    opacity: 1 !important;
}

/* ── Main content heading ───────────────────────────────────────────── */
[data-testid="stMain"] h1 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    color: var(--text) !important;
    border-bottom: 2px solid var(--accent) !important;
    padding-bottom: 0.55rem !important;
    margin-bottom: 0.2rem !important;
}

[data-testid="stCaptionContainer"] p {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.01em !important;
    margin-top: 0.25rem !important;
}

/* ── Chat message cards ──────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.6rem !important;
    box-shadow: 0 1px 3px rgba(26, 35, 50, 0.06) !important;
}

/* User messages — tinted blue */
[data-testid="stChatMessage"]:nth-child(odd) {
    background-color: var(--accent-light) !important;
    border-color: #C5D8EC !important;
}

/* ── Referenced sources expander ────────────────────────────────────── */
details[data-testid="stExpander"] {
    background-color: #F4F7FA !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    margin-top: 0.5rem !important;
}

details[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    padding: 0.5rem 0.75rem !important;
}

details[data-testid="stExpander"] > div {
    font-size: 0.85rem !important;
    color: var(--text-muted) !important;
    padding: 0.25rem 0.75rem 0.75rem !important;
}

/* ── Alerts ─────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 4px !important;
    font-size: 0.85rem !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Chat input bar ──────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    border-top: 1px solid var(--border) !important;
    background-color: var(--ground) !important;
}

[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    border-radius: 4px !important;
    border: 1px solid var(--border) !important;
    background-color: var(--surface) !important;
    color: var(--text) !important;
}

/* ── Spinner text ───────────────────────────────────────────────────── */
[data-testid="stSpinner"] p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    color: var(--text-muted) !important;
}

/* ── Progress bar ───────────────────────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
    background-color: var(--accent) !important;
}

/* ── Scrollbar ──────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Hide Streamlit chrome ──────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)

st.title("DocChat")
st.caption("Document-grounded Q&A. Answers are drawn exclusively from your uploaded sources and include citations.")

# ── Session state initialisation ─────────────────────────────────────
for key, default in [
    ("messages", []),
    ("uploader_key", 0),
    ("removed_files", set()),
    ("pending_question", None),
    ("last_followups", []),
    ("summarize_target", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def _build_chat_history(messages: list, max_turns: int = 4) -> list:
    """Convert the last N message pairs to LangChain message objects."""
    history = []
    for msg in messages[-(max_turns * 2):]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    return history


# ── Sidebar — control panel ───────────────────────────────────────────
with st.sidebar:
    st.header("Model")

    provider = st.selectbox(
        "LLM Provider",
        options=["Groq (Llama3.3-70B)", "Anthropic (Claude-Sonnet-4.6)"],
        help="Select the language model used to generate answers.",
    )

    response_length = st.selectbox(
        "Response Length",
        options=["Short", "Medium", "Long"],
        index=1,
        help="Short: 2-3 sentences. Medium: a few paragraphs. Long: detailed with full coverage.",
    )

    use_rag = st.checkbox(
        "Use indexed documents",
        value=False,
        help="When checked, answers are grounded in your uploaded documents.",
    )

    st.divider()
    st.header("Documents")

    # ── URL ingestion ──────────────────────────────────────────────────
    with st.expander("Add from URL"):
        url_input = st.text_input("Web page URL", placeholder="https://...", label_visibility="collapsed")
        if st.button("Fetch & Index URL", use_container_width=True):
            if url_input.strip():
                try:
                    with st.spinner("Fetching page..."):
                        chunks = process_url(url_input.strip())
                    progress = st.progress(0, text="Embedding & indexing...")
                    add_documents(
                        chunks,
                        progress_callback=lambda done, total: progress.progress(
                            done / total, text=f"Embedding & indexing... ({done}/{total} chunks)"
                        ),
                    )
                    st.success(f"Indexed {len(chunks)} chunks.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to fetch URL: {e}")
            else:
                st.warning("Enter a URL first.")

    # ── File upload ────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploaded_files:
        current_names = {f.name for f in uploaded_files}
        st.session_state.removed_files &= current_names
        visible = [f for f in uploaded_files if f.name not in st.session_state.removed_files]

        if not visible:
            st.session_state.removed_files = set()
            st.session_state.uploader_key += 1
            st.rerun()

        with st.expander(f"{len(visible)} document{'s' if len(visible) != 1 else ''} selected"):
            with st.container(height=min(160, len(visible) * 42), border=False):
                for f in visible:
                    display_name = f.name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
                    if st.button(f"✕  {display_name}", key=f"remove_{f.name}", use_container_width=True):
                        st.session_state.removed_files.add(f.name)
                        st.rerun()

        selected_files = visible

        if selected_files and st.button("Process Documents", use_container_width=True):
            # Pre-read bytes in main thread before handing off to workers
            file_data = [(f.name, f.read()) for f in selected_files]
            progress = st.progress(0, text=f"Extracting text... (0/{len(file_data)})")
            all_chunks = []
            errors = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(file_data))) as executor:
                futures = {
                    executor.submit(process_uploaded_file, data, name): name
                    for name, data in file_data
                }
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    fname = futures[future]
                    try:
                        all_chunks.extend(future.result())
                    except Exception as e:
                        errors.append(f"{fname}: {e}")
                    completed += 1
                    progress.progress(
                        0.5 * completed / len(file_data),
                        text=f"Extracting text... ({completed}/{len(file_data)})",
                    )

            for err in errors:
                st.error(f"Failed to process {err}")

            if all_chunks:
                try:
                    add_documents(
                        all_chunks,
                        progress_callback=lambda done, total: progress.progress(
                            0.5 + 0.5 * done / total,
                            text=f"Embedding & indexing... ({done}/{total} chunks)",
                        ),
                    )
                    progress.progress(1.0, text="Done.")
                    st.success(f"Indexed {len(all_chunks)} chunks from {len(file_data) - len(errors)} file(s).")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
            else:
                progress.progress(1.0, text="Done.")

            st.session_state.removed_files = set()
            st.session_state.uploader_key += 1
            st.rerun()

    # ── Indexed documents management ──────────────────────────────────
    indexed_sources = get_indexed_sources()
    if indexed_sources:
        st.divider()
        st.header("Indexed Documents")
        source_counts = get_source_chunk_counts()
        with st.container(height=min(240, len(indexed_sources) * 58), border=False):
            for source in indexed_sources:
                label = source if len(source) <= 24 else source[:22] + "…"
                count = source_counts.get(source, 0)
                chunk_label = f"{count} chunk{'s' if count != 1 else ''}" if count else "chunks: unknown"
                col_name, col_sum, col_del = st.columns([5, 1, 1])
                col_name.markdown(
                    f"<small style='color:#7A90A8;font-family:monospace'>{label}</small>"
                    f"<br><span style='color:#4A6080;font-size:0.65rem'>{chunk_label}</span>",
                    unsafe_allow_html=True,
                )
                if col_sum.button("≡", key=f"sum_{source}", help=f"Summarize {source}"):
                    st.session_state.summarize_target = source
                    st.rerun()
                if col_del.button("✕", key=f"del_{source}", help=f"Remove {source} from index"):
                    try:
                        delete_document_by_source(source)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    st.divider()
    doc_count = get_collection_count()
    st.metric("Indexed chunks", doc_count)

    if st.button("Clear Index", use_container_width=True, type="secondary"):
        clear_collection()
        st.warning("Vector index cleared.")
        st.rerun()

    # ── Export chat ────────────────────────────────────────────────────
    if st.session_state.messages:
        st.divider()
        st.header("Session")
        md_parts = ["# DocChat Session\n"]
        for msg in st.session_state.messages:
            role = "**You**" if msg["role"] == "user" else "**DocChat**"
            md_parts.append(f"{role}\n\n{msg['content']}")
            if msg.get("sources"):
                md_parts.append(f"\n*Sources: {', '.join(msg['sources'])}*")
        st.download_button(
            "Export Chat",
            data="\n\n---\n\n".join(md_parts),
            file_name="docchat-session.md",
            mime="text/markdown",
            use_container_width=True,
        )


# ── Summarization result ──────────────────────────────────────────────
if st.session_state.summarize_target:
    source = st.session_state.summarize_target
    st.session_state.summarize_target = None
    with st.spinner(f"Summarizing {source}..."):
        try:
            summary = summarize_document(source, provider)
        except Exception as e:
            summary = f"Summarization failed: {e}"
    with st.expander(f"Summary — {source}", expanded=True):
        st.markdown(summary)


# ── Chat history ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("source_chunks"):
            with st.expander("Referenced Sources"):
                for sc in msg["source_chunks"]:
                    st.markdown(f"**{sc['label']}**")
                    preview = sc["content"][:400] + "…" if len(sc["content"]) > 400 else sc["content"]
                    st.caption(preview)
        elif msg.get("sources"):
            with st.expander("Referenced Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

# ── Follow-up question pills (most recent exchange only) ─────────────
if st.session_state.last_followups:
    st.markdown(
        "<p style='font-size:0.78rem;color:#5C6B7A;font-family:IBM Plex Mono,monospace;"
        "letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.4rem'>"
        "Suggested follow-ups</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(st.session_state.last_followups))
    for i, q in enumerate(st.session_state.last_followups):
        if cols[i].button(q, key=f"followup_{i}_{q[:15]}"):
            st.session_state.pending_question = q
            st.session_state.last_followups = []
            st.rerun()


# ── Chat input ────────────────────────────────────────────────────────
typed = st.chat_input("Ask anything...")
prompt = typed or st.session_state.pending_question
if st.session_state.pending_question:
    st.session_state.pending_question = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            chat_history = _build_chat_history(st.session_state.messages[:-1])
            result = query_rag(prompt, provider, response_length, use_rag, chat_history, stream=True)
            answer = st.write_stream(result["stream"])
            sources = result["sources"]
            source_chunks = result["source_chunks"]
            grounded = result.get("grounded", bool(sources))
        except Exception as e:
            answer = f"An error occurred: {e}"
            sources = []
            source_chunks = []
            grounded = False

        if source_chunks:
            with st.expander("Referenced Sources"):
                for sc in source_chunks:
                    st.markdown(f"**{sc['label']}**")
                    preview = sc["content"][:400] + "…" if len(sc["content"]) > 400 else sc["content"]
                    st.caption(preview)
        elif not grounded:
            st.caption("No documents indexed — answered from LLM knowledge.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "source_chunks": source_chunks,
    })

    # Generate follow-up questions in the background (non-blocking feel via Groq)
    if grounded and sources:
        followups = generate_followup_questions(prompt, answer)
        st.session_state.last_followups = followups
        if followups:
            st.rerun()
