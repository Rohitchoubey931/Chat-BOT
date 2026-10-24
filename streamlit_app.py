"""Web UI. Run with: streamlit run streamlit_app.py"""
import streamlit as st

from app.config import ALLOWED_EXTENSIONS, DOCS_DIR
from app.llm import LLMError
from app.rag import RAGPipeline

st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model...")
def load_pipeline() -> RAGPipeline:
    return RAGPipeline()


pipeline = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Sidebar: document management ----------------
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload PDF, TXT or MD files",
        type=[ext.lstrip(".") for ext in ALLOWED_EXTENSIONS],
        accept_multiple_files=True,
    )
    if uploads and st.button("Index uploaded files"):
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        for upload in uploads:
            path = DOCS_DIR / upload.name
            path.write_bytes(upload.getvalue())
            chunks = pipeline.ingest_file(path)
            st.success(f"{upload.name}: {chunks} chunks indexed")

    if st.button("Re-index docs folder"):
        st.info(pipeline.ingest_folder())

    st.subheader("Indexed files")
    docs = pipeline.list_documents()
    if not docs:
        st.caption("Nothing indexed yet.")
    for name in docs:
        col1, col2 = st.columns([4, 1])
        col1.write(name)
        if col2.button("✕", key=f"del_{name}"):
            pipeline.delete_document(name)
            st.rerun()

    voice = st.toggle("Voice answers", value=False)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- Main: chat ----------------
st.title("📚 RAG Chatbot")
st.caption("Ask questions about your documents. Answers include their sources.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for s in msg.get("sources", []):
            page = f", p. {s['page']}" if s["page"] else ""
            with st.expander(f"{s['source']}{page}  (score {s['score']})"):
                st.write(s["snippet"])

if question := st.chat_input("Ask a question about your documents"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = pipeline.ask(question, history=history)
        except LLMError as exc:
            st.error(str(exc))
            st.stop()

        st.markdown(result["answer"])
        for s in result["sources"]:
            page = f", p. {s['page']}" if s["page"] else ""
            with st.expander(f"{s['source']}{page}  (score {s['score']})"):
                st.write(s["snippet"])

        if voice and result["sources"]:
            from app.tts import text_to_speech

            try:
                st.audio(str(text_to_speech(result["answer"])))
            except Exception as exc:
                st.warning(f"Text-to-speech failed: {exc}")

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
