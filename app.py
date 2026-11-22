import os
import tempfile
import traceback

import streamlit as st

from rag_engine import CampusQueryRAG


st.set_page_config(
    page_title="CampusQuery",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 CampusQuery")
st.caption("Transformer-Based Academic Question Answering using Hybrid Retrieval-Augmented Generation")

if "rag" not in st.session_state:
    st.session_state.rag = None
if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = None
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Evidence passages", min_value=3, max_value=7, value=5)
    alpha = st.slider(
        "Hybrid weight (dense)",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
        help="Higher values give more weight to semantic FAISS retrieval; lower values give more weight to BM25 lexical retrieval.",
    )
    st.divider()
    st.markdown("**Pipeline**")
    st.write("PDF → PyMuPDF → chunks → SentenceTransformer → FAISS + BM25 → hybrid retrieval → FLAN-T5 → answer + pages")

uploaded = st.file_uploader("Upload an academic PDF", type=["pdf"])

if uploaded is not None:
    file_changed = st.session_state.indexed_file != uploaded.name
    if file_changed:
        st.session_state.rag = None
        st.session_state.indexed_file = None

    if st.button("🔎 Build / Rebuild Index", type="primary", use_container_width=True):
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getbuffer())
                temp_path = tmp.name

            with st.spinner("Loading models and building the hybrid index..."):
                rag = CampusQueryRAG(alpha=alpha)
                num_chunks = rag.build_index(temp_path)

            st.session_state.rag = rag
            st.session_state.indexed_file = uploaded.name
            st.success(f"Index ready: {num_chunks} chunks from **{uploaded.name}**")
        except Exception as exc:
            st.error(f"Could not build the index: {exc}")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

if st.session_state.rag is not None:
    st.success(f"Indexed: {st.session_state.indexed_file}")

    st.subheader("Ask a question")
    question = st.text_area(
        "Question",
        placeholder="Example: What is RAG evaluation?",
        height=90,
    )

    if st.button("💬 Get Answer", type="primary", disabled=not question.strip(), use_container_width=True):
        with st.spinner("Retrieving evidence and generating a grounded answer..."):
            try:
                answer, evidence = st.session_state.rag.answer(question, top_k=top_k)
                st.session_state.history.insert(0, {
                    "question": question,
                    "answer": answer,
                    "evidence": evidence,
                })
            except Exception as exc:
                st.error(f"Question answering failed: {exc}")
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())

    if st.session_state.history:
        latest = st.session_state.history[0]
        st.subheader("Answer")
        st.info(latest["answer"])

        st.subheader("Retrieved evidence")
        for item in latest["evidence"]:
            with st.expander(
                f"#{item['rank']} — Page {item['page']} — Hybrid score {item['hybrid_score']:.3f}"
            ):
                st.write(item["text"])
                st.caption(
                    f"Dense: {item['dense_score']:.3f} | BM25: {item['bm25_score']:.3f} | Hybrid: {item['hybrid_score']:.3f}"
                )

        if len(st.session_state.history) > 1:
            st.subheader("Question history")
            for old in st.session_state.history[1:6]:
                st.markdown(f"**Q:** {old['question']}  \n**A:** {old['answer']}")
                st.divider()
else:
    st.info("Upload a searchable academic PDF, then click **Build / Rebuild Index**.")
    st.markdown(
        "### What this version fixes\n"
        "- Better page-aware chunking\n"
        "- FAISS semantic retrieval + BM25 lexical retrieval\n"
        "- Normalized hybrid scoring\n"
        "- Strong grounded generation prompt\n"
        "- Deterministic FLAN-T5 generation\n"
        "- Automatic fallback when the generator returns a heading/empty answer\n"
        "- Page-level evidence and retrieval scores\n"
        "- Clear error for scanned/image-only PDFs"
    )
