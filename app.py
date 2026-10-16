from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_engine import (
    CHAT_MODEL,
    EMBEDDING_MODEL,
    answer_question,
    extract_text,
    get_collection,
    rebuild_index,
    retrieve,
)

load_dotenv()
st.set_page_config(page_title="Enterprise Knowledge Assistant | RAG Demo", page_icon="📚", layout="wide")

st.title("Enterprise Knowledge Assistant")
st.caption("Interview demo: Retrieval-Augmented Generation (RAG) powered by Gemini + ChromaDB")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. Upload or select documents\n"
        "2. Extract text and split into overlapping chunks\n"
        "3. Generate Gemini embeddings\n"
        "4. Store vectors in local ChromaDB\n"
        "5. Retrieve relevant chunks for a question\n"
        "6. Ask Gemini to answer from retrieved evidence"
    )
    st.divider()
    st.markdown(f"**Generation model:** `{CHAT_MODEL}`")
    st.markdown(f"**Embedding model:** `{EMBEDDING_MODEL}`")
    st.markdown("**Vector store:** Local ChromaDB with cosine distance")
    st.caption("Demo files and embeddings stay in your local project folder. Text is sent to Gemini API for embeddings and answer generation.")

sample_path = Path(__file__).parent / "sample_data" / "company_handbook.txt"
use_sample = st.checkbox("Include the built-in fictional company handbook", value=True)
uploaded_files = st.file_uploader(
    "Upload additional knowledge sources",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
    help="Text-based PDFs are supported. OCR for scanned PDFs is not included in this demo.",
)

if st.button("Build / Rebuild Knowledge Index", type="primary", use_container_width=True):
    input_documents = []
    if use_sample:
        input_documents.append({"source": sample_path.name, "text": sample_path.read_text(encoding="utf-8")})

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                text = extract_text(uploaded_file.name, uploaded_file.getvalue())
                input_documents.append({"source": uploaded_file.name, "text": text})
            except Exception as exc:
                st.error(f"Could not process {uploaded_file.name}: {exc}")

    if not input_documents:
        st.error("Select the sample handbook or upload at least one valid file.")
    else:
        progress = st.progress(0, text="Preparing documents and generating embeddings...")
        status = st.empty()
        try:
            def update_progress(current: int, total: int) -> None:
                progress.progress(current / total, text=f"Indexing chunks: {current}/{total}")
            total_chunks = rebuild_index(input_documents, progress_callback=update_progress)
            st.session_state["rag_indexed"] = True
            st.session_state["rag_chunk_count"] = total_chunks
            status.success(f"Index built successfully: {total_chunks} chunks from {len(input_documents)} source file(s).")
        except Exception as exc:
            status.error(f"Indexing failed: {exc}")

try:
    _, collection = get_collection()
    current_count = collection.count()
except Exception:
    current_count = 0

if current_count:
    st.success(f"Local index is ready with {current_count} chunks.")
else:
    st.info("No index found yet. Build the index above before asking questions.")

st.divider()
st.subheader("Ask your knowledge base")
question = st.text_input(
    "Question",
    placeholder="e.g., What should an engineer do after a failed production deployment?",
)
top_k = st.slider("Number of passages to retrieve (top-K)", min_value=1, max_value=8, value=4)

if st.button("Ask Gemini", use_container_width=True, disabled=(current_count == 0)):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Retrieving passages and generating an answer..."):
            try:
                passages = retrieve(question, top_k=top_k)
                answer = answer_question(question, passages)
                st.markdown("### Answer")
                st.markdown(answer)
                st.markdown("### Retrieved evidence")
                for number, passage in enumerate(passages, start=1):
                    with st.expander(
                        f"[S{number}] {passage['source']} · chunk {passage['chunk_index']} · similarity {passage['similarity']:.3f}"
                    ):
                        st.write(passage["text"])
                        st.caption(
                            f"Cosine distance: {passage['distance']:.3f}. Lower distance generally means a closer vector match."
                        )
            except Exception as exc:
                st.error(f"The question could not be processed: {exc}")

with st.expander("Try these interview demo questions"):
    st.markdown(
        "- What should an engineer include when reporting a production incident?\n"
        "- Why should API keys not be committed to source control?\n"
        "- What is the difference between RPO and RTO?\n"
        "- How should an engineer respond if a deployment causes a severe regression?\n"
        "- How can a RAG assistant handle a question that is not answered by its documents?\n"
        "- What should be considered before sending company data to an external AI service?"
    )

with st.expander("Project limitations and production improvements"):
    st.markdown(
        "- This is a local interview demo, not a production multi-user service.\n"
        "- Scanned PDF OCR, access-control filters, authentication, and automated document synchronization are not implemented.\n"
        "- The vector index is persisted locally in `chroma_db/`; rebuild the index when source documents change.\n"
        "- For production, add tenant-aware authorization before retrieval, document lifecycle handling, evaluation datasets, monitoring, rate limits, retries, and secret management.\n"
        "- Retrieved content is sent to Gemini for embeddings and answer generation. Do not upload confidential data unless your organization permits that processing."
    )
