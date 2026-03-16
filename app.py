import streamlit as st
import os
from rag_pipeline import ingest_pdf, ask_question

st.set_page_config(page_title="Lecture-Saver 3000")

st.title("🎓 Lecture-Saver 3000")
st.write("Upload your lecture PDFs and chat with them.")

# =========================
# PDF Upload
# =========================
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)


if uploaded_files:
    os.makedirs("uploads", exist_ok=True)
    saved_paths = []

    for uploaded_file in uploaded_files:
        pdf_path = os.path.join("uploads", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        saved_paths.append(pdf_path)

    if st.button("Ingest PDFs"):
        from rag_pipeline import ingest_multiple_pdfs
        ingest_multiple_pdfs(saved_paths)
        st.success("All PDFs ingested successfully!")


        

# =========================
# Chat Section
# =========================
question = st.text_input("Ask a question about the lecture:")

if st.button("Ask") and question:
    answer, sources = ask_question(question)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for src in sources:
        st.write(f"- {src}")
