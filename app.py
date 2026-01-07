import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage  # Add this import at the top

# ==============================
# Helper : créer embeddings
# ==============================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(model, texts):
    return model.encode(texts, show_progress_bar=True)

# ==============================
# Helper : créer FAISS index
# ==============================
@st.cache_resource
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))
    return index

# ==============================
# Helper : rechercher les chunks
# ==============================
def retrieve_chunks(query, index, chunks, model, top_k=4):
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb, dtype=np.float32), top_k)
    results = [chunks[i] for i in I[0]]
    return results

# ==============================
# Helper : générer réponse avec Gemini
# ==============================
@st.cache_resource
def load_gemini():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0.2
    )

def generate_answer(gemini, context_chunks, question):
    context_text = "\n\n".join(context_chunks)
    
    messages = [
        SystemMessage(content=f"""
Answer the question using ONLY the following context.
If the answer is not in the context, say "I don't know".

<context>
{context_text}
</context>
"""),
        HumanMessage(content=question)
    ]
    
    response = gemini.invoke(messages)
    return response.content  # .content extracts the string answer




# ==============================
# Streamlit App
# ==============================
def main():
    st.set_page_config(layout="wide")
    st.title("📚 RAG Chatbot with Gemini-2.5-Flash")

    # Sidebar : upload PDF
    st.sidebar.title("📂 Data Loader")
    pdf_docs = st.sidebar.file_uploader(
        "Upload your PDF files", accept_multiple_files=True, type=["pdf"]
    )
    process_btn = st.sidebar.button("🚀 Process PDFs")

    # Initialize session state
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "index" not in st.session_state:
        st.session_state.index = None
    if "model" not in st.session_state:
        st.session_state.model = load_embedding_model()
    if "gemini" not in st.session_state:
        st.session_state.gemini = load_gemini()

    # Process PDFs
    if process_btn and pdf_docs:
        raw_text = ""
        for pdf in pdf_docs:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    raw_text += text

        # Split text into chunks
        chunk_size = 1000
        chunk_overlap = 200
        chunks = []
        start = 0
        while start < len(raw_text):
            end = start + chunk_size
            chunk = raw_text[start:end]
            chunks.append(chunk)
            start += chunk_size - chunk_overlap

        st.session_state.chunks = chunks
        embeddings = embed_texts(st.session_state.model, chunks)
        st.session_state.index = create_faiss_index(embeddings)

        st.success(f"✅ {len(chunks)} chunks created and FAISS index ready!")

    # Chat
    st.subheader("💬 Chatbot")
    user_question = st.text_input("Ask a question about your PDFs:")

    if user_question:
        if not st.session_state.chunks or st.session_state.index is None:
            st.warning("⚠️ Please upload and process PDFs first.")
        else:
            top_chunks = retrieve_chunks(
                user_question,
                st.session_state.index,
                st.session_state.chunks,
                st.session_state.model,
                top_k=4
            )
            answer = generate_answer(st.session_state.gemini, top_chunks, user_question)
            st.markdown(answer)

if __name__ == "__main__":
    main()
