import streamlit as st
import os
import pdfplumber
import docx
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("API_KEY")
client = Groq(api_key=api_key)

# Initialize FAISS and Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight, fast embeddings
dimension = 384  # Output size of MiniLM embeddings
faiss_index = faiss.IndexFlatL2(dimension)
documents = []  # Stores document texts
doc_embeddings = []  # Stores embeddings

# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to summarize document
def summarize_document(text):
    prompt = f"Summarize the following legal document:\n\n{text}"
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5,
        max_completion_tokens=500,
        top_p=1,
        stream=False
    )
    
    return response.choices[0].message.content.strip()

# Function to assess risks
def assess_risks(text):
    prompt = f"Analyze the following legal document and identify all potential risks. Provide details in a structured format:\n\n{text}"
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5,
        max_completion_tokens=500,
        top_p=1,
        stream=False
    )
    
    return response.choices[0].message.content.strip()

# Function to add document to FAISS
def add_to_faiss(text):
    global faiss_index, documents, doc_embeddings
    embedding = embedding_model.encode([text])
    faiss_index.add(np.array(embedding, dtype=np.float32))
    documents.append(text)
    doc_embeddings.append(embedding)

# Function to retrieve similar documents using FAISS
def retrieve_similar(text, top_k=2):
    embedding = embedding_model.encode([text])
    _, indices = faiss_index.search(np.array(embedding, dtype=np.float32), top_k)
    return [documents[i] for i in indices[0] if i < len(documents)]

# Function for RAG-based legal chatbot
def legal_chatbot(question, text):
    retrieved_docs = retrieve_similar(question)
    context = "\n\n".join(retrieved_docs) + "\n\n" + text if retrieved_docs else text
    
    prompt = f"The user has a question about this legal document:\n\n{context}\n\nUser Question: {question}\n\nProvide a clear and concise legal response."
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5,
        max_completion_tokens=500,
        top_p=1,
        stream=False
    )
    
    return response.choices[0].message.content.strip()

# Streamlit UI
st.set_page_config(page_title="AI-Powered Legal Document Assistant", layout="wide")

st.title("📜 AI-Powered Legal Document Assistant")

# Centered File Upload
st.markdown("<h3 style='text-align: center;'>📂 Upload Your Legal Document</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"], label_visibility="collapsed")

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]

    # Extract text
    if file_extension == "pdf":
        document_text = extract_text_from_pdf(uploaded_file)
    elif file_extension == "docx":
        document_text = extract_text_from_docx(uploaded_file)
    else:
        document_text = uploaded_file.getvalue().decode("utf-8")

    # Store document in FAISS
    add_to_faiss(document_text)

    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["📄 Executive Summary", "⚠ Legal Risk Assessment", "💬 Legal Query Resolution"])

    with tab1:
        st.subheader("📄 Executive Summary")
        summary = summarize_document(document_text)
        st.write(summary)

        # Download Summary
        summary_filename = "Legal_Summary.txt"
        with open(summary_filename, "w") as f:
            f.write(summary)

        with open(summary_filename, "rb") as f:
            st.download_button("📥 Download Summary (TXT)", f, file_name=summary_filename)

    with tab2:
        st.subheader("⚠ Legal Risk Assessment")
        risks = assess_risks(document_text)
        st.write(risks)

    with tab3:
        st.subheader("💬 Legal Query Resolution")
        user_question = st.text_input("🔍 Ask a question about this document:")

        if user_question:
            chatbot_response = legal_chatbot(user_question, document_text)
            st.write("🧑‍⚖ *Response:*")
            st.write(chatbot_response)
