# All rights reserved.
#* Copyright (c) 2025 VidzAI
#* This software and associated documentation files are the property of VidzAI.
#* No part of this software may be copied, modified, distributed, or used 
#* without explicit permission from VidzAI.

import streamlit as st
import os
import pdfplumber
import docx
import faiss
import numpy as np
import tiktoken
import matplotlib.pyplot as plt
import requests
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("API_KEY")
client = Groq(api_key=api_key)

# Initialize FAISS and Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384  
faiss_index = faiss.IndexFlatL2(dimension)
documents = []
doc_embeddings = []

# Risk Keywords & Scores
RISK_KEYWORDS = {
    "penalty": 8, "breach": 9, "liability": 7, "compliance": 5, 
    "sanction": 8, "lawsuit": 9, "violation": 10, "termination": 6
}

# GDPR Compliance Keywords
GDPR_KEYWORDS = {
    "personal data": 8, "data processing": 9, "user consent": 7, 
    "data breach": 10, "data protection": 8, "right to be forgotten": 9, 
    "third-party sharing": 7, "cookie policy": 6, "privacy policy": 10
}

# Function to send summary via email
def send_email(user_email, document_summary):
    FORMSPREE_URL = "https://formspree.io/f/mrbpzlor"  # Replace with your Formspree endpoint
    payload = {
        "email": user_email,
        "subject": "Your Legal Document Summary",
        "message": document_summary
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(FORMSPREE_URL, json=payload, headers=headers)

    if response.status_code == 200:
        return "✅ Email sent successfully!"
    else:
        return f"❌ Failed to send email. Error: {response.text}"

# Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            text += extracted + "\n" if extracted else ""
    return text.strip()

# Extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Split large text into smaller chunks
def split_text(text, max_tokens=5000):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = [tokens[i: i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [encoding.decode(chunk) for chunk in chunks]

# Summarize large document
def summarize_large_document(text):
    text_chunks = split_text(text, max_tokens=5000)
    summaries = []
    
    for chunk in text_chunks:
        prompt = f"Summarize the following legal document section:\n\n{chunk}"
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
            max_completion_tokens=500
        )
        summaries.append(response.choices[0].message.content.strip())
    
    return "\n\n".join(summaries)

# Assess risks
def assess_risks(text):
    risk_scores = {}
    for line in text.split("\n"):
        detected_keywords = [word for word in RISK_KEYWORDS if word in line.lower()]
        if detected_keywords:
            risk_score = sum(RISK_KEYWORDS[word] for word in detected_keywords)
            risk_scores[", ".join(detected_keywords)] = risk_score
    return risk_scores

# GDPR Compliance Check
def check_gdpr_compliance(text):
    gdpr_issues = {}
    for line in text.split("\n"):
        detected_keywords = [word for word in GDPR_KEYWORDS if word in line.lower()]
        if detected_keywords:
            risk_score = sum(GDPR_KEYWORDS[word] for word in detected_keywords)
            gdpr_issues[", ".join(detected_keywords)] = risk_score
    return gdpr_issues

# Visualize Risk Analysis
def plot_risk_analysis(risk_scores, title):
    if not risk_scores:
        st.write("✅ No significant risks detected!")
        return
    
    clauses = list(risk_scores.keys())
    scores = list(risk_scores.values())
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(clauses, scores, color='red')
    ax.set_xlabel("Risk Score")
    ax.set_title(title)
    
    st.pyplot(fig)

# Add document to FAISS
def add_to_faiss(text):
    global faiss_index, documents, doc_embeddings
    embedding = embedding_model.encode([text])
    faiss_index.add(np.array(embedding, dtype=np.float32))
    documents.append(text)
    doc_embeddings.append(embedding)

# Legal Chatbot with RAG
def legal_chatbot(user_question, document_text):
    global faiss_index, documents
    
    # Embed the user question
    question_embedding = embedding_model.encode([user_question])
    
    # Search FAISS index for relevant document chunks
    distances, indices = faiss_index.search(np.array(question_embedding, dtype=np.float32), k=1)
    
    if indices[0][0] == -1 or not documents:
        return "I couldn't find relevant information in the document to answer your question."
    
    # Retrieve the most relevant chunk
    relevant_chunk = documents[indices[0][0]]
    
    # Construct prompt with retrieved context
    prompt = (
        f"You are a legal assistant specialized in compliance and GDPR. Based on the following document context, "
        f"answer the user's question:\n\n"
        f"**Document Context:**\n{relevant_chunk}\n\n"
        f"**User Question:**\n{user_question}\n\n"
        f"Provide a concise and accurate response."
    )
    
    # Call Groq API for response
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5,
        max_completion_tokens=300
    )
    
    return response.choices[0].message.content.strip()

# Streamlit UI
st.set_page_config(page_title="🔍 AI-Powered Legal Compliance Analysis", layout="wide")
st.title("📜 AI-Powered Legal Document & Summarizer ")

# Sidebar for file upload
st.sidebar.header("📂 Upload Your Legal Document")
st.sidebar.write("Note: This AI assistant is for informational purposes only and should not replace professional legal advice.")
uploaded_file = st.sidebar.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📜 Summarization", "⚠ Risk Analysis", "🛡 ChatBot & GDPR", "📥 Download Reports", "📧 Email Summary"])

    # **Tab 1: Document Summarization**
    with tab1:
        st.subheader("📜 Document Summary")
        summary = summarize_large_document(document_text)
        st.write(summary)

        # Download Summary
        summary_filename = "Legal_Summary.txt"
        with open(summary_filename, "w") as f:
            f.write(summary)

        with open(summary_filename, "rb") as f:
            st.download_button("📥 Download Summary (TXT)", f, file_name=summary_filename)

    # **Tab 2: Risk Analysis**
    with tab2:
        st.subheader("⚠ Identified Risks")
        risks = assess_risks(document_text)
        plot_risk_analysis(risks, "⚠ Risk Analysis of Legal Document")

    # **Tab 3: GDPR Compliance**
    with tab3:
        st.subheader("🛡 GDPR Compliance")
        gdpr_issues = check_gdpr_compliance(document_text)
        if gdpr_issues:
            plot_risk_analysis(gdpr_issues, "🛡 GDPR Compliance Risks")
        else:
            st.success("✅ No GDPR compliance issues detected!")
        
        st.subheader("💬 Legal Assistance Chatbot (RAG Enabled)")
        user_question = st.text_input("Ask a question about the document:")
        
        if user_question:
            chatbot_response = legal_chatbot(user_question, document_text)
            st.write("🧑‍⚖️ **Response:**")
            st.write(chatbot_response)

    # **Tab 4: Download Reports**
    with tab4:
        st.subheader("📥 Download Reports")
        
        # Generate GDPR Compliance Report
        gdpr_report = "GDPR Compliance Report:\n\n" + "\n".join([f"{k}: {v}" for k, v in gdpr_issues.items()])
        gdpr_filename = "GDPR_Compliance_Report.txt"
        with open(gdpr_filename, "w") as f:
            f.write(gdpr_report)

        with open(gdpr_filename, "rb") as f:
            st.download_button("📥 Download GDPR Compliance Report (TXT)", f, file_name=gdpr_filename)

    # **Tab 5: Email Summary**
    with tab5:
        st.subheader("📧 Receive Summary via Email")
        user_email = st.text_input("Enter your email:")
        if st.button("Send Summary"):
            if user_email:
                email_status = send_email(user_email, summary)
                st.success(email_status)
            else:
                st.warning("⚠ Please enter a valid email address.")
