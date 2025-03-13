import streamlit as st
import os
import PyPDF2
import docx2txt
import spacy
import subprocess
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Ensure SpaCy model is installed
@st.cache_resource
def ensure_spacy_model():
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError:
        st.warning(f"Downloading {model_name} model...")
        subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
        return spacy.load(model_name)

# ✅ Load NLP model
nlp = ensure_spacy_model()

# 📄 Extract text from resumes (PDF/DOCX)
def extract_text_from_file(uploaded_file):
    text = ""
    if uploaded_file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    elif uploaded_file.name.endswith('.docx'):
        text = docx2txt.process(uploaded_file)
    return text

# 🔍 Preprocess text for NLP analysis
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# 📊 Rank Resumes based on Job Description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarity_scores

# 🎨 Custom CSS Styling
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(to right, #ff9966, #ff5e62);
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            color: white;
            padding: 10px;
            background: linear-gradient(to right, #6a11cb, #2575fc);
            border-radius: 10px;
            margin-bottom: 15px;
        }
        .subtitle {
            font-size: 22px;
            color: #333;
            text-align: center;
            font-weight: bold;
        }
        textarea {
            width: 100% !important;
            height: 180px !important;
            border-radius: 12px;
            border: 2px solid #3498db;
            padding: 12px;
            font-size: 18px;
            background: #f9f9f9;
        }
        .stFileUploader {
            border-radius: 12px;
            border: 2px solid #27ae60;
            background: #f5f5f5;
        }
        .stButton>button {
            background: linear-gradient(to right, #ff512f, #dd2476);
            color: white;
            padding: 14px 30px;
            border-radius: 12px;
            font-size: 20px;
            font-weight: bold;
            width: 100%;
            transition: all 0.3s;
            cursor: pointer;
        }
        .stButton>button:hover {
            background: linear-gradient(to right, #dd2476, #ff512f);
            transform: scale(1.05);
        }
        .result-box {
            padding: 15px;
            background: #e3f2fd;
            border-radius: 10px;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# 🎯 Streamlit UI Layout
st.markdown("<div class='title'>📄 AI Resume Screening & Candidate Ranking System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>🚀 Find the best candidates based on job descriptions using AI!</div><br>", unsafe_allow_html=True)

# 🔍 Job Description Input
st.subheader("🔍 Paste the Job Description Here:")
job_desc = st.text_area("", placeholder="Enter the job description...")

# 📂 Resume Upload Section
st.subheader("📂 Upload Resumes (PDF/DOCX)")
uploaded_files = st.file_uploader("Drag & drop files or click to browse", accept_multiple_files=True, type=["pdf", "docx"])

# 🚀 Analyze & Rank Button
if st.button("🔎 Analyze & Rank Candidates"):
    if not job_desc:
        st.error("❌ Please enter a job description.")
    elif not uploaded_files:
        st.error("❌ Please upload at least one resume.")
    else:
        # ✅ Preprocess job description
        job_desc_clean = preprocess_text(job_desc)

        # ✅ Extract and preprocess resumes
        resume_texts = [extract_text_from_file(f) for f in uploaded_files]
        resume_texts_clean = [preprocess_text(text) for text in resume_texts]

        # ✅ Rank Resumes
        scores = rank_resumes(job_desc_clean, resume_texts_clean)
        ranked_candidates = sorted(zip(uploaded_files, scores), key=lambda x: x[1], reverse=True)

        # 🏆 Display Results
        st.subheader("🏆 Ranked Candidates:")
        for i, (resume, score) in enumerate(ranked_candidates):
            st.markdown(f"""
                <div class='result-box'>
                    <b>{i+1}. {resume.name}</b> - 🔹 Similarity Score: <b>{score:.2f}</b>
                </div><br>
            """, unsafe_allow_html=True)
