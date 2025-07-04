import streamlit as st
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Handle Google genai import with better error handling
try:
    import google.generativeai as genai
except ImportError as e:
    st.error(f"Failed to import Google Generative AI library: {e}")
    st.error("Please ensure google-generativeai is correctly installed in your requirements.txt")
    st.stop()

import pandas as pd
import fitz  # PyMuPDF for PDF
from uuid import uuid4

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Debug prints to help diagnose missing/empty secrets
print("SUPABASE_URL:", SUPABASE_URL)
print("SUPABASE_KEY:", SUPABASE_KEY)
print("SUPABASE_URL type:", type(SUPABASE_URL))
print("SUPABASE_KEY type:", type(SUPABASE_KEY))
print("SUPABASE_URL length:", len(SUPABASE_URL) if SUPABASE_URL else "None")
print("SUPABASE_KEY length:", len(SUPABASE_KEY) if SUPABASE_KEY else "None")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("SUPABASE_URL and SUPABASE_KEY must be set in your environment variables or Streamlit secrets. Please check your Streamlit Cloud Secrets configuration.")
    st.error(f"SUPABASE_URL: {'SET' if SUPABASE_URL else 'MISSING'}")
    st.error(f"SUPABASE_KEY: {'SET' if SUPABASE_KEY else 'MISSING'}")
    st.stop()

# Validate URL format
if not SUPABASE_URL.startswith("https://") or not SUPABASE_URL.endswith(".supabase.co"):
    st.error(f"Invalid SUPABASE_URL format. Expected: https://xxx.supabase.co, Got: {SUPABASE_URL}")
    st.stop()

# Validate key format (should be a JWT token)
if not SUPABASE_KEY.startswith("eyJ"):
    st.error(f"Invalid SUPABASE_KEY format. Should start with 'eyJ'. Got: {SUPABASE_KEY[:10]}...")
    st.stop()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY must be set in your environment variables or Streamlit secrets.")
    st.stop()

# Configure the API key
genai.configure(api_key=GEMINI_API_KEY)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.title("📄 RAG Vector Uploader")
uploaded_file = st.file_uploader("Upload PDF, CSV, or Excel", type=["pdf", "csv", "xlsx"])

# --- Utilities ---
def extract_text(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)
        return df.to_csv(index=False)
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
        return df.to_csv(index=False)
    return ""

def chunk_text(text, chunk_size=500, overlap=200):
    # Simple chunking: split text into chunks of chunk_size characters with overlap
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        try:
            # Use the correct API for embeddings
            result = genai.embed_content(
                model="text-embedding-004",
                content=chunk,
                task_type="retrieval_document"
            )
            if result and hasattr(result, 'embedding'):
                embeddings.append({
                    "content": chunk,
                    "embedding": result.embedding
                })
            else:
                st.error("Embedding failed: No embeddings returned from Gemini API.")
        except Exception as e:
            st.error(f"Embedding failed: {e}")
    return embeddings

# --- Main Logic ---
if uploaded_file:
    st.info("🔍 Extracting text...")
    raw_text = extract_text(uploaded_file)
    st.success("✅ Text extracted.")

    st.info("✂ Chunking...")
    chunks = chunk_text(raw_text)
    st.success(f"✅ {len(chunks)} chunks created.")

    st.info("🧠 Embedding...")
    embedded_chunks = embed_chunks(chunks)
    if not embedded_chunks:
        st.error("❌ No embeddings were created. Please check your Gemini API key or quota.")
    else:
        st.success("✅ Embeddings created.")

        st.info("📦 Uploading to Supabase...")
        for i, entry in enumerate(embedded_chunks):
            try:
                supabase.table("documents").insert({
                    "file_name": uploaded_file.name,
                    "chunk": entry["content"],
                    "embedding": entry["embedding"],
                    "chunk_index": i
                }).execute()
            except Exception as e:
                st.error(f"Failed to upload chunk {i}: {e}")

        st.success("🎉 All chunks uploaded to Supabase Vector DB!")