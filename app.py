import streamlit as st
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import requests

# Google Generative AI import and configuration
try:
    import google.generativeai as genai
except ImportError as e:
    st.error(f"Failed to import Google Generative AI library: {e}")
    st.stop()

import pandas as pd
import fitz  # PyMuPDF for PDF
from uuid import uuid4

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Debug prints
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

if not SUPABASE_URL.startswith("https://") or not SUPABASE_URL.endswith(".supabase.co"):
    st.error(f"Invalid SUPABASE_URL format. Expected: https://xxx.supabase.co, Got: {SUPABASE_URL}")
    st.stop()

if not SUPABASE_KEY.startswith("eyJ"):
    st.error(f"Invalid SUPABASE_KEY format. Should start with 'eyJ'. Got: {SUPABASE_KEY[:10]}...")
    st.stop()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY must be set in your environment variables or Streamlit secrets.")
    st.stop()

# --- Gemini Embedding via REST API ---
EMBEDDING_URL = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={GEMINI_API_KEY}"

def get_gemini_embedding(text):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "models/embedding-001",
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_DOCUMENT"
    }
    try:
        response = requests.post(EMBEDDING_URL, headers=headers, json=data)
        if response.status_code == 200:
            embedding = response.json().get("embedding", {}).get("values")
            if embedding:
                return embedding
            else:
                print("No embedding returned in response:", response.json())
                return None
        else:
            print("Gemini API error:", response.text)
            return None
    except Exception as e:
        print(f"Exception during Gemini API call: {e}")
        return None

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Test Supabase connection
try:
    # Try a simple query to test the connection
    result = supabase.table("documents").select("*", count="exact").limit(1).execute()
    print("✅ Supabase connection successful")
except Exception as e:
    st.error(f"❌ Supabase connection failed: {e}")
    st.error("Please check your SUPABASE_URL and SUPABASE_KEY")
    st.error(f"URL: {SUPABASE_URL}")
    st.error(f"Key starts with: {SUPABASE_KEY[:20] if SUPABASE_KEY else 'None'}...")
    st.stop()

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
        embedding = get_gemini_embedding(chunk)
        if embedding:
            embeddings.append({
                "content": chunk,
                "embedding": embedding
            })
        else:
            st.error("Embedding failed: No embeddings returned from Gemini API.")
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
        upload_success_count = 0
        upload_failed_count = 0
        
        for i, entry in enumerate(embedded_chunks):
            try:
                result = supabase.table("documents").insert({
                    "file_name": uploaded_file.name,
                    "chunk": entry["content"],
                    "embedding": entry["embedding"],
                    "chunk_index": i
                }).execute()
                upload_success_count += 1
                print(f"✅ Uploaded chunk {i}")
            except Exception as e:
                upload_failed_count += 1
                st.error(f"Failed to upload chunk {i}: {e}")
                # If it's an authentication error, stop trying
                if "Invalid API key" in str(e) or "authentication" in str(e).lower():
                    st.error("❌ Authentication failed. Please check your Supabase API key.")
                    st.error("Stopping upload process due to authentication errors.")
                    break

        if upload_failed_count == 0:
            st.success(f"🎉 All {upload_success_count} chunks uploaded to Supabase Vector DB!")
        elif upload_success_count > 0:
            st.warning(f"⚠️ {upload_success_count} chunks uploaded, {upload_failed_count} failed.")
        else:
            st.error("❌ No chunks were uploaded due to errors.")