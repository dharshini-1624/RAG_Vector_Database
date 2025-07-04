import streamlit as st
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from google import genai
from google.genai import types
import pandas as pd
import fitz  # PyMuPDF for PDF
from uuid import uuid4

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

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
            response = client.models.embed_content(
                model="text-embedding-004",
                contents=chunk
            )
            if response.embeddings and len(response.embeddings) > 0:
                embedding = response.embeddings[0].values
                embeddings.append({
                    "content": chunk,
                    "embedding": embedding
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