# Streamlit-based RAG Chatbot for Driving Theory Book

import os
#import time
import numpy as np
import polars as pl
import streamlit as st
from google import genai
from google.genai import types

# Set up Gemini client (ensure your API key is available as an env variable)
client = genai.Client(api_key=os.getenv("API_KEY"))


# === Embedding Generator ===
def create_embeddings(text, model="models/embedding-001", task_type="SEMANTIC_SIMILARITY"):
    try:
        response = client.models.embed_content(
            model=model,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type)
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"Embedding Error: {e}")
        return np.zeros(768)

# === Cosine Similarity ===
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# === Vector Storage ===
class VectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add(self, text, vector, meta):
        self.vectors.append(np.array(vector))
        self.texts.append(text)
        self.metadata.append(meta)

    def semantic_search(self, query_vector, k=10):
        scores = [(i, cosine_similarity(query_vector, v)) for i, v in enumerate(self.vectors)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [{"text": self.texts[i], "meta": self.metadata[i]} for i, _ in scores[:k]]

    def save(self, file_path):
        df = pl.DataFrame({"vectors": self.vectors, "texts": self.texts, "metadata": self.metadata})
        df.write_parquet(file_path)

    def load(self, file_path):
        df = pl.read_parquet(file_path)
        self.vectors = df["vectors"].to_list()
        self.texts = df["texts"].to_list()
        self.metadata = df["metadata"].to_list()

# === Save/Load Helpers ===
def save_vector_store(store, file_path="embeddings.parquet"):
    store.save(file_path)
    print(f"Saved vector store to {file_path}")

def load_vector_store(file_path="embeddings.parquet"):
    store = VectorStore()
    store.load(file_path)
    print(f"Loaded vector store from {file_path}")
    return store

# === Answer Generator ===
def generate_answer(query, matched_sentences):
    if not matched_sentences:
        return "I don't know."

    context = "\n".join([entry["text"] for entry in matched_sentences])
    system_prompt = (
        "You are a helpful assistant. Use the provided context to answer the user's question.\n"
        "If the answer is clearly stated or implied in the context, provide it concisely.\n"
        "If it's not found in the context, say 'I don't know.' Do not make up information."
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Question: {query}\n\nContext:\n{context}",
            config=types.GenerateContentConfig(system_instruction=system_prompt)
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Error: Unable to generate response."



# === Streamlit App ===
st.set_page_config(page_title="Driving Theory Chatbot", layout="wide")
st.title(" 🚗 Driving Theory RAG Chatbot")

# Load or build the vector store
@st.cache_resource
def load_vector_store_cached():
    store = VectorStore()
    store.load("embeddings.parquet")
    return store
store = load_vector_store_cached()

# Input box
question = st.text_input("Ask a question from the Driving Theory Book:")

if question:
    with st.spinner("Retrieving relevant information..."):
        query_vector = create_embeddings(question)
        top_sentences = store.semantic_search(query_vector, k=5)
        answer = generate_answer(question, top_sentences)

    st.markdown("### Answer ✨")
    st.write(answer)
    st.divider()

    st.markdown("###  Context")
    for s in top_sentences:
        st.markdown(f"- *{s['text']}* (Page {s['meta']['page']}, Sentence {s['meta']['index']})")
