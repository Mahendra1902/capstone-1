# app.py
import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load FAISS index and logs
index = faiss.read_index("incident_faiss.index")
with open("incident_logs.pkl", "rb") as f:
    incident_logs = pickle.load(f)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Configure Gemini
genai.configure(api_key="AIzaSyBmUYQdImYbjPJesYFoMHVEfibp5l1CKBc")
model = genai.GenerativeModel("gemini-1.5-flash")

st.title("AI-Powered Industrial Safety Monitoring")

# User input
query = st.text_input("Enter safety observation or incident:")
if query:
    # Embed and search
    q_embed = embedder.encode([query])
    D, I = index.search(np.array(q_embed).astype("float32"), k=3)
    similar_incidents = [incident_logs[i] for i in I[0]]

    st.subheader("🧠 Related Past Incidents")
    for i, incident in enumerate(similar_incidents):
        st.markdown(f"**{i+1}.** {incident}")

    # Generate RAG-based recommendation
    context = "\n".join(similar_incidents)
    prompt = f"Context: {context}\n\nNew Observation: {query}\n\nWhat preventive action is recommended?"
    response = model.generate_content(prompt)

    st.subheader("✅ Recommended Action")
    st.write(response.text)
