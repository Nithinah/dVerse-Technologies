import asyncio
import warnings
import sys
import faiss
import base64
import torch
import streamlit as st
import pandas as pd
import numpy as np
import json
from transformers import DistilBertTokenizer, DistilBertModel, GPT2Tokenizer, GPT2LMHeadModel
from sklearn.metrics.pairwise import cosine_similarity

# Fix the Issues
try:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except RuntimeError:
    pass

warnings.filterwarnings("ignore", category=UserWarning)

# CSS Styling
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)



set_background("D:\\dVERSE\\hdfc bank 1.jpg")

# Load Data and Data Preprocessing
DATA_PATH = "D:\\dVERSE\\HDFC_Faq.txt"
with open(DATA_PATH, 'r', encoding='utf-8') as file:
    data = json.load(file)

df = pd.DataFrame(data)
questions = df['question'].tolist()
answers = df['answer'].tolist()

# DistilBERT Model - Best Model Chosen for retreival among others
@st.cache_resource
def load_bert():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    return tokenizer, model

bert_tokenizer, bert_model = load_bert()

# BERT embeddings
def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

@st.cache_resource
def get_all_embeddings():
    embeddings = np.vstack([get_bert_embedding(q) for q in questions])
    return embeddings

bert_embeddings = get_all_embeddings()


# GPT-2 - Medium for Fallback - to answer questions apart from Dataset

@st.cache_resource
def load_gpt():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium") 
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    model.eval()
    return tokenizer, model

gpt_tokenizer, gpt_model = load_gpt()

# FAISS Index
index = faiss.IndexFlatL2(bert_embeddings.shape[1])
index.add(bert_embeddings)


context_history = []


# Matching and Fallback Logic

def get_answer(user_input):
    global context_history

    
    user_input = user_input.strip().lower()

    
    user_embedding = get_bert_embedding(user_input)
    D, I = index.search(user_embedding, 1)
    best_match_idx = I[0][0]
    best_score = D[0][0]

    
    input_length = len(user_input.split())
    threshold = 0.5 + 0.05 * (input_length / 10)

    if best_score < threshold:
        return answers[best_match_idx], f"Matched with similarity score: {1 - best_score:.2f}"

    
    context_history.append(user_input)
    context = "\n".join(context_history[-3:])

    prompt = f"""
Answer the following question clearly and accurately:
{context}

Question: {user_input}
Answer:
"""

    input_ids = gpt_tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        output = gpt_model.generate(
            input_ids,
            max_length=80,
            num_return_sequences=1,
            temperature=0.7,
            top_k=60,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=gpt_tokenizer.eos_token_id
        )

    generated_response = gpt_tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_response, "Generated using enhanced GPT fallback"

# Streamlit UI

st.title("🤖 HDFC Hybrid FAQ Chatbot")


user_input = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if user_input:
        response, source = get_answer(user_input)
        st.success(f"**Answer:** {response}")
        st.info(f"**Source:** {source}")
    else:
        st.warning("Please enter a question.")


# Save Models and Embeddings

import joblib

if st.button("Save Model"):
    np.save("bert_embeddings.npy", bert_embeddings)
    df.to_csv("faq_dataset.csv", index=False)
    
    gpt_model.save_pretrained("gpt_model_medium")
    bert_model.save_pretrained("bert_model_medium")
    
    st.success("Models and embeddings saved successfully!")


# Load Saved Models

if st.button("Load Model"):
    bert_embeddings = np.load("bert_embeddings.npy")
    df = pd.read_csv("faq_dataset.csv")
    
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt_model_medium")
    bert_model = DistilBertModel.from_pretrained("bert_model_medium")
    
    st.success("Models and embeddings loaded successfully!")

