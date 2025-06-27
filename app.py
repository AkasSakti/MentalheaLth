import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# ================== CONFIG ==================
st.set_page_config(page_title="Mental Health Tweet Classifier", layout="centered")
st.title("🧠 Mental Health Tweet Classifier")
st.markdown("Klasifikasi tweet terkait kesehatan mental menggunakan model BiLSTM dan GloVe.")

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    model_path = os.path.join("..", "MentalheaLth", "artifacts", "mental_health_model.h5")
    return tf.keras.models.load_model(model_path)

model = load_model()

# ================== LOAD TOKENIZER SETUP ==================
@st.cache_resource
def get_tokenizer_and_config():
    num_words = 20000
    oov_token = '<OOV>'
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    df = pd.read_csv(os.path.join("dataset", "train.csv"))
    df['tweet'] = df['tweet'].astype(str).apply(clean_text)
    tokenizer.fit_on_texts(df['tweet'])
    oov_index = tokenizer.word_index[oov_token]
    return tokenizer, oov_index, num_words

# ================== CLEAN TEXT ==================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^a-z\s]", '', text)
    return re.sub(r"\s+", " ", text).strip()

tokenizer, oov_index, num_words = get_tokenizer_and_config()

# ================== INFERENCE FUNCTION ==================
def predict_text(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    seq = [[t if t < num_words else oov_index for t in s] for s in seq]
    padded = pad_sequences(seq, maxlen=50, padding='post')
    prob = model.predict(padded)[0][0]
    label = "🟢 Positive" if prob > 0.5 else "🔴 Negative"
    return label, prob

# ================== MAIN FORM ==================
with st.form("input_form"):
    user_input = st.text_area("Masukkan tweet yang ingin diklasifikasi:")
    submit = st.form_submit_button("🔍 Prediksi")

if submit and user_input:
    label, prob = predict_text(user_input)
    st.markdown(f"### Hasil Prediksi: {label}")
    st.progress(int(prob * 100))
    st.markdown(f"`Probabilitas`: {prob:.4f}")

# ================== BATCH PREDIKSI ==================
st.subheader("📤 Upload File CSV untuk Prediksi Massal")
uploaded = st.file_uploader("Upload file CSV dengan kolom `tweet`", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    if 'tweet' not in df.columns:
        st.error("❌ Kolom 'tweet' tidak ditemukan.")
    else:
        df['clean'] = df['tweet'].astype(str).apply(clean_text)
        seqs = tokenizer.texts_to_sequences(df['clean'])
        seqs = [[t if t < num_words else oov_index for t in s] for s in seqs]
        padded = pad_sequences(seqs, maxlen=50, padding='post')
        probs = model.predict(padded).flatten()
        labels = ["Positive" if p > 0.5 else "Negative" for p in probs]
        df['prediction'] = labels
        df['confidence'] = probs
        st.dataframe(df[['tweet', 'prediction', 'confidence']].head(10))
        st.download_button("💾 Download Hasil", df.to_csv(index=False), "prediction_results.csv", "text/csv")

# ================== GRAFIK EVALUASI ==================
st.subheader("📈 Visualisasi Evaluasi Model")
artifact_dir = os.path.join("..", "MentalheaLth", "artifacts")
for chart in ["training_confusion_matrix.png", "training_precision_recall_curve.png", "training_roc_curve.png"]:
    path = os.path.join(artifact_dir, chart)
    if os.path.exists(path):
        st.image(path, caption=chart.replace("_", " ").title(), use_column_width=True)
    else:
        st.warning(f"Gagal menemukan {chart}")
