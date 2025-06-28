import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import re
import pickle

# ================== DARK EXPRESSIONS ==================
dark_expressions = [
    # Ekspresi eksplisit ingin mati
    "i want to die", "i wanna die", "want to kill myself", "want to end it all", "i'm done with life",
    "i wish i were dead", "i should be dead", "thinking of suicide", "i'm suicidal", "suicidal thoughts",
    
    # Keputusasaan dan kehilangan harapan
    "i can't go on", "i can't do this anymore", "i'm not okay", "i'm broken", "nothing matters",
    "i give up", "no reason to live", "everything is pointless", "what's the point", "i'm tired of living",

    # Merasa tidak terlihat / tidak penting
    "no one would care if i died", "no one cares", "i'm invisible", "i'm alone", "i feel empty",
    "i'm a burden", "everyone hates me", "they'd be better without me", "i hate myself", "i'm worthless",

    # Ekspresi kesedihan mendalam
    "crying myself to sleep", "hurts so much", "i'm always sad", "every day is a struggle", 
    "i can't stop crying", "my heart is heavy", "nothing feels real", "dead inside",

    # Bahasa implisit atau metaforis
    "lost in darkness", "falling apart", "suffocating", "drowning in my thoughts", 
    "fading away", "life is meaningless", "there's no light", "trapped in my mind"
]

def contains_dark_expression(text):
    text = text.lower()
    return int(any(expr in text for expr in dark_expressions))

    

# ================== SETUP STREAMLIT ==================
st.set_page_config(page_title="Mental Health Tweet Classifier", layout="centered")
st.title("🧠 Mental Health Tweet Classifier")
st.markdown("Klasifikasi tweet terkait kesehatan mental menggunakan model BiLSTM dan GloVe.")

# ================== TEXT CLEANING ==================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^a-z\s]", '', text)
    return re.sub(r"\s+", " ", text).strip()

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("artifacts/mental_health_model.keras")

# ================== LOAD TOKENIZER FROM PICKLE ==================
@st.cache_resource
def get_tokenizer_and_config():
    tokenizer_path = os.path.join("artifacts", "tokenizer.pkl")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    oov_index = tokenizer.word_index.get('<OOV>', 1)
    return tokenizer, oov_index, 20000

# ================== LOAD THRESHOLD ==================
def load_threshold():
    threshold_path = os.path.join("artifacts", "threshold.txt")
    with open(threshold_path, "r") as f:
        t = float(f.read().strip())
        assert 0 < t < 1, "Threshold harus antara 0 dan 1"
        return t

# ================== SETUP ==================
model = load_model()
tokenizer, oov_index, num_words = get_tokenizer_and_config()
threshold = load_threshold()

# ================== INFERENCE FUNCTION ==================
def predict_text(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    seq = [[t if t < num_words else oov_index for t in s] for s in seq]
    padded = pad_sequences(seq, maxlen=50, padding='post')
    prob = model.predict(padded, verbose=0)[0][0]
    label = "Depresi" if prob > threshold else "Tidak Depresi"
    return label, prob

# ================== MAIN FORM ==================
with st.form("input_form"):
    user_input = st.text_area("Masukkan tweet yang ingin diklasifikasi:")
    submit = st.form_submit_button("🔍 Prediksi")

if submit and user_input:
    label, prob = predict_text(user_input)
    st.markdown(f"### Hasil Prediksi: **{label}**")
    st.write(f"🧪 Probabilitas model: `{prob:.4f}`")
    st.write(f"🎯 Threshold aktif: `{threshold:.2f}`")

    is_dark = contains_dark_expression(user_input)
    if prob < 0.1 and is_dark:
        st.warning("⚠️ Kalimat mengandung ekspresi berisiko, tapi model tidak yakin. Perlu review manual!")
    elif is_dark:
        st.info("Kalimat mengandung ekspresi berisiko.")

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
        probs = model.predict(padded, verbose=0).flatten()
        df['prediction'] = ["Depresi" if p > threshold else "Tidak Depresi" for p in probs]
        df['confidence'] = probs
        st.dataframe(df[['tweet', 'prediction', 'confidence']].head(10))
        st.download_button("💾 Download Hasil", df.to_csv(index=False), "prediction_results.csv", "text/csv")

# ================== GRAFIK EVALUASI ==================
st.subheader("📈 Visualisasi Evaluasi Model")
for chart in ["training_confusion_matrix.png", "training_precision_recall_curve.png", "training_roc_curve.png"]:
    path = os.path.join("artifacts", chart)
    if os.path.exists(path):
        st.image(path, caption=chart.replace("_", " ").title(), use_column_width=True)
    else:
        st.warning(f"Gagal menemukan {chart}")
