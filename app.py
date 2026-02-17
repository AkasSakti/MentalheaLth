import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import re
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class CompatibleInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, *args, batch_shape=None, **kwargs):
        # Backward compatibility for model configs that store `batch_shape`.
        if batch_shape is not None and "batch_input_shape" not in kwargs:
            kwargs["batch_input_shape"] = tuple(batch_shape)
        super().__init__(*args, **kwargs)


class CompatibleDTypePolicy(tf.keras.mixed_precision.Policy):
    @classmethod
    def from_config(cls, config):
        return tf.keras.mixed_precision.Policy(config.get("name", "float32"))

# ================== DARK EXPRESSIONS ==================
dark_expressions = [
    "i want to die", "i wanna die", "want to kill myself", "want to end it all", "i'm done with life",
    "i wish i were dead", "i should be dead", "thinking of suicide", "i'm suicidal", "suicidal thoughts",
    "i can't go on", "i can't do this anymore", "i'm not okay", "i'm broken", "nothing matters",
    "i give up", "no reason to live", "everything is pointless", "what's the point", "i'm tired of living",
    "no one would care if i died", "no one cares", "i'm invisible", "i'm alone", "i feel empty",
    "i'm a burden", "everyone hates me", "they'd be better without me", "i hate myself", "i'm worthless",
    "crying myself to sleep", "hurts so much", "i'm always sad", "every day is a struggle", 
    "i can't stop crying", "my heart is heavy", "nothing feels real", "dead inside",
    "lost in darkness", "falling apart", "suffocating", "drowning in my thoughts", 
    "fading away", "life is meaningless", "there's no light", "trapped in my mind"
]

def contains_dark_expression(text):
    text = text.lower()
    return int(any(expr in text for expr in dark_expressions))

# ================== SETUP STREAMLIT ==================
st.set_page_config(page_title="Mental Health Tweet Classifier", layout="centered")
st.title("🧠 Mental Health Tweet Classifier")
st.markdown("BiLSTM-Based Mental Health Classification on Twitter Data with GloVe Representations.")

# ================== TEXT CLEANING ==================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^a-z\s]", '', text)
    return re.sub(r"\s+", " ", text).strip()

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    model_paths = [
        os.path.join("artifacts", "mental_health_model.keras"),
        os.path.join("artifacts", "mental_health_model.h5"),
    ]
    last_error = None
    for path in model_paths:
        if os.path.exists(path):
            try:
                return tf.keras.models.load_model(
                    path,
                    compile=False,
                    safe_mode=False,
                    custom_objects={
                        "InputLayer": CompatibleInputLayer,
                        "DTypePolicy": CompatibleDTypePolicy,
                    },
                )
            except TypeError:
                return tf.keras.models.load_model(
                    path,
                    compile=False,
                    custom_objects={
                        "InputLayer": CompatibleInputLayer,
                        "DTypePolicy": CompatibleDTypePolicy,
                    },
                )
            except Exception as e:
                last_error = e
    raise RuntimeError(
        "Failed to load model from artifacts/mental_health_model.keras or artifacts/mental_health_model.h5. "
        f"Original error: {last_error}"
    )

@st.cache_resource
def get_tokenizer_and_config():
    tokenizer_path = os.path.join("artifacts", "tokenizer.pkl")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    oov_index = tokenizer.word_index.get('<OOV>', 1)
    return tokenizer, oov_index, 20000

def load_threshold():
    threshold_path = os.path.join("artifacts", "threshold.txt")
    with open(threshold_path, "r") as f:
        t = float(f.read().strip())
        assert 0 < t < 1, "Threshold must be between 0 and 1"
        return t

@st.cache_resource
def load_tfidf_reference():
    df = pd.read_csv("artifacts/tfidf_reference.csv")
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['tweet'])
    return df, tfidf, tfidf_matrix

model = load_model()
tokenizer, oov_index, num_words = get_tokenizer_and_config()
threshold = load_threshold()
corpus_df, tfidf_vec, tfidf_matrix = load_tfidf_reference()

# ================== INFERENCE ==================
def predict_text(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    seq = [[t if t < num_words else oov_index for t in s] for s in seq]
    padded = pad_sequences(seq, maxlen=50, padding='post')
    prob = model.predict(padded, verbose=0)[0][0]
    label = "Depression" if prob > threshold else "No Depression"
    return label, prob, cleaned

def find_most_similar(cleaned_text):
    tfidf_input = tfidf_vec.transform([cleaned_text])
    scores = cosine_similarity(tfidf_input, tfidf_matrix)
    idx = np.argmax(scores)
    sim_score = scores[0][idx]
    sim_text = corpus_df.iloc[idx]['tweet']
    sim_label = corpus_df.iloc[idx]['label']
    return sim_text, sim_label, sim_score

# ================== MAIN FORM ==================
with st.form("input_form"):
    user_input = st.text_area("Enter the tweet you want to classify:")
    submit = st.form_submit_button("🔍 Predict")

if submit and user_input:
    label, prob, cleaned = predict_text(user_input)
    st.markdown(f"### Prediction Result: **{label}**")
    st.write(f"🧪 Model Probability: `{prob:.4f}`")
    st.write(f"🎯 Active Threshold: `{threshold:.2f}`")

    is_dark = contains_dark_expression(user_input)
    if prob < 0.2:
        sim_text, sim_label, sim_score = find_most_similar(cleaned)
        st.info("✳️ The model is uncertain. Here is the most similar reference from the corpus:")
        st.code(sim_text)
        st.write(f"➡️ Reference label: **{'Depression' if sim_label == 1 else 'No Depression'}**, Similarity: `{sim_score:.3f}`")

    if prob < 0.1 and is_dark:
        st.warning("⚠️ The sentence contains potentially risky expressions, but the model is uncertain. Manual review is required.")
    elif is_dark:
        st.info("The sentence contains potentially risky expressions.")

# ================== BATCH PREDIKSI ==================
st.subheader("📤 Upload CSV File for Batch Prediction")
uploaded = st.file_uploader("Upload a CSV file that contains a `tweet` column.", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    if 'tweet' not in df.columns:
        st.error("❌ Column `tweet` not found.")
    else:
        df['clean'] = df['tweet'].astype(str).apply(clean_text)
        seqs = tokenizer.texts_to_sequences(df['clean'])
        seqs = [[t if t < num_words else oov_index for t in s] for s in seqs]
        padded = pad_sequences(seqs, maxlen=50, padding='post')
        probs = model.predict(padded, verbose=0).flatten()
        df['prediction'] = ["Depression" if p > threshold else "No Depression" for p in probs]
        df['confidence'] = probs
        st.dataframe(df[['tweet', 'prediction', 'confidence']].head(10))
        st.download_button("💾 Download Results", df.to_csv(index=False), "prediction_results.csv", "text/csv")

# ================== GRAFIK EVALUASI ==================
st.subheader("📈 Model Evaluation Visualizations")
for chart in ["training_confusion_matrix.png", "training_precision_recall_curve.png", "training_roc_curve.png"]:
    path = os.path.join("artifacts", chart)
    if os.path.exists(path):
        st.image(path, caption=chart.replace("_", " ").title(), use_column_width=True)
    else:
        st.warning(f"Cannot find {chart}")
