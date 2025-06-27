# ================== LIBRARY & SETUP ==================
import os, re, nltk, shap, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.tensorflow

# ========== SETUP PATH ==========
base_dir = r"dataset"
artifacts = os.path.join('..', 'MentalheaLth', 'artifacts')
os.makedirs(artifacts, exist_ok=True)

# ================== CHAOS DROPOUT FUNCTION ==================
def chaos_dropout_sequence(length, x0=0.7, r=3.9):
    drops, x = [], x0
    for _ in range(length):
        x = r * x * (1 - x)
        drops.append(np.clip(x, 0.2, 0.5))
    return drops

# ================== TEXT CLEANING ==================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|@\w+|#\w+|[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# ================== LOAD GLOVE ==================
def load_glove(path, dim, word_index, vocab_size):
    emb = {}
    with open(path, encoding='utf8') as f:
        for line in f:
            w, *c = line.split()
            emb[w] = np.array(c, dtype='float32')
    print("Loaded", len(emb), "GloVe vectors")
    m = np.zeros((vocab_size, dim))
    for w, idx in word_index.items():
        if idx < vocab_size and w in emb:
            m[idx] = emb[w]
    return m

# ================== DATA LOADING ==================
df = pd.read_csv(os.path.join(base_dir, "train.csv"))
df['tweet'] = df['tweet'].astype(str).apply(clean_text)
print("Distribusi label awal:", df['label'].value_counts())

# ================== TOKENIZER ==================
num_words = 20000
tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
tokenizer.fit_on_texts(df['tweet'])
oov_index = tokenizer.word_index['<OOV>']
sequences = tokenizer.texts_to_sequences(df['tweet'])
sequences = [[t if t < num_words else oov_index for t in seq] for seq in sequences]
X = pad_sequences(sequences, maxlen=50, padding='post')
y = df['label'].values

embedding_matrix = load_glove(os.path.join(base_dir, "glove.6B.100d.txt"), 100, tokenizer.word_index, num_words)

# ================== SMOTE ==================
X_flat = X.reshape(len(X), -1)
sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_resample(X_flat, y)
X_sm = X_sm.reshape(-1, 50)
print("Distribusi setelah SMOTE:", np.bincount(y_sm))

X_train, X_val, y_train, y_val = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42)

# ================== Modelling ===================
mlflow.set_tracking_uri("https://dagshub.com/AkasSakti/MentalheaLth.mlflow")
os.environ['MLFLOW_TRACKING_USERNAME'] = 'AkasSakti'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'b916cd7137017c46794da717b21ff3e3275aca0b'

mlflow.tensorflow.autolog()

with mlflow.start_run(run_name="mental_health_bilstm"):
    chaos_rates = chaos_dropout_sequence(2)
    model = Sequential([
        Embedding(input_dim=num_words, output_dim=100, weights=[embedding_matrix], input_length=50, trainable=True),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
        LSTM(32, return_sequences=False, dropout=0.2),
        BatchNormalization(),
        Dropout(chaos_rates[0]),
        Dense(32, activation='relu'),
        Dropout(chaos_rates[1]),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy', Precision(), Recall(), AUC()])
    model.summary()

    # ================== TRAIN ==================
    es = EarlyStopping(patience=3, restore_best_weights=True)
    rlr = ReduceLROnPlateau(patience=2, factor=0.5)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=10, batch_size=64, callbacks=[es, rlr])

    # ================== EVALUATION ==================
    val_probs = model.predict(X_val)
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.3, 0.7, 0.01):
        preds = (val_probs > t).astype(int)
        f = f1_score(y_val, preds)
        if f > best_f1:
            best_f1, best_t = f, t

    print(f"\nBest threshold: {best_t:.2f}, F1: {best_f1:.3f}")
    val_preds = (val_probs > best_t).astype(int)
    print(classification_report(y_val, val_preds))
    print("AUC:", roc_auc_score(y_val, val_probs))

    # ================== CONFUSION MATRIX ==================
    plt.figure(figsize=(5,5))
    sns.heatmap(confusion_matrix(y_val, val_preds), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    confusion_path = os.path.join(artifacts, "training_confusion_matrix.png")
    plt.savefig(confusion_path)
    plt.close()

    # ================== PRC ==================
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_val, val_probs)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    prc_path = os.path.join(artifacts, "training_precision_recall_curve.png")
    plt.savefig(prc_path)
    plt.close()

    # ================== ROC ==================
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_val, val_probs)
    plt.figure()
    plt.plot(fpr, tpr, marker='.')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    roc_path = os.path.join(artifacts, "training_roc_curve.png")
    plt.savefig(roc_path)
    plt.close()

    # ================== SHAP Estimator HTML (FIXED) ==================
    explainer = shap.Explainer(model, X_val[:200])
    sv = explainer(X_val[:200])
    estimator_html_path = os.path.join(artifacts, "estimator.html")
    shap_html = shap.plots.force(sv[0], matplotlib=False)
    shap.save_html(estimator_html_path, shap_html)

    # ================== SAVE MODEL ==================
    model_path = os.path.join(artifacts, "mental_health_model.h5")
    submission_path = os.path.join(artifacts, "submission.csv")
    model.save(model_path)
    print("Model disimpan di:", model_path)

    # ================== SUBMISSION ==================
    test_df = pd.read_csv(os.path.join(base_dir, "test.csv"))
    test_df['tweet'] = test_df['tweet'].astype(str).apply(clean_text)
    seq = tokenizer.texts_to_sequences(test_df['tweet'])
    seq = [[t if t < num_words else oov_index for t in s] for s in seq]
    tpad = pad_sequences(seq, maxlen=50, padding='post')
    preds = (model.predict(tpad) > best_t).astype(int).flatten()
    pd.DataFrame({'id': test_df['id'], 'label': preds}).to_csv(submission_path, index=False)
    print("✅ Selesai dan submission.csv telah dibuat di:", submission_path)

    # ================== LOG TO MLFLOW ==================
    mlflow.log_artifact(model_path)
    mlflow.log_artifact(submission_path)
    mlflow.log_artifact(confusion_path)
    mlflow.log_artifact(prc_path)
    mlflow.log_artifact(roc_path)
    mlflow.log_artifact(estimator_html_path)
    print("✅ Semua artefak dilog ke MLflow/DagsHub.")

# ================== STREAMLIT INFO ==================
print("\nUntuk menjalankan aplikasi Streamlit, gunakan perintah:")
print("streamlit run app.py")
print("Jika di server publik, akses: http://localhost:8501 atau sesuai alamat yang diberikan Streamlit.")
