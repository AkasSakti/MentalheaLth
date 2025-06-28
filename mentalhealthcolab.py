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
from sklearn.metrics import *
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.tensorflow
import joblib

# ========== SETUP PATH ==========
base_dir = "dataset"
artifacts = os.path.join("MentalheaLth", "artifacts")
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
X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_flat, y)
X_sm = X_sm.reshape(-1, 50)

X_train, X_val, y_train, y_val = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42)

# ================== MLFLOW SETUP ==================
mlflow.set_tracking_uri("https://dagshub.com/AkasSakti/MentalheaLth.mlflow")
mlflow_username = os.getenv('MLFLOW_TRACKING_USERNAME')
mlflow_password = os.getenv('MLFLOW_TRACKING_PASSWORD')
os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_username
os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
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

    es = EarlyStopping(patience=3, restore_best_weights=True)
    rlr = ReduceLROnPlateau(patience=2, factor=0.5)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64, callbacks=[es, rlr])

    # ================== EVALUATION ==================
    val_probs = model.predict(X_val).flatten()
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.3, 0.7, 0.01):
        preds = (val_probs > t).astype(int)
        f = f1_score(y_val, preds)
        if f > best_f1:
            best_f1, best_t = f, t
    val_preds = (val_probs > best_t).astype(int)

    acc = accuracy_score(y_val, val_preds)
    prec = precision_score(y_val, val_preds)
    rec = recall_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds)
    logloss = log_loss(y_val, val_probs)
    roc = roc_auc_score(y_val, val_probs)
    mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "log_loss": logloss, "roc_auc": roc})

    print(f"\n=== METRICS ===\nAccuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}\nLog Loss: {logloss:.4f}\nROC AUC: {roc:.4f}")

    # ================== PLOTTING ==================
    plt.figure(figsize=(5,5))
    sns.heatmap(confusion_matrix(y_val, val_preds), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    conf_path = os.path.join(artifacts, "training_confusion_matrix.png")
    plt.savefig(conf_path); plt.close()

    p, r, _ = precision_recall_curve(y_val, val_probs)
    plt.plot(r, p, marker='.')
    plt.title("Precision-Recall Curve"); plt.xlabel("Recall"); plt.ylabel("Precision")
    prc_path = os.path.join(artifacts, "training_precision_recall_curve.png")
    plt.savefig(prc_path); plt.close()

    fpr, tpr, _ = roc_curve(y_val, val_probs)
    plt.plot(fpr, tpr, marker='.')
    plt.title("ROC Curve"); plt.xlabel("FPR"); plt.ylabel("TPR")
    roc_path = os.path.join(artifacts, "training_roc_curve.png")
    plt.savefig(roc_path); plt.close()

    # ================== SHAP HTML ==================
    explainer = shap.Explainer(model, X_val[:200])
    sv = explainer(X_val[:200])
    shap_html = shap.plots.force(sv[0], matplotlib=False)
    est_path = os.path.join(artifacts, "estimator.html")
    shap.save_html(est_path, shap_html)

    # ================== MODEL SAVE & SUBMISSION ==================
    h5_path = os.path.join(artifacts, "mental_health_model.h5")
    model.save(h5_path)

    test_df = pd.read_csv(os.path.join(base_dir, "test.csv"))
    test_df['tweet'] = test_df['tweet'].astype(str).apply(clean_text)
    seq = tokenizer.texts_to_sequences(test_df['tweet'])
    seq = [[t if t < num_words else oov_index for t in s] for s in seq]
    tpad = pad_sequences(seq, maxlen=50, padding='post')
    preds = (model.predict(tpad) > best_t).astype(int).flatten()
    submission_path = os.path.join(artifacts, "submission.csv")
    pd.DataFrame({'id': test_df['id'], 'label': preds}).to_csv(submission_path, index=False)

    # ================== SAVE ARTIFACTS ==================
    model_pkl_path = os.path.join(artifacts, "model.pkl")
    joblib.dump(model, model_pkl_path)

    req_path = os.path.join(artifacts, "requirements.txt")
    with open(req_path, "w") as f:
        f.write("tensorflow\nmlflow\nscikit-learn\npandas\nnumpy\nshap\nmatplotlib\nseaborn\nimbalanced-learn\n")

    from mlflow.utils.environment import _mlflow_conda_env
    conda_env = _mlflow_conda_env(
        additional_pip_deps=["tensorflow", "mlflow", "scikit-learn", "pandas", "numpy", "shap", "matplotlib", "seaborn", "imbalanced-learn"]
    )
    conda_env_path = os.path.join(artifacts, "conda.yaml")
    with open(conda_env_path, "w") as f:
        f.write(conda_env)

    mlflow.tensorflow.save_model(model, path=os.path.join(artifacts, "model"))
    python_env_path = os.path.join(artifacts, "python_env.yaml")
    with open(python_env_path, "w") as f:
        f.write(conda_env)

    for f in [h5_path, submission_path, conf_path, prc_path, roc_path, est_path, model_pkl_path, req_path, conda_env_path, python_env_path]:
        mlflow.log_artifact(f)

    print("✅ Semua artefak berhasil dilog ke MLflow/DagsHub.")

print("\n📌 Jalankan `streamlit run app.py` untuk melihat hasil.")
