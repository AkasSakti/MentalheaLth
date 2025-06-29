import os, re, yaml, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import tensorflow as tf
import shap
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

# ========== PATHS ==========
base_dir = "dataset"
artifacts = os.path.join("MentalheaLth", "artifacts")
os.makedirs(artifacts, exist_ok=True)

# ========== FUNCTIONS ==========
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def load_glove(path, dim, word_index, vocab_size):
    emb = {}
    with open(path, encoding='utf8') as f:
        for line in f:
            w, *c = line.split()
            emb[w] = np.array(c, dtype='float32')
    m = np.zeros((vocab_size, dim))
    for w, i in word_index.items():
        if i < vocab_size and w in emb:
            m[i] = emb[w]
    return m

def chaos_dropout_sequence(length, x0=0.7, r=3.9):
    drops, x = [], x0
    for _ in range(length):
        x = r * x * (1 - x)
        drops.append(np.clip(x, 0.2, 0.5))
    return drops

# ========== DATA PREP ==========
df = pd.read_csv(os.path.join(base_dir, "train.csv"))
df['tweet'] = df['tweet'].astype(str).apply(clean_text)
y = df['label'].values

tokenizer = Tokenizer(num_words=20000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['tweet'])
oov_index = tokenizer.word_index['<OOV>']
seq = tokenizer.texts_to_sequences(df['tweet'])
seq = [[t if t < 20000 else oov_index for t in s] for s in seq]
X = pad_sequences(seq, maxlen=50, padding='post')

X_flat = X.reshape(len(X), -1)
X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_flat, y)
X_sm = X_sm.reshape(-1, 50)

X_train, X_val, y_train, y_val = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42)
embedding_matrix = load_glove(os.path.join(base_dir, "glove.6B.100d.txt"), 100, tokenizer.word_index, 20000)

# ========== MLFLOW ==========
mlflow.set_tracking_uri("https://dagshub.com/AkasSakti/MentalheaLth.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow.tensorflow.autolog()

with mlflow.start_run(run_name="mental_health_bilstm"):
    chaos_rates = chaos_dropout_sequence(2)
    model = Sequential([
        Embedding(input_dim=20000, output_dim=100, weights=[embedding_matrix], input_length=50, trainable=True),
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

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64,
                        callbacks=[EarlyStopping(patience=3, restore_best_weights=True),
                                   ReduceLROnPlateau(patience=2, factor=0.5)])

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

    mlflow.log_metrics({
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1_score": f1, "log_loss": logloss, "roc_auc": roc
    })

    plt.figure(figsize=(5, 5))
    sns.heatmap(confusion_matrix(y_val, val_preds), annot=True, fmt='d', cmap='Blues')
    conf_path = os.path.join(artifacts, "training_confusion_matrix.png")
    plt.savefig(conf_path); plt.close()

    p, r, _ = precision_recall_curve(y_val, val_probs)
    plt.plot(r, p)
    prc_path = os.path.join(artifacts, "training_precision_recall_curve.png")
    plt.savefig(prc_path); plt.close()

    fpr, tpr, _ = roc_curve(y_val, val_probs)
    plt.plot(fpr, tpr)
    roc_path = os.path.join(artifacts, "training_roc_curve.png")
    plt.savefig(roc_path); plt.close()

    explainer = shap.Explainer(lambda x: model(x, training=False), X_val[:50])
    sv = explainer(X_val[:50])
    viz = shap.plots.force(sv[0], matplotlib=False)
    est_path = os.path.join(artifacts, "estimator.html")
    shap.save_html(est_path, viz)

    test_df = pd.read_csv(os.path.join(base_dir, "test.csv"))
    test_df['tweet'] = test_df['tweet'].astype(str).apply(clean_text)
    seq = tokenizer.texts_to_sequences(test_df['tweet'])
    seq = [[t if t < 20000 else oov_index for t in s] for s in seq]
    test_pad = pad_sequences(seq, maxlen=50, padding='post')
    test_preds = (model.predict(test_pad) > best_t).astype(int).flatten()
    submission_path = os.path.join(artifacts, "submission.csv")
    pd.DataFrame({'id': test_df['id'], 'label': test_preds}).to_csv(submission_path, index=False)

    model_path = os.path.join(artifacts, "mental_health_model.keras")
    model.save(model_path)

    req_path = os.path.join(artifacts, "requirements.txt")
    with open(req_path, "w") as f:
        f.write("\n".join([
            "tensorflow", "mlflow", "scikit-learn", "pandas", "numpy",
            "matplotlib", "seaborn", "imbalanced-learn", "shap", "pyyaml"
        ]))

    from mlflow.utils.environment import _mlflow_conda_env
    conda_env = _mlflow_conda_env(additional_pip_deps=[
        "tensorflow", "mlflow", "scikit-learn", "pandas",
        "numpy", "matplotlib", "seaborn", "imbalanced-learn", "shap", "pyyaml"
    ])

    conda_path = os.path.join(artifacts, "conda.yaml")
    with open(conda_path, "w") as f:
        yaml.dump(conda_env, f)

    pyenv_path = os.path.join(artifacts, "python_env.yaml")
    with open(pyenv_path, "w") as f:
        yaml.dump(conda_env, f)

    for f in [model_path, submission_path, conf_path, prc_path, roc_path,
              est_path, req_path, conda_path, pyenv_path]:
        if os.path.exists(f) and os.path.getsize(f) > 0:
            mlflow.log_artifact(f)

    mlflow.tensorflow.save_model(model, path=os.path.join(artifacts, "model"))

    # ========== SAVE TF-IDF CORPUS REFERENCE ==========
    corpus_out = os.path.join(artifacts, "tfidf_reference.csv")
    df_pos = df[df['label'] == 1].sample(n=100, random_state=42)
    df_neg = df[df['label'] == 0].sample(n=100, random_state=42)
    corpus_df = pd.concat([df_pos, df_neg])
    corpus_df = corpus_df[['tweet', 'label']]
    corpus_df.to_csv(corpus_out, index=False)
    mlflow.log_artifact(corpus_out)

    print("✅ Artefak tersimpan & dilog ke MLflow/DagsHub.")
    print("✅ TF-IDF corpus referensi tersimpan:", corpus_out)
