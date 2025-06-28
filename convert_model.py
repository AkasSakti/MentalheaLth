import tensorflow as tf
import os

input_path = "artifacts/mental_health_model.h5"
output_path = "artifacts/mental_health_model.keras"

if os.path.exists(input_path):
    model = tf.keras.models.load_model(input_path)
    model.save(output_path)
    print(f"✅ Model dikonversi ke {output_path}")
    # opsional hapus .h5
    os.remove(input_path)
    print(f"🗑 File {input_path} dihapus.")
else:
    print(f"❌ File {input_path} tidak ditemukan.")
