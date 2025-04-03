# Train MobileNetV2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import cv2
import numpy as np
import os
import sqlite3

# Tải MobileNetV2 pretrained
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)  # Embedding 128 chiều
model = Model(inputs=base_model.input, outputs=x)

# Chuẩn bị dữ liệu từ data/raw/
data_dir = "data/raw/"
students = ["student01", "student02"]
images = []
for student in students:
    for img_file in os.listdir(f"{data_dir}/{student}"):
        img = cv2.imread(f"{data_dir}/{student}/{img_file}")
        img = cv2.resize(img, (224, 224)) / 255.0
        images.append(img)

# Trích xuất embedding
embeddings = model.predict(np.array(images))

# Lưu embedding vào SQLite
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS faces (id TEXT, name TEXT, embedding BLOB)")
cursor.execute("DELETE FROM faces")  # Xóa dữ liệu cũ

# Trung bình embedding cho mỗi sinh viên
for i, student in enumerate(students):
    avg_embedding = np.mean(embeddings[i*3:(i+1)*3], axis=0)  # Trung bình 3 ảnh
    cursor.execute("INSERT INTO faces (id, name, embedding) VALUES (?, ?, ?)",
                   (f"SV00{i+1}", student, avg_embedding.tobytes()))

conn.commit()
conn.close()
model.save("models/mobilenet_model.h5")
print("Đã train và lưu mô hình!")