# 🎓 Ứng dụng AI vào điểm danh bằng khuôn mặt có tích hợp chống gian lận

Đây là một hệ thống điểm danh tự động ứng dụng trí tuệ nhân tạo, sử dụng nhận diện khuôn mặt kết hợp với kỹ thuật chống gian lận (anti-spoofing). Mục tiêu là tăng hiệu quả quản lý lớp học, hạn chế hành vi gian lận như dùng ảnh hoặc video giả mạo để điểm danh thay. Mô hình được thiết kế với mục tiêu sử dụng cho các thiết bị nhúng có cấu hình hạn chế.

---

## ✅ Tính năng chính

- Nhận diện khuôn mặt thời gian thực từ video hoặc webcam.
- Phát hiện hành vi gian lận như sử dụng ảnh tĩnh hoặc video để giả mạo.
- Ghi nhận thời gian và danh tính người điểm danh vào hệ thống.
- Giao diện người dùng đồ họa trực quan với Dear PyGui.
- Trích xuất vector đặc trưng (embedding) và lưu trữ vào cơ sở dữ liệu.
- Tạo báo cáo điểm danh dạng `.csv`.

---

## 📁 Cấu trúc thư mục chính
```bash
├── data/                        # Dữ liệu chính của dự án
│   ├── raw/                     # Dữ liệu thô, chưa xử lý
│   ├── train/                   # Dữ liệu huấn luyện
│   ├── validation/              # Dữ liệu kiểm định
│   ├── test/                    # Dữ liệu kiểm thử
│   └── videos/                  # Video gốc để trích xuất khuôn mặt
├── models/                      # Mô hình đã huấn luyện (.tflite)
├── exports/                     # Báo cáo điểm danh (.csv)
├── src/                         # Mã nguồn chính
│   ├── main.py                  # File khởi động ứng dụng
│   ├── gui.py                   # Giao diện người dùng
│   ├── database.py              # Cơ sở dữ liệu điểm danh
│   ├── attendance.py            # Logic xử lý điểm danh
│   ├── extract_faces.py         # Trích xuất khuôn mặt từ video/ảnh
│   ├── extract_embeding.py      # Trích xuất và lưu embedding
│   ├── face_recognition.py      # Nhận diện khuôn mặt
│   ├── anti_spoofing.py         # Phát hiện gian lận
│   └── data_processing.py       # Chuẩn bị và xử lý dữ liệu
├── test_data/                   # Dữ liệu kiểm thử riêng biệt
└── requirements.txt             # Danh sách thư viện cần cài đặt
```

## 🧠 Công nghệ sử dụng
- **Python 3.9.11**
- **TensorFlow / Keras** – Huấn luyện mô hình nhận diện khuôn mặt với kiến trúc MobileNetV2
- **MobileNetV2** – Mô hình nhẹ, tối ưu hiệu suất, phù hợp triển khai trên các thiết bị nhúng hoặc di động có cấu hình hạn chế
- **OpenCV** – Xử lý ảnh và video
- **Mediapipe** – Phát hiện hành vi giả mạo (anti-spoofing)
- **DearPyGui** – Giao diện người dùng
- **SQLite3** – Quản lý thông tin sinh viên và dữ liệu điểm danh

---

## ⚙️ Cài đặt và sử dụng

### 🔽 1. Tải dự án về

```bash
git clone https://github.com/tqHungdev0605/automatic_attendance_model.git
cd automatic_attendance_model
```

### 🧪 2. Tạo môi trường ảo
```bash
python -m venv .venv

# Kích hoạt môi trường
# Trên Windows:
.venv\Scripts\activate
# Trên macOS / Linux:
source .venv/bin/activate
```

### 📦 3. Cài đặt thư viện
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### ▶️ 4. Chạy ứng dụng
```bash
python src/main.py
```


