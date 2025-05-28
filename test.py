import math
import time
import os
import cv2
import numpy as np
from ultralytics import YOLO

class AntiSpoofing:
    def __init__(self, model_path='models/l_version_1_300.pt'):
        """
        Khởi tạo module phát hiện khuôn mặt giả mạo sử dụng YOLO
        
        Args:
            model_path: Đường dẫn đến file mô hình YOLO
        """
        # Kiểm tra file mô hình
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại {model_path}")
        
        # Tải mô hình YOLO
        self.model = YOLO(model_path)
        print(f"Đã tải mô hình YOLO từ {model_path}")
        
        # Tên các lớp
        self.class_names = ["fake", "real"]
        
        # Ngưỡng độ tin cậy
        self.confidence_threshold = 0.6
        
        # Trạng thái xác thực
        self.is_live = False
        self.liveness_score = 0.0
        self.start_time = None
        self.verification_time = 1.0  # 1 giây để xác thực
    
    def check_liveness(self, image):
        """
        Kiểm tra tính sinh động của khuôn mặt
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            Tuple (is_live, liveness_score, processed_image)
        """
        # Tạo bản sao ảnh để vẽ kết quả
        processed_image = image.copy()
        
        # Phát hiện với YOLO
        results = self.model(image, stream=True, verbose=False)
        
        # Mặc định là không phát hiện khuôn mặt thật
        self.is_live = False
        
        # Xử lý kết quả
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Lấy tọa độ bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                
                # Lấy độ tin cậy
                confidence = float(box.conf[0])
                
                # Lấy lớp dự đoán
                cls_id = int(box.cls[0])
                
                # Chỉ xử lý kết quả có độ tin cậy cao
                if confidence > self.confidence_threshold:
                    # Khởi tạo thời gian bắt đầu nếu chưa có
                    if self.start_time is None:
                        self.start_time = time.time()
                    
                    # Xác định trạng thái và màu
                    if self.class_names[cls_id] == 'real':
                        predict_real = True
                        color = (0, 255, 0)  # Xanh lá = thật
                    else:
                        predict_real = False
                        color = (0, 0, 255)  # Đỏ = giả
                    
                    # Cập nhật điểm liveness
                    self.liveness_score = confidence
                    
                    # Kiểm tra thời gian và kết quả dự đoán
                    elapsed_time = time.time() - self.start_time
                    self.is_live = predict_real and elapsed_time >= self.verification_time
                    
                    # Vẽ hộp bao quanh khuôn mặt
                    cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Vẽ các góc (như cvzone.cornerRect)
                    l = 30  # Độ dài của đường góc
                    t = 5   # Độ dày của đường góc
                    # Góc trên bên trái
                    cv2.line(processed_image, (x1, y1), (x1 + l, y1), color, t)
                    cv2.line(processed_image, (x1, y1), (x1, y1 + l), color, t)
                    # Góc trên bên phải
                    cv2.line(processed_image, (x2, y1), (x2 - l, y1), color, t)
                    cv2.line(processed_image, (x2, y1), (x2, y1 + l), color, t)
                    # Góc dưới bên trái
                    cv2.line(processed_image, (x1, y2), (x1 + l, y2), color, t)
                    cv2.line(processed_image, (x1, y2), (x1, y2 - l), color, t)
                    # Góc dưới bên phải
                    cv2.line(processed_image, (x2, y2), (x2 - l, y2), color, t)
                    cv2.line(processed_image, (x2, y2), (x2, y2 - l), color, t)
                    
                    # Vẽ nhãn và điểm tin cậy
                    label = f"{self.class_names[cls_id].upper()} {int(confidence*100)}%"
                    
                    # Tạo background cho text
                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    text_w, text_h = text_size
                    cv2.rectangle(processed_image, (x1, max(0, y1-text_h-10)), 
                                 (x1+text_w+10, y1), color, -1)
                    
                    # Vẽ text
                    cv2.putText(processed_image, label, (x1+5, max(25, y1-5)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return self.is_live, self.liveness_score, processed_image
    
    def reset(self):
        """Reset trạng thái để bắt đầu xác thực mới"""
        self.is_live = False
        self.liveness_score = 0.0
        self.start_time = None

def test_anti_spoofing():
    """Test module chống giả mạo với mô hình YOLO"""
    model_path = 'models/l_version_1_300.pt'
    
    if not os.path.exists(model_path):
        print(f"Warning: Không tìm thấy file mô hình tại {model_path}")
        print("Vui lòng đảm bảo file mô hình có sẵn tại đường dẫn đã chỉ định")
        return
    
    try:
        anti_spoof = AntiSpoofing(model_path)
    except Exception as e:
        print(f"Lỗi khi khởi tạo mô hình: {e}")
        return
    
    # Mở camera
    cap = cv2.VideoCapture(0)
    
    # Thiết lập kích thước
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height
    
    if not cap.isOpened():
        print("Không thể mở camera!")
        return
    
    print("Nhấn 'q' để thoát hoặc 'r' để reset")
    
    # Biến đo FPS
    prev_frame_time = 0
    new_frame_time = 0
    
    try:
        while True:
            # Tính FPS
            new_frame_time = time.time()
            
            # Đọc frame từ camera
            ret, frame = cap.read()
            if not ret:
                break
            
            # Kiểm tra liveness
            is_live, score, processed_frame = anti_spoof.check_liveness(frame)
            
            # Hiển thị FPS
            fps = 1 / (new_frame_time - prev_frame_time) if new_frame_time > prev_frame_time else 0
            prev_frame_time = new_frame_time
            
            cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Hiển thị kết quả
            cv2.imshow("Anti-Spoofing with YOLO", processed_frame)
            
            # Xử lý phím nhấn
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                anti_spoof.reset()
                print("Reset liveness detection")
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_anti_spoofing()