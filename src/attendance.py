import cv2
import numpy as np
import pickle
import time
import os
from datetime import datetime

class AttendanceSystem:
    def __init__(self, database, face_model, antispoof_model):
        """
        Hệ thống điểm danh đơn giản
        
        Args:
            database: Database quản lý sinh viên
            face_model: Mô hình nhận diện khuôn mặt
            antispoof_model: Mô hình chống fake
        """
        self.db = database
        self.face_recognition = face_model
        self.antispoof = antispoof_model
        
        # Lưu trữ embedding của sinh viên
        self.student_embeddings = {}
        
        # Cài đặt
        self.confidence_threshold = 0.7  # Ngưỡng tin cậy
        self.cooldown_time = 5          # Thời gian chờ giữa các lần điểm danh (giây)
        self.last_attendance = {}       # Lưu thời gian điểm danh cuối
        
        # Load embeddings từ database
        self.load_all_embeddings()
    
    def load_all_embeddings(self):
        """Load embedding của tất cả sinh viên"""
        try:
            students = self.db.get_all_students()
            loaded_count = 0
            
            for student in students:
                student_id = student['student_id']
                embedding_path = student['embedding_path']
                
                if embedding_path and os.path.exists(embedding_path):
                    try:
                        with open(embedding_path, 'rb') as f:
                            embedding = pickle.load(f)
                        
                        self.student_embeddings[student_id] = {
                            'embedding': embedding,
                            'name': student['full_name'],
                            'class': student['class_name']
                        }
                        loaded_count += 1
                        
                    except Exception as e:
                        print(f"Lỗi load embedding {student_id}: {e}")
                else:
                    print(f"Sinh viên {student_id} chưa có dữ liệu khuôn mặt")
            
            print(f"Đã load {loaded_count} embedding sinh viên")
            
        except Exception as e:
            print(f"Lỗi load embeddings: {e}")
    
    def add_new_student(self, student_id, full_name, class_name, video_path=None):
        """
        Thêm sinh viên mới và tự động tạo embedding từ video
        
        Args:
            student_id: Mã sinh viên
            full_name: Họ tên
            class_name: Lớp
            video_path: Đường dẫn video để tạo embedding (optional)
        """
        try:
            # Tạo thư mục embedding nếu chưa có
            embedding_dir = "data/embeddings"
            os.makedirs(embedding_dir, exist_ok=True)
            
            embedding_path = None
            
            # Nếu có video, tạo embedding
            if video_path and os.path.exists(video_path):
                embedding_path = os.path.join(embedding_dir, f"{student_id}.pkl")
                
                # Tạo embedding từ video (giả sử có hàm này)
                embedding = self.create_embedding_from_video(video_path)
                if embedding is not None:
                    with open(embedding_path, 'wb') as f:
                        pickle.dump(embedding, f)
                    print(f"Đã tạo embedding từ video: {video_path}")
                else:
                    embedding_path = None
                    print("Không thể tạo embedding từ video")
            
            # Thêm vào database
            success = self.db.add_student(student_id, full_name, class_name, embedding_path)
            
            if success and embedding_path:
                # Thêm vào memory
                self.student_embeddings[student_id] = {
                    'embedding': embedding,
                    'name': full_name,
                    'class': class_name
                }
            
            return success
            
        except Exception as e:
            print(f"Lỗi thêm sinh viên: {e}")
            return False
    
    def create_embedding_from_video(self, video_path):
        """
        Tạo embedding từ video (đơn giản hóa)
        Trong thực tế sẽ trích xuất nhiều frame và tạo embedding trung bình
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            embeddings = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret or frame_count > 50:  # Lấy tối đa 50 frame
                    break
                
                # Nhận diện khuôn mặt và tạo embedding
                faces = self.face_recognition.detect_faces(frame)
                if len(faces) > 0:
                    face_embedding = self.face_recognition.get_embedding(faces[0])
                    if face_embedding is not None:
                        embeddings.append(face_embedding)
                
                frame_count += 1
            
            cap.release()
            
            if len(embeddings) > 0:
                # Trả về embedding trung bình
                return np.mean(embeddings, axis=0)
            else:
                return None
                
        except Exception as e:
            print(f"Lỗi tạo embedding từ video: {e}")
            return None
    
    def remove_student(self, student_id):
        """Xóa sinh viên"""
        try:
            # Xóa từ database (sẽ tự động xóa file embedding)
            success = self.db.delete_student(student_id)
            
            # Xóa từ memory
            if student_id in self.student_embeddings:
                del self.student_embeddings[student_id]
                print(f"Đã xóa embedding {student_id} khỏi bộ nhớ")
            
            return success
            
        except Exception as e:
            print(f"Lỗi xóa sinh viên: {e}")
            return False
    
    def process_camera_frame(self, frame):
        """
        Xử lý frame từ camera để nhận diện và điểm danh
        
        Args:
            frame: Frame từ camera
            
        Returns:
            (processed_frame, attendance_results)
        """
        if not self.student_embeddings:
            return frame, []
        
        current_time = time.time()
        attendance_results = []
        
        try:
            # Nhận diện khuôn mặt
            result_frame, detections = self.face_recognition.process_frame(frame.copy())
            
            for detection in detections:
                x, y, w, h, student_id, confidence = detection
                
                # Kiểm tra confidence
                if confidence < self.confidence_threshold:
                    self.draw_low_confidence(result_frame, x, y, w, h, confidence)
                    continue
                
                # Kiểm tra sinh viên có trong hệ thống không
                if student_id not in self.student_embeddings:
                    self.draw_unknown_face(result_frame, x, y, w, h, student_id)
                    continue
                
                # Kiểm tra cooldown
                if student_id in self.last_attendance:
                    if current_time - self.last_attendance[student_id] < self.cooldown_time:
                        self.draw_cooldown(result_frame, x, y, w, h, student_id)
                        continue
                
                # Kiểm tra chống fake
                face_crop = frame[y:y+h, x:x+w]
                is_real, fake_confidence = self.check_fake_face(face_crop)
                
                if not is_real:
                    self.draw_fake_detected(result_frame, x, y, w, h, student_id)
                    continue
                
                # Điểm danh thành công
                success = self.db.mark_attendance(student_id, confidence)
                
                if success:
                    # Cập nhật thời gian điểm danh cuối
                    self.last_attendance[student_id] = current_time
                    
                    # Vẽ kết quả thành công
                    self.draw_attendance_success(result_frame, x, y, w, h, student_id, confidence)
                    
                    # Thêm vào kết quả
                    student_info = self.student_embeddings[student_id]
                    attendance_results.append({
                        'student_id': student_id,
                        'student_name': student_info['name'],
                        'confidence': confidence,
                        'fake_confidence': fake_confidence,
                        'time': datetime.now(),
                        'success': True
                    })
                else:
                    self.draw_attendance_failed(result_frame, x, y, w, h, student_id)
            
            # Vẽ thông tin hệ thống
            self.draw_system_info(result_frame)
            
            return result_frame, attendance_results
            
        except Exception as e:
            print(f"Lỗi xử lý frame: {e}")
            return frame, []
    
    def check_fake_face(self, face_image):
        """Kiểm tra khuôn mặt fake"""
        try:
            is_real, confidence = self.antispoof.check_liveness(face_image)
            return is_real, confidence
        except Exception as e:
            print(f"Lỗi kiểm tra fake: {e}")
            return True, 1.0  # Nếu lỗi thì cho qua
    
    # ========== CÁC HÀM VẼ GIAO DIỆN ==========
    def draw_attendance_success(self, frame, x, y, w, h, student_id, confidence):
        """Vẽ điểm danh thành công"""
        student_info = self.student_embeddings[student_id]
        color = (0, 255, 0)  # Xanh lá
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, student_info['name'], (x, y - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"ID: {student_id}", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"DIEM DANH THANH CONG", (x, y + h + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Tin cay: {confidence:.2f}", (x, y + h + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def draw_fake_detected(self, frame, x, y, w, h, student_id):
        """Vẽ phát hiện fake"""
        color = (0, 0, 255)  # Đỏ
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv2.putText(frame, "PHAT HIEN GIAN LAN!", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"ID: {student_id}", (x, y + h + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def draw_unknown_face(self, frame, x, y, w, h, detected_id):
        """Vẽ khuôn mặt không xác định"""
        color = (128, 128, 128)  # Xám
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, "KHONG XAC DINH", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def draw_low_confidence(self, frame, x, y, w, h, confidence):
        """Vẽ độ tin cậy thấp"""
        color = (0, 165, 255)  # Cam
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, "DO TIN CAY THAP", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"{confidence:.2f}", (x, y + h + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def draw_cooldown(self, frame, x, y, w, h, student_id):
        """Vẽ thời gian chờ"""
        student_info = self.student_embeddings[student_id]
        color = (0, 255, 255)  # Vàng
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, student_info['name'], (x, y - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, "VUI LONG CHO...", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def draw_attendance_failed(self, frame, x, y, w, h, student_id):
        """Vẽ điểm danh thất bại"""
        color = (0, 0, 255)  # Đỏ
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, "DIEM DANH THAT BAI", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def draw_system_info(self, frame):
        """Vẽ thông tin hệ thống"""
        # Thông tin cơ bản
        total_students = len(self.student_embeddings)
        today_attendance = len(self.db.get_attendance_today())
        
        # Vẽ khung thông tin
        cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 80), (255, 255, 255), 2)
        
        # Vẽ text
        cv2.putText(frame, f"Tong sinh vien: {total_students}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Diem danh hom nay: {today_attendance}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Thoi gian: {datetime.now().strftime('%H:%M:%S')}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # ========== HÀM TIỆN ÍCH ==========
    def get_student_list(self):
        """Lấy danh sách sinh viên"""
        return self.db.get_all_students()
    
    def get_today_attendance(self):
        """Lấy điểm danh hôm nay"""
        return self.db.get_attendance_today()
    
    def get_attendance_history(self, student_id=None):
        """Lấy lịch sử điểm danh"""
        return self.db.get_attendance_history(student_id)


# ==================== TEST ===================
def test_attendance_system():
    """Test hệ thống điểm danh"""
    from database import SimpleDatabase
    
    # Mock classes
    class MockFaceModel:
        def process_frame(self, frame):
            return frame, [(100, 100, 150, 200, "SV001", 0.95)]
        
        def detect_faces(self, frame):
            return [frame[100:300, 100:250]]  # Mock face crop
        
        def get_embedding(self, face):
            return np.random.rand(128)  # Mock embedding
    
    class MockAntiSpoof:
        def check_liveness(self, face):
            return True, 0.85
    
    # Khởi tạo hệ thống
    db = SimpleDatabase("data/test_simple.db")
    face_model = MockFaceModel()
    antispoof_model = MockAntiSpoof()
    
    system = AttendanceSystem(db, face_model, antispoof_model)
    
    print("=== TEST HE THONG DIEM DANH DON GIAN ===")
    
    # Test thêm sinh viên
    system.add_new_student("SV001", "Nguyen Van A", "CNTT01")
    system.add_new_student("SV002", "Tran Thi B", "CNTT01") 
    
    # Test xử lý camera
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    processed_frame, results = system.process_camera_frame(test_frame)
    
    print(f"Ket qua diem danh: {results}")
    
    # Test lấy thông tin
    students = system.get_student_list()
    print(f"Danh sach sinh vien: {len(students)} nguoi")
    
    attendance_today = system.get_today_attendance()
    print(f"Diem danh hom nay: {len(attendance_today)} luot")
    
    print("Test hoan thanh!")

if __name__ == "__main__":
    test_attendance_system()