import cv2
import numpy as np
import os
import time
from datetime import datetime
from extract_embeding import FaceRecognition
from database import FaceDatabase
from anti_spoofing import AntiSpoofing
from scipy.spatial.distance import cosine
import csv

class AttendanceSystem:
    def __init__(self, model_path, db_path="face_attendance.db", threshold=0.6):
        self.face_recognizer = FaceRecognition(model_path, threshold)
        self.db = FaceDatabase(db_path)
        self.anti_spoof = AntiSpoofing()
        self.threshold = threshold

    def add_student_from_image(self, image_path, student_id, name):
        """Thêm sinh viên từ ảnh"""
        img = cv2.imread(image_path)
        if img is None:
            return False, "Không đọc được ảnh. Kiểm tra đường dẫn hoặc định dạng file."

        ok, face_img, _ = self.face_recognizer.detect_face(img)
        if not ok:
            return False, "Không phát hiện khuôn mặt trong ảnh."

        embedding = self.face_recognizer.extract_embedding(face_img)

        os.makedirs("student_faces", exist_ok=True)
        face_path = f"student_faces/{student_id}.jpg"
        cv2.imwrite(face_path, face_img)

        if self.db.add_student(student_id, name, embedding, face_path):
            return True, "Thêm sinh viên thành công."
        return False, "Sinh viên đã tồn tại."

    def add_student_from_camera(self, student_id, name):
        """Thêm sinh viên từ camera"""
        max_retries = 3
        cap = None
        for attempt in range(max_retries):
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                break
            time.sleep(1)
        if not cap or not cap.isOpened():
            return False, f"Không mở được camera sau {max_retries} lần thử."

        captured = False
        start = time.time()
        countdown = 3

        print(f"Đang đăng ký sinh viên: {name} (ID: {student_id})")
        print("Hướng mặt về camera và chờ đếm ngược...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_show = frame.copy()
            detected, face, (x, y, w, h) = self.face_recognizer.detect_face(frame)

            if detected:
                cv2.rectangle(frame_show, (x, y), (x+w, y+h), (0, 255, 0), 2)
                elapsed = time.time() - start
                remain = max(0, countdown - int(elapsed))
                cv2.putText(frame_show, f"Chup sau: {remain}s", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if remain == 0:
                    captured = True
                    break
            else:
                start = time.time()
                cv2.putText(frame_show, "Khong thay khuon mat", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(frame_show, "Nhan 'q' de thoat", (10, frame_show.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Dang ky khuon mat", frame_show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if not captured:
            return False, "Không chụp được mặt."

        embedding = self.face_recognizer.extract_embedding(face)
        os.makedirs("student_faces", exist_ok=True)
        face_path = f"student_faces/{student_id}.jpg"
        cv2.imwrite(face_path, face)

        if self.db.add_student(student_id, name, embedding, face_path):
            return True, "Thêm sinh viên thành công."
        return False, "Sinh viên đã tồn tại."

    def delete_student(self, student_id):
        """Xóa sinh viên"""
        student = self.db.get_student(student_id)
        if not student:
            return False, f"Không tìm thấy sinh viên {student_id}."

        ok = self.db.delete_student(student_id)
        if ok and student['photo_path'] and os.path.exists(student['photo_path']):
            try:
                os.remove(student['photo_path'])
            except:
                pass
        return ok, "Đã xóa sinh viên." if ok else "Xóa thất bại."

    def run_attendance(self):
        """Chạy hệ thống điểm danh"""
        max_retries = 3
        cap = None
        for attempt in range(max_retries):
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                break
            time.sleep(1)
        if not cap or not cap.isOpened():
            return False, "Không mở được camera sau {} lần thử.".format(max_retries), []

        students = self.db.get_all_students()
        if not students:
            cap.release()
            return False, "Không có sinh viên trong cơ sở dữ liệu.", []

        # Lấy danh sách sinh viên đã điểm danh hôm nay
        today = datetime.now().strftime("%Y-%m-%d")
        today_attendance = self.db.get_attendance(today)
        attended_today = {record['student_id'] for record in today_attendance}
        
        last_detection = {}  # Để tránh nhận diện liên tục cùng 1 người
        attended_students = []
        
        print("Bắt đầu điểm danh...")
        print("Nhấn 'q' để kết thúc")
        print(f"Sinh viên đã điểm danh hôm nay: {len(attended_today)}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_show = frame.copy()
            liveness, processed_image = self.anti_spoof.check_liveness(frame)
            
            if not liveness:
                frame_show = processed_image
            else:
                frame_show = frame.copy()
                detected, face, (x, y, w, h) = self.face_recognizer.detect_face(frame)
                
                if detected:
                    cv2.rectangle(frame_show, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    embedding = self.face_recognizer.extract_embedding(face)
                    max_similarity = 0
                    best_student = None

                    for student in students:
                        student_embedding = self.db.embedding_from_db(student['embedding'])
                        current_embedding = self.face_recognizer.extract_embedding(face)
                        distance = cosine(current_embedding, student_embedding)
                        similarity = 1 - distance
                        
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_student = student

                    if max_similarity >= self.threshold:
                        current_time = time.time()
                        student_id = best_student['id']
                        
                        # Kiểm tra xem sinh viên đã điểm danh hôm nay chưa
                        if student_id in attended_today:
                            cv2.putText(frame_show, f"Da diem danh: {best_student['name']}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        else:
                            # Kiểm tra thời gian để tránh nhận diện liên tục
                            if student_id not in last_detection or (current_time - last_detection[student_id] > 3):
                                # Thực hiện điểm danh
                                if self.db.mark_attendance(student_id, max_similarity):
                                    attended_today.add(student_id)  # Thêm vào danh sách đã điểm danh
                                    last_detection[student_id] = current_time
                                    
                                    attended_students.append({
                                        'id': student_id,
                                        'name': best_student['name'],
                                        'time': datetime.now().strftime("%H:%M:%S"),
                                        'similarity': max_similarity
                                    })
                                    
                                    cv2.putText(frame_show, f"Diem danh thanh cong: {best_student['name']}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    self.anti_spoof.reset()
                                    print(f"✓ Điểm danh: {best_student['name']} (ID: {student_id}) - Độ chính xác: {max_similarity:.2f}")
                                else:
                                    cv2.putText(frame_show, f"Loi diem danh: {best_student['name']}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            else:
                                cv2.putText(frame_show, f"Vua nhan dien: {best_student['name']}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    else:
                        cv2.putText(frame_show, "Khong nhan dien duoc", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(frame_show, "Khong thay khuon mat", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Hiển thị thông tin
            cv2.putText(frame_show, f"Da diem danh hom nay: {len(attended_today)} sinh vien", (10, frame_show.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame_show, f"Moi diem danh: {len(attended_students)} sinh vien", (10, frame_show.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame_show, "Nhan 'q' de thoat", (10, frame_show.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Diem danh", frame_show)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return True, "Diem danh hoan tat.", attended_students

    def view_attendance(self, date=None):
        """Xem danh sách điểm danh"""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        records = self.db.get_attendance(date)
        if not records:
            return [], f"Khong co du lieu diem danh ngay {date}."
        return records, f"{len(records)} sinh vien da diem danh ngay {date}."

    def print_student_list(self):
        """In danh sách sinh viên"""
        students = self.db.get_all_students()
        if not students:
            print("Khong co sinh vien nao trong he thong.")
            return
        
        print("\n" + "="*60)
        print("DANH SACH SINH VIEN")
        print("="*60)
        print(f"{'STT':<5} {'ID':<15} {'Ten':<30}")
        print("-"*60)
        
        for i, student in enumerate(students, 1):
            print(f"{i:<5} {student['id']:<15} {student['name']:<30}")
        
        print("-"*60)
        print(f"Tong cong: {len(students)} sinh vien")
        print("="*60)

    def print_attendance_list(self, date=None):
        """In danh sách điểm danh"""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
            
        records, message = self.view_attendance(date)
        
        if not records:
            print(f"\nKhông có dữ liệu điểm danh cho ngày {date}")
            return
        
        print("\n" + "="*80)
        print(f"DANH SÁCH ĐIỂM DANH NGÀY {date}")
        print("="*80)
        print(f"{'STT':<5} {'ID':<15} {'Tên':<25} {'Giờ':<10} {'Độ chính xác':<12}")
        print("-"*80)
        
        for i, record in enumerate(records, 1):
            similarity = f"{record['similarity']:.2f}" if record['similarity'] else "N/A"
            print(f"{i:<5} {record['student_id']:<15} {record['name']:<25} {record['time']:<10} {similarity:<12}")
        
        print("-"*80)
        print(f"Tổng cộng: {len(records)} sinh viên đã điểm danh")
        print("="*80)

    def export_attendance_to_csv(self, output_path, date=None):
        """Xuất dữ liệu điểm danh ra file CSV"""
        records, _ = self.view_attendance(date)
        if not records:
            return False, "Không có dữ liệu để xuất."

        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['id', 'student_id', 'name', 'date', 'time', 'status', 'similarity'])
                writer.writeheader()
                for r in records:
                    writer.writerow(r)
            return True, f"Xuất thành công: {output_path}."
        except Exception as e:
            return False, f"Lỗi khi xuất: {str(e)}."

    def get_statistics(self):
        """Lấy thống kê tổng quan"""
        students = self.db.get_all_students()
        all_attendance = self.db.get_all_attendance()
        today_attendance = self.db.get_attendance()
        
        return {
            'total_students': len(students),
            'total_attendance_records': len(all_attendance),
            'today_attendance': len(today_attendance)
        }