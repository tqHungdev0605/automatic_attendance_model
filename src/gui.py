import dearpygui.dearpygui as dpg
import cv2
import threading
import time
import os
import numpy as np

# Import các module
from database import SimpleDatabase
from attendance import AttendanceSystem
from face_recognition import FaceRecognition
from anti_spoofing import AntiSpoofing

class AttendanceGUI:
    def __init__(self):
        # Khởi tạo database
        self.db = SimpleDatabase("data/students.db")
        
        # Khởi tạo AI models
        self.face_model = None
        self.antispoof_model = None
        self.attendance_system = None
        
        # Camera
        self.camera = None
        self.is_camera_running = False
        self.camera_thread = None
        self.current_frame = None
        
        # Trạng thái
        self.is_recording = False
        self.is_attending = False
        self.video_writer = None
        
        # Khởi tạo AI models
        self.init_ai_models()
        
        self.setup_gui()
    
    def init_ai_models(self):
        """Khởi tạo các mô hình AI"""
        try:
            print("Đang khởi tạo mô hình nhận diện khuôn mặt...")
            self.face_model = FaceRecognition(
                model_path="models/model.tflite",
                class_indices_path="models/class_indices.json"
            )
            print("✅ Mô hình nhận diện OK")
            
            print("Đang khởi tạo mô hình chống gian lận...")
            self.antispoof_model = AntiSpoofing()
            print("✅ Mô hình chống gian lận OK")
            
            # Khởi tạo hệ thống điểm danh
            self.attendance_system = AttendanceSystem(
                self.db, self.face_model, self.antispoof_model
            )
            print("✅ Hệ thống điểm danh OK")
            
        except Exception as e:
            print(f"⚠️ Lỗi khởi tạo AI models: {e}")
            print("Sử dụng mock models để test...")
            self.use_mock_models()
    
    def use_mock_models(self):
        """Sử dụng mock models khi không có model thật"""
        self.face_model = MockFaceRecognition()
        self.antispoof_model = MockAntiSpoofing()
        self.attendance_system = AttendanceSystem(
            self.db, self.face_model, self.antispoof_model
        )
    
    def setup_gui(self):
        """Tạo giao diện"""
        dpg.create_context()
        
        # Tạo texture cho camera
        with dpg.texture_registry():
            # Texture cho camera preview
            dpg.add_raw_texture(640, 480, np.zeros((480, 640, 4), dtype=np.float32),
                              tag="camera_texture", format=dpg.mvFormat_Float_rgba)
        
        # Cửa sổ chính
        with dpg.window(label="Hệ thống điểm danh", tag="main_window"):
            
            dpg.add_text("HE THONG DIEM DANH KHUON MAT", color=(0, 150, 255))
            dpg.add_separator()
            
            # Menu chính
            with dpg.group(horizontal=True):
                dpg.add_button(label="Them sinh vien", callback=self.show_add_window, 
                              width=150, height=35)
                dpg.add_button(label="Xoa sinh vien", callback=self.show_delete_window, 
                              width=150, height=35)
                dpg.add_button(label="Diem danh", callback=self.show_attendance_window, 
                              width=150, height=35)
            
            dpg.add_separator()
            
            # Thông tin hệ thống
            dpg.add_text("Tong sinh vien: 0", tag="total_count")
            dpg.add_text("Diem danh hom nay: 0", tag="today_count")
            dpg.add_text("Trang thai: San sang", tag="system_status", color=(0, 255, 0))
            
            dpg.add_separator()
            
            # Danh sách sinh viên
            dpg.add_text("Danh sach sinh vien:")
            with dpg.table(header_row=True, tag="student_table", 
                          borders_innerH=True, borders_outerH=True):
                dpg.add_table_column(label="Ma SV")
                dpg.add_table_column(label="Ho ten")
                dpg.add_table_column(label="Lop")
                dpg.add_table_column(label="Du lieu khuon mat")
        
        self.update_info()
        
        dpg.create_viewport(title="Diem danh khuon mat", width=900, height=700)
        dpg.setup_dearpygui()
        dpg.set_primary_window("main_window", True)
    
    def show_add_window(self):
        """Cửa sổ thêm sinh viên"""
        if dpg.does_item_exist("add_window"):
            dpg.delete_item("add_window")
        
        with dpg.window(label="Them sinh vien", tag="add_window", 
                       width=500, height=600, modal=True):
            
            dpg.add_text("THONG TIN SINH VIEN", color=(0, 150, 255))
            dpg.add_separator()
            
            dpg.add_input_text(label="Ma sinh vien", tag="add_id", width=200)
            dpg.add_input_text(label="Ho ten", tag="add_name", width=300)
            dpg.add_input_text(label="Lop", tag="add_class", width=200)
            
            dpg.add_separator()
            dpg.add_text("THU THAP DU LIEU KHUON MAT", color=(0, 150, 255))
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Bat dau quay", callback=self.start_record,
                              tag="start_record_btn", width=120)
                dpg.add_button(label="Dung quay", callback=self.stop_record,
                              tag="stop_record_btn", width=120, enabled=False)
            
            # Hiển thị camera
            dpg.add_text("Camera preview:")
            dpg.add_image("camera_texture", width=400, height=300)
            
            dpg.add_text("Trang thai: Chua bat dau", tag="record_status")
            
            dpg.add_separator()
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Luu sinh vien", callback=self.save_student, width=120)
                dpg.add_button(label="Huy", callback=self.close_add_window, width=120)
        
        # Bắt đầu camera cho preview
        self.start_camera()
    
    def show_delete_window(self):
        """Cửa sổ xóa sinh viên"""
        if dpg.does_item_exist("delete_window"):
            dpg.delete_item("delete_window")
        
        students = self.db.get_all_students()
        items = [f"{s['student_id']} - {s['full_name']}" for s in students]
        
        with dpg.window(label="Xoa sinh vien", tag="delete_window", 
                       width=400, height=300, modal=True):
            
            if items:
                dpg.add_text("Chon sinh vien can xoa:")
                dpg.add_listbox(items, tag="delete_list", num_items=8, width=350)
                dpg.add_separator()
                dpg.add_text("Canh bao: Hanh dong nay khong the hoan tac!", color=(255, 255, 0))
                
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Xac nhan xoa", callback=self.delete_student, 
                                  width=120, color=(255, 100, 100))
                    dpg.add_button(label="Huy", callback=lambda: dpg.delete_item("delete_window"), 
                                  width=120)
            else:
                dpg.add_text("Khong co sinh vien nao trong he thong")
                dpg.add_button(label="Dong", callback=lambda: dpg.delete_item("delete_window"))
    
    def show_attendance_window(self):
        """Cửa sổ điểm danh"""
        if dpg.does_item_exist("attendance_window"):
            dpg.delete_item("attendance_window")
        
        with dpg.window(label="Diem danh", tag="attendance_window", 
                       width=700, height=650):
            
            dpg.add_text("DIEM DANH BANG KHUON MAT", color=(0, 150, 255))
            dpg.add_separator()
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Bat dau diem danh", callback=self.start_attendance,
                              tag="start_att_btn", width=150)
                dpg.add_button(label="Dung diem danh", callback=self.stop_attendance,
                              tag="stop_att_btn", width=150, enabled=False)
            
            dpg.add_separator()
            
            # Hiển thị camera
            dpg.add_text("Camera diem danh:")
            dpg.add_image("camera_texture", width=500, height=375)
            
            dpg.add_separator()
            
            # Thông tin điểm danh
            dpg.add_text("THONG TIN DIEM DANH", color=(255, 200, 0))
            dpg.add_text("Trang thai: Chua bat dau", tag="attendance_status")
            dpg.add_text("Da diem danh: 0 sinh vien", tag="attendance_count")
            dpg.add_text("Vua diem danh: Chua co", tag="last_student")
            dpg.add_text("Tin cay: 0.00", tag="confidence_score")
        
        # Bắt đầu camera
        self.start_camera()
    
    def start_camera(self):
        """Bắt đầu camera"""
        if not self.camera:
            try:
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    self.show_message("Loi: Khong the mo camera!")
                    return False
                
                # Cấu hình camera
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                
                print("✅ Camera đã sẵn sàng")
                
            except Exception as e:
                self.show_message(f"Loi khoi tao camera: {e}")
                return False
        
        # Bắt đầu camera thread
        if not self.is_camera_running:
            self.is_camera_running = True
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
        
        return True
    
    def camera_loop(self):
        """Vòng lặp xử lý camera"""
        while self.is_camera_running:
            if self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    self.current_frame = frame.copy()
                    
                    # Ghi video nếu đang recording
                    if self.is_recording and self.video_writer:
                        self.video_writer.write(frame)
                    
                    # Xử lý điểm danh nếu đang attend
                    if self.is_attending:
                        processed_frame = self.process_attendance_frame(frame)
                        self.update_camera_display(processed_frame)
                    else:
                        self.update_camera_display(frame)
            
            time.sleep(0.033)  # ~30 FPS
    
    def update_camera_display(self, frame):
        """Cập nhật hiển thị camera"""
        try:
            # Chuyển đổi BGR sang RGBA
            frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            # Chuẩn hóa về [0,1]
            frame_normalized = frame_rgba.astype(np.float32) / 255.0
            # Cập nhật texture
            dpg.set_value("camera_texture", frame_normalized.flatten())
        except Exception as e:
            print(f"Lỗi cập nhật camera display: {e}")
    
    def process_attendance_frame(self, frame):
        """Xử lý frame cho điểm danh"""
        try:
            if self.attendance_system:
                processed_frame, results = self.attendance_system.process_camera_frame(frame)
                
                # Cập nhật thông tin điểm danh
                if results:
                    for result in results:
                        if result.get('success', False):
                            name = result['student_name']
                            confidence = result['confidence']
                            
                            # Cập nhật giao diện
                            dpg.set_value("last_student", f"Vua diem danh: {name}")
                            dpg.set_value("confidence_score", f"Tin cay: {confidence:.2f}")
                            
                            # Cập nhật số lượng
                            today_count = len(self.db.get_attendance_today())
                            dpg.set_value("attendance_count", f"Da diem danh: {today_count} sinh vien")
                            
                            print(f"✅ Điểm danh: {name} (tin cậy: {confidence:.2f})")
                
                return processed_frame
            else:
                return frame
                
        except Exception as e:
            print(f"Lỗi xử lý điểm danh: {e}")
            return frame
    
    def start_record(self):
        """Bắt đầu quay video"""
        if not self.start_camera():
            return
        
        try:
            # Khởi tạo video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter("temp_student_video.avi", fourcc, 20.0, (640, 480))
            
            self.is_recording = True
            dpg.configure_item("start_record_btn", enabled=False)
            dpg.configure_item("stop_record_btn", enabled=True)
            dpg.set_value("record_status", "Trang thai: Dang quay video...")
            
            print("🎬 Bắt đầu quay video")
            
        except Exception as e:
            self.show_message(f"Loi bat dau quay: {e}")
    
    def stop_record(self):
        """Dừng quay video"""
        self.is_recording = False
        dpg.configure_item("start_record_btn", enabled=True)
        dpg.configure_item("stop_record_btn", enabled=False)
        dpg.set_value("record_status", "Trang thai: Da dung quay")
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        print("⏹️ Đã dừng quay video")
    
    def save_student(self):
        """Lưu sinh viên mới"""
        student_id = dpg.get_value("add_id").strip()
        name = dpg.get_value("add_name").strip()
        class_name = dpg.get_value("add_class").strip()
        
        # Kiểm tra thông tin
        if not all([student_id, name, class_name]):
            self.show_message("Vui long nhap day du thong tin!")
            return
        
        # Kiểm tra video
        if not os.path.exists("temp_student_video.avi"):
            self.show_message("Vui long quay video khuon mat truoc!")
            return
        
        try:
            # Thêm sinh viên với video
            success = self.attendance_system.add_new_student(
                student_id, name, class_name, "temp_student_video.avi"
            )
            
            if success:
                self.show_message(f"Them sinh vien {name} thanh cong!")
                
                # Xóa file video tạm
                if os.path.exists("temp_student_video.avi"):
                    os.remove("temp_student_video.avi")
                
                # Cập nhật giao diện
                self.update_info()
                self.close_add_window()
                
                print(f"✅ Đã thêm sinh viên: {name}")
                
            else:
                self.show_message("Khong the them sinh vien! Co the ma SV da ton tai.")
                
        except Exception as e:
            self.show_message(f"Loi them sinh vien: {e}")
    
    def close_add_window(self):
        """Đóng cửa sổ thêm sinh viên"""
        # Dừng recording nếu đang quay
        if self.is_recording:
            self.stop_record()
        
        # Xóa file tạm nếu có
        if os.path.exists("temp_student_video.avi"):
            os.remove("temp_student_video.avi")
        
        dpg.delete_item("add_window")
    
    def delete_student(self):
        """Xóa sinh viên"""
        selected = dpg.get_value("delete_list")
        if not selected:
            self.show_message("Vui long chon sinh vien!")
            return
        
        student_id = selected.split(" - ")[0]
        
        try:
            success = self.attendance_system.remove_student(student_id)
            
            if success:
                self.show_message(f"Xoa sinh vien {student_id} thanh cong!")
                self.update_info()
                dpg.delete_item("delete_window")
                print(f"🗑️ Đã xóa sinh viên: {student_id}")
            else:
                self.show_message("Khong the xoa sinh vien!")
                
        except Exception as e:
            self.show_message(f"Loi xoa sinh vien: {e}")
    
    def start_attendance(self):
        """Bắt đầu điểm danh"""
        if not self.start_camera():
            return
        
        self.is_attending = True
        dpg.configure_item("start_att_btn", enabled=False)
        dpg.configure_item("stop_att_btn", enabled=True)
        dpg.set_value("attendance_status", "Trang thai: Dang diem danh...")
        
        print("📷 Bắt đầu điểm danh")
    
    def stop_attendance(self):
        """Dừng điểm danh"""
        self.is_attending = False
        dpg.configure_item("start_att_btn", enabled=True)
        dpg.configure_item("stop_att_btn", enabled=False)
        dpg.set_value("attendance_status", "Trang thai: Da dung")
        
        print("⏹️ Đã dừng điểm danh")
    
    def update_info(self):
        """Cập nhật thông tin hệ thống"""
        students = self.db.get_all_students()
        today = self.db.get_attendance_today()
        
        dpg.set_value("total_count", f"Tong sinh vien: {len(students)}")
        dpg.set_value("today_count", f"Diem danh hom nay: {len(today)}")
        
        # Cập nhật bảng
        self.update_table(students)
    
    def update_table(self, students):
        """Cập nhật bảng sinh viên"""
        # Xóa hàng cũ
        if dpg.does_item_exist("student_table"):
            children = dpg.get_item_children("student_table", slot=1)
            for child in children:
                dpg.delete_item(child)
        
        # Thêm hàng mới
        for student in students:
            with dpg.table_row(parent="student_table"):
                dpg.add_text(student['student_id'])
                dpg.add_text(student['full_name'])
                dpg.add_text(student['class_name'])
                
                # Trạng thái dữ liệu khuôn mặt
                if student['embedding_path'] and os.path.exists(student['embedding_path']):
                    dpg.add_text("Co", color=(0, 255, 0))
                else:
                    dpg.add_text("Chua co", color=(255, 255, 0))
    
    def show_message(self, message):
        """Hiển thị thông báo"""
        if dpg.does_item_exist("message_popup"):
            dpg.delete_item("message_popup")
        
        with dpg.window(label="Thong bao", tag="message_popup", 
                       width=300, height=120, modal=True):
            dpg.add_text(message)
            dpg.add_separator()
            dpg.add_button(label="OK", callback=lambda: dpg.delete_item("message_popup"), 
                          width=100)
    
    def run(self):
        """Chạy ứng dụng"""
        dpg.show_viewport()
        
        try:
            while dpg.is_dearpygui_running():
                dpg.render_dearpygui_frame()
        except KeyboardInterrupt:
            print("Ứng dụng đã được dừng")
        
        self.cleanup()
    
    def cleanup(self):
        """Dọn dẹp tài nguyên"""
        print("Đang dọn dẹp tài nguyên...")
        
        self.is_camera_running = False
        self.is_attending = False
        self.is_recording = False
        
        if self.camera:
            self.camera.release()
        
        if self.video_writer:
            self.video_writer.release()
        
        # Xóa file tạm
        temp_files = ["temp_student_video.avi"]
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
        
        dpg.destroy_context()
        print("✅ Dọn dẹp hoàn tất")


# Mock classes khi không có model thật
class MockFaceRecognition:
    def __init__(self, *args, **kwargs):
        print("⚠️ Sử dụng Mock Face Recognition")
    
    def process_frame(self, frame):
        # Giả lập nhận diện được sinh viên SV001
        return frame, [(100, 100, 150, 200, "SV001", 0.95)]
    
    def detect_faces(self, frame):
        return [frame[100:300, 100:250]]
    
    def get_embedding(self, face):
        return np.random.rand(128)

class MockAntiSpoofing:
    def __init__(self):
        print("⚠️ Sử dụng Mock Anti-Spoofing")
    
    def check_liveness(self, face_image):
        return True, 0.85


def main():
    """Chạy ứng dụng chính"""
    try:
        # Tạo thư mục cần thiết
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/embeddings", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        print("🚀 Khởi động hệ thống điểm danh...")
        
        # Khởi tạo và chạy ứng dụng
        app = AttendanceGUI()
        app.run()
        
    except Exception as e:
        print(f"❌ Lỗi chạy ứng dụng: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()