import dearpygui.dearpygui as dpg
import cv2
import numpy as np
import threading
import time
import os
import json
from datetime import datetime
import queue

# Import các module của dự án
from face_recognition import FaceRecognition
from anti_spoofing import AntiSpoofing

class AttendanceGUI:
    def __init__(self):
        """Khởi tạo giao diện điểm danh"""
        self.camera = None
        self.camera_thread = None
        self.is_running = False
        self.current_frame = None
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Khởi tạo các module AI
        self.face_recognition = None
        self.anti_spoofing = AntiSpoofing()
        
        # Trạng thái ứng dụng
        self.anti_spoofing_enabled = True  # Luôn bật chống gian lận
        self.last_recognition_time = 0
        self.recognition_cooldown = 1.0  # Giảm cooldown xuống 1 giây
        
        # Trạng thái kiểm tra liveness
        self.pending_recognitions = {}  # Lưu kết quả nhận diện chờ xác thực
        self.liveness_verified_time = 0
        self.liveness_valid_duration = 10.0  # Liveness có hiệu lực 10 giây
        
        # Dữ liệu điểm danh
        self.attendance_records = []
        self.student_list = []
        
        # Kích thước cửa sổ
        self.window_width = 1200
        self.window_height = 800
        self.camera_width = 640
        self.camera_height = 480
        
        # Khởi tạo DearPyGui
        dpg.create_context()
        self.setup_gui()

    def setup_gui(self):
        """Thiết lập giao diện người dùng"""
        # Tạo viewport
        dpg.create_viewport(
            title="Hệ thống điểm danh bằng nhận diện khuôn mặt - Tích hợp chống gian lận",
            width=self.window_width,
            height=self.window_height,
            resizable=False
        )
        
        # Tạo texture cho camera
        self.setup_camera_texture()
        
        # Tạo cửa sổ chính
        with dpg.window(label="Main Window", tag="main_window", no_close=True, no_move=True, no_resize=True):
            dpg.set_primary_window("main_window", True)
            
            # Header
            with dpg.group(horizontal=True):
                dpg.add_text("HE THONG DIEM DANH THONG MINH - CHONG GIAN LAN TICH HOP", 
                           color=[255, 255, 255], tag="title_text")
                dpg.add_spacer(width=50)
                dpg.add_text("", tag="current_time")
            
            dpg.add_separator()
            
            # Main content - chia thành 2 cột
            with dpg.group(horizontal=True):
                # Cột trái - Camera và điều khiển
                with dpg.child_window(width=680, height=600, border=True, tag="left_panel"):
                    dpg.add_text("CAMERA & NHAN DIEN THONG MINH", color=[100, 200, 100])
                    dpg.add_separator()
                    
                    # Camera display
                    dpg.add_image("camera_texture", width=self.camera_width, height=self.camera_height)
                    
                    dpg.add_separator()
                    
                    # Điều khiển camera
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="BAT CAMERA", callback=self.start_camera, tag="start_camera_btn")
                        dpg.add_button(label="TAT CAMERA", callback=self.stop_camera, tag="stop_camera_btn")
                        dpg.add_button(label="CHUP ANH", callback=self.capture_photo, tag="capture_btn")
                        dpg.add_button(label="RESET XAC THUC", callback=self.reset_verification, tag="reset_btn")
                    
                    dpg.add_separator()
                    
                    # Cấu hình bảo mật
                    dpg.add_text("CAU HINH BAO MAT:")
                    dpg.add_checkbox(
                        label="Bat kiem tra chong gia mao", 
                        default_value=True,
                        callback=self.toggle_anti_spoofing,
                        tag="anti_spoofing_checkbox"
                    )
                    with dpg.group(horizontal=True):
                        dpg.add_text("Thoi gian hieu luc xac thuc:")
                        dpg.add_slider_float(
                            label="giay", 
                            default_value=10.0,
                            min_value=5.0,
                            max_value=30.0,
                            callback=self.update_liveness_duration,
                            tag="liveness_duration_slider",
                            width=150
                        )
                    
                    dpg.add_separator()
                    
                    # Trạng thái hệ thống
                    dpg.add_text("TRANG THAI HE THONG:")
                    with dpg.group():
                        dpg.add_text("Model: Not loaded", tag="model_status", color=[255, 100, 100])
                        dpg.add_text("Camera: Disconnected", tag="camera_status", color=[255, 100, 100])
                        dpg.add_text("Liveness: Standby", tag="liveness_status", color=[200, 200, 200])
                        dpg.add_text("Recognition: Standby", tag="recognition_status", color=[200, 200, 200])
                
                # Cột phải - Kết quả và điều khiển
                with dpg.child_window(width=500, height=600, border=True, tag="right_panel"):
                    dpg.add_text("KET QUA & QUAN LY", color=[100, 200, 100])
                    dpg.add_separator()
                    
                    # Load model section
                    dpg.add_text("TAI MODEL:")
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(
                            default_value="models/model.tflite",
                            width=300,
                            tag="model_path_input"
                        )
                        dpg.add_button(label="TAI MODEL", callback=self.load_model)
                    
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(
                            default_value="models/class_indices.json",
                            width=300,
                            tag="class_indices_path_input"
                        )
                        dpg.add_button(label="TAI DANH SACH LOP", callback=self.load_class_indices)
                    
                    dpg.add_separator()
                    
                    # Quy trình xác thực
                    dpg.add_text("QUY TRINH XAC THUC:")
                    with dpg.group():
                        dpg.add_text("Buoc 1: Phat hien khuon mat", tag="step1_status", color=[200, 200, 200])
                        dpg.add_text("Buoc 2: Kiem tra tinh song", tag="step2_status", color=[200, 200, 200])
                        dpg.add_text("Buoc 3: Nhan dien danh tinh", tag="step3_status", color=[200, 200, 200])
                        dpg.add_text("Buoc 4: Ghi nhan diem danh", tag="step4_status", color=[200, 200, 200])
                    
                    dpg.add_separator()
                    
                    # Kết quả nhận diện hiện tại
                    dpg.add_text("KET QUA HIEN TAI:")
                    with dpg.group():
                        dpg.add_text("Ten sinh vien: ---", tag="current_student_name", color=[255, 255, 100])
                        dpg.add_text("Do tin cay: ---", tag="current_confidence", color=[255, 255, 100])
                        dpg.add_text("Trang thai xac thuc: ---", tag="current_verification", color=[255, 255, 100])
                        dpg.add_text("Thoi gian con lai: ---", tag="time_remaining", color=[255, 255, 100])
                    
                    dpg.add_separator()
                    
                    # Danh sách điểm danh
                    dpg.add_text("LICH SU DIEM DANH:")
                    with dpg.child_window(height=200, border=True, tag="attendance_list"):
                        with dpg.table(
                            header_row=True,
                            borders_innerH=True,
                            borders_outerH=True,
                            borders_innerV=True,
                            borders_outerV=True,
                            tag="attendance_table"
                        ):
                            dpg.add_table_column(label="STT", width_fixed=True, init_width_or_weight=40)
                            dpg.add_table_column(label="Ten SV", width_fixed=True, init_width_or_weight=120)
                            dpg.add_table_column(label="Thoi gian", width_fixed=True, init_width_or_weight=100)
                            dpg.add_table_column(label="Tin cay", width_fixed=True, init_width_or_weight=60)
                            dpg.add_table_column(label="Xac thuc", width_fixed=True, init_width_or_weight=80)
                    
                    dpg.add_separator()
                    
                    # Điều khiển điểm danh
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="XOA DANH SACH", callback=self.clear_attendance)
                        dpg.add_button(label="XUAT BAO CAO", callback=self.export_report)
            
            # Status bar
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text("Ready", tag="status_bar", color=[100, 255, 100])
                dpg.add_spacer(width=200)
                dpg.add_text("Tong so sinh vien da diem danh: 0", tag="total_count")

    def setup_camera_texture(self):
        """Thiết lập texture cho hiển thị camera"""
        # Tạo texture trống ban đầu
        blank_data = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        blank_data.fill(50)  # Màu xám đậm
        
        # Thêm text "No Camera" vào ảnh trống
        cv2.putText(blank_data, "NO CAMERA", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Chuyển đổi từ BGR sang RGBA
        blank_data_rgba = cv2.cvtColor(blank_data, cv2.COLOR_BGR2RGBA)
        blank_data_flat = blank_data_rgba.flatten().astype(np.float32) / 255.0
        
        # Tạo texture
        with dpg.texture_registry():
            dpg.add_raw_texture(
                width=self.camera_width,
                height=self.camera_height,
                default_value=blank_data_flat,
                format=dpg.mvFormat_Float_rgba,
                tag="camera_texture"
            )

    def update_time(self):
        """Cập nhật thời gian hiện tại"""
        current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        dpg.set_value("current_time", f"Thoi gian: {current_time}")

    def toggle_anti_spoofing(self, sender, app_data):
        """Bật/tắt chống giả mạo"""
        self.anti_spoofing_enabled = app_data
        if app_data:
            dpg.set_value("status_bar", "Da bat kiem tra chong gia mao")
        else:
            dpg.set_value("status_bar", "Da tat kiem tra chong gia mao")
            # Reset liveness khi tắt
            self.liveness_verified_time = 0

    def update_liveness_duration(self, sender, app_data):
        """Cập nhật thời gian hiệu lực của liveness"""
        self.liveness_valid_duration = app_data

    def load_model(self):
        """Tải mô hình nhận diện khuôn mặt"""
        try:
            model_path = dpg.get_value("model_path_input")
            class_indices_path = dpg.get_value("class_indices_path_input")
            
            if not os.path.exists(model_path):
                dpg.set_value("model_status", f"Model not found: {model_path}")
                dpg.set_value("status_bar", "Loi: Khong tim thay file model")
                return
            
            if not os.path.exists(class_indices_path):
                dpg.set_value("model_status", f"Class indices not found: {class_indices_path}")
                dpg.set_value("status_bar", "Loi: Khong tim thay file class indices")
                return
            
            # Tải mô hình
            self.face_recognition = FaceRecognition(model_path, class_indices_path)
            dpg.set_value("model_status", "Model: Loaded")
            dpg.set_value("status_bar", "Model da duoc tai thanh cong")
            
        except Exception as e:
            dpg.set_value("model_status", f"Model: Error - {str(e)}")
            dpg.set_value("status_bar", f"Loi khi tai model: {str(e)}")

    def load_class_indices(self):
        """Tải danh sách lớp"""
        try:
            class_indices_path = dpg.get_value("class_indices_path_input")
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
            
            self.student_list = list(class_indices.keys())
            dpg.set_value("status_bar", f"Da tai {len(self.student_list)} sinh vien")
            
        except Exception as e:
            dpg.set_value("status_bar", f"Loi khi tai danh sach lop: {str(e)}")

    def start_camera(self):
        """Bắt đầu camera"""
        if self.is_running:
            return
        
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                dpg.set_value("camera_status", "Camera: Failed")
                dpg.set_value("status_bar", "Loi: Khong the mo camera")
                return
            
            # Thiết lập độ phân giải
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            
            self.is_running = True
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            dpg.set_value("camera_status", "Camera: Connected")
            dpg.set_value("status_bar", "Camera da duoc bat")
            
        except Exception as e:
            dpg.set_value("camera_status", f"Camera: Error - {str(e)}")
            dpg.set_value("status_bar", f"Loi camera: {str(e)}")

    def stop_camera(self):
        """Dừng camera"""
        self.is_running = False
        
        if self.camera_thread:
            self.camera_thread.join(timeout=2.0)
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        # Reset texture về trạng thái ban đầu
        self.setup_camera_texture()
        
        dpg.set_value("camera_status", "Camera: Disconnected")
        dpg.set_value("status_bar", "Camera da duoc tat")

    def reset_verification(self):
        """Reset quá trình xác thực"""
        self.anti_spoofing.reset()
        self.pending_recognitions.clear()
        self.liveness_verified_time = 0
        self.last_recognition_time = 0
        
        # Reset trạng thái các bước
        dpg.set_value("step1_status", "Buoc 1: Phat hien khuon mat")
        dpg.set_value("step2_status", "Buoc 2: Kiem tra tinh song")
        dpg.set_value("step3_status", "Buoc 3: Nhan dien danh tinh")
        dpg.set_value("step4_status", "Buoc 4: Ghi nhan diem danh")
        
        dpg.set_value("current_student_name", "Ten sinh vien: ---")
        dpg.set_value("current_confidence", "Do tin cay: ---")
        dpg.set_value("current_verification", "Trang thai xac thuc: ---")
        dpg.set_value("time_remaining", "Thoi gian con lai: ---")
        
        dpg.set_value("status_bar", "Da reset quy trinh xac thuc")

    def camera_loop(self):
        """Vòng lặp xử lý camera"""
        while self.is_running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # Lật ảnh theo chiều ngang
                frame = cv2.flip(frame, 1)
                self.current_frame = frame.copy()
                
                # Xử lý frame tích hợp
                processed_frame = self.process_integrated_frame(frame)
                
                # Cập nhật texture
                self.update_camera_texture(processed_frame)
                
                # Cập nhật thời gian
                self.update_time()
                
                # Cập nhật trạng thái
                self.update_verification_status()
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Camera loop error: {e}")
                break

    def process_integrated_frame(self, frame):
        """Xử lý frame tích hợp cả nhận diện và chống giả mạo"""
        # Khởi tạo frame kết quả
        result_frame = frame.copy()
        
        # Bước 1: Luôn thực hiện nhận diện khuôn mặt
        recognition_results = []
        if self.face_recognition is not None:
            try:
                result_frame, faces = self.face_recognition.process_frame(result_frame)
                recognition_results = faces
                
                if faces:
                    dpg.set_value("step1_status", "Buoc 1: Phat hien khuon mat ✓")
                    
                    # Hiển thị kết quả nhận diện
                    for (x, y, w, h, class_name, confidence) in faces:
                        if class_name != "Unknown":
                            dpg.set_value("current_student_name", f"Ten sinh vien: {class_name}")
                            dpg.set_value("current_confidence", f"Do tin cay: {confidence:.2f}")
                            dpg.set_value("step3_status", "Buoc 3: Nhan dien danh tinh ✓")
                else:
                    dpg.set_value("step1_status", "Buoc 1: Phat hien khuon mat")
                    dpg.set_value("step3_status", "Buoc 3: Nhan dien danh tinh")
                    
            except Exception as e:
                cv2.putText(result_frame, f"Recognition error: {str(e)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Bước 2: Kiểm tra chống giả mạo (nếu được bật)
        is_live = True  # Mặc định True nếu không bật chống giả mạo
        if self.anti_spoofing_enabled:
            try:
                is_live, result_frame = self.anti_spoofing.check_liveness(result_frame)
                
                if is_live:
                    dpg.set_value("step2_status", "Buoc 2: Kiem tra tinh song ✓")
                    dpg.set_value("liveness_status", "Liveness: VERIFIED")
                    dpg.set_value("current_verification", "Trang thai xac thuc: XAC THUC THANH CONG")
                    self.liveness_verified_time = time.time()
                else:
                    dpg.set_value("step2_status", "Buoc 2: Kiem tra tinh song (dang kiem tra...)")
                    dpg.set_value("liveness_status", "Liveness: VERIFYING")
                    dpg.set_value("current_verification", "Trang thai xac thuc: DANG XAC THUC...")
                    
            except Exception as e:
                cv2.putText(result_frame, f"Anti-spoofing error: {str(e)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            dpg.set_value("step2_status", "Buoc 2: Kiem tra tinh song (da tat)")
            dpg.set_value("liveness_status", "Liveness: DISABLED")
            dpg.set_value("current_verification", "Trang thai xac thuc: KHONG BAT")
        
        # Bước 3: Xử lý điểm danh nếu đã xác thực
        if recognition_results and (is_live or not self.anti_spoofing_enabled):
            self.process_attendance(recognition_results, is_live)
        
        # Hiển thị thông tin tổng hợp trên frame
        self.draw_overlay_info(result_frame, recognition_results, is_live)
        
        return result_frame

    def process_attendance(self, recognition_results, is_live):
        """Xử lý điểm danh khi đã xác thực"""
        current_time = time.time()
        
        # Kiểm tra cooldown
        if current_time - self.last_recognition_time < self.recognition_cooldown:
            return
        
        # Kiểm tra liveness có còn hiệu lực không
        if self.anti_spoofing_enabled:
            if current_time - self.liveness_verified_time > self.liveness_valid_duration:
                return  # Liveness đã hết hiệu lực
        
        # Xử lý từng kết quả nhận diện
        for (x, y, w, h, class_name, confidence) in recognition_results:
            if class_name != "Unknown" and confidence > 0.7:
                # Kiểm tra xem sinh viên đã điểm danh chưa
                already_attended = any(record['name'] == class_name for record in self.attendance_records)
                
                if not already_attended:
                    # Thêm vào danh sách điểm danh
                    verification_status = "XAC THUC" if (is_live or not self.anti_spoofing_enabled) else "KHONG XAC THUC"
                    self.add_attendance_record(class_name, confidence, verification_status)
                    self.last_recognition_time = current_time
                    
                    dpg.set_value("step4_status", "Buoc 4: Ghi nhan diem danh ✓")
                    break

    def draw_overlay_info(self, frame, recognition_results, is_live):
        """Vẽ thông tin overlay lên frame"""
        # Vẽ trạng thái liveness
        liveness_text = "LIVE" if is_live else "VERIFYING"
        liveness_color = (0, 255, 0) if is_live else (0, 255, 255)
        
        if self.anti_spoofing_enabled:
            cv2.putText(frame, f"Liveness: {liveness_text}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, liveness_color, 2)
        else:
            cv2.putText(frame, "Liveness: DISABLED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
        
        # Vẽ số lượng đã điểm danh
        cv2.putText(frame, f"Attended: {len(self.attendance_records)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Vẽ thời gian còn lại của liveness (nếu có)
        if self.anti_spoofing_enabled and self.liveness_verified_time > 0:
            remaining = self.liveness_valid_duration - (time.time() - self.liveness_verified_time)
            if remaining > 0:
                cv2.putText(frame, f"Liveness valid: {remaining:.1f}s", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def update_verification_status(self):
        """Cập nhật trạng thái xác thực"""
        if self.anti_spoofing_enabled and self.liveness_verified_time > 0:
            remaining = self.liveness_valid_duration - (time.time() - self.liveness_verified_time)
            if remaining > 0:
                dpg.set_value("time_remaining", f"Thoi gian con lai: {remaining:.1f}s")
            else:
                dpg.set_value("time_remaining", "Thoi gian con lai: Het han")
                dpg.set_value("current_verification", "Trang thai xac thuc: HET HAN")
        else:
            dpg.set_value("time_remaining", "Thoi gian con lai: ---")

    def update_camera_texture(self, frame):
        """Cập nhật texture camera"""
        try:
            # Resize frame nếu cần
            if frame.shape[:2] != (self.camera_height, self.camera_width):
                frame = cv2.resize(frame, (self.camera_width, self.camera_height))
            
            # Chuyển đổi BGR sang RGBA
            frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame_flat = frame_rgba.flatten().astype(np.float32) / 255.0
            
            # Cập nhật texture
            dpg.set_value("camera_texture", frame_flat)
            
        except Exception as e:
            print(f"Texture update error: {e}")

    def add_attendance_record(self, student_name, confidence, verification_status):
        """Thêm bản ghi điểm danh"""
        # Thêm bản ghi mới
        current_time = datetime.now().strftime("%H:%M:%S")
        record = {
            'name': student_name,
            'time': current_time,
            'confidence': confidence,
            'verification': verification_status
        }
        self.attendance_records.append(record)
        
        # Cập nhật giao diện
        self.update_attendance_display()
        
        # Cập nhật tổng số
        dpg.set_value("total_count", f"Tong so sinh vien da diem danh: {len(self.attendance_records)}")
        dpg.set_value("status_bar", f"Da diem danh: {student_name} - {verification_status}")

    def update_attendance_display(self):
        """Cập nhật hiển thị danh sách điểm danh"""
        # Xóa các hàng cũ
        if dpg.does_item_exist("attendance_table"):
            children = dpg.get_item_children("attendance_table", slot=1)
            for child in children:
                dpg.delete_item(child)
        
        # Thêm các hàng mới
        for i, record in enumerate(self.attendance_records):
            with dpg.table_row(parent="attendance_table"):
                dpg.add_text(str(i + 1))
                dpg.add_text(record['name'])
                dpg.add_text(record['time'])
                dpg.add_text(f"{record['confidence']:.2f}")
                
                # Màu sắc cho trạng thái xác thực
                verification_color = [0, 255, 0] if record['verification'] == "XAC THUC" else [255, 100, 100]
                dpg.add_text(record['verification'], color=verification_color)

    def capture_photo(self):
        """Chụp ảnh từ camera"""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            dpg.set_value("status_bar", f"Da chup anh: {filename}")

    def clear_attendance(self):
       """Xóa danh sách điểm danh"""
       self.attendance_records.clear()
       self.update_attendance_display()
       dpg.set_value("total_count", "Tong so sinh vien da diem danh: 0")
       dpg.set_value("status_bar", "Da xoa danh sach diem danh")
       
       # Reset các trạng thái hiện tại
       dpg.set_value("current_student_name", "Ten sinh vien: ---")
       dpg.set_value("current_confidence", "Do tin cay: ---")
       dpg.set_value("current_verification", "Trang thai xac thuc: ---")

    def export_report(self):
        """Xuất báo cáo điểm danh"""
        if not self.attendance_records:
            dpg.set_value("status_bar", "Khong co du lieu de xuat bao cao")
            return
       
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"attendance_report_{timestamp}.txt"
            
            # Tính thống kê
            total_students = len(self.attendance_records)
            verified_students = len([r for r in self.attendance_records if r['verification'] == "XAC THUC"])
            unverified_students = total_students - verified_students
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("BAO CAO DIEM DANH TICH HOP CHONG GIAN LAN\n")
                f.write("=" * 60 + "\n")
                f.write(f"Ngay tao: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
                f.write(f"He thong: Nhan dien khuon mat + Chong gia mao\n\n")
                
                f.write("THONG KE TONG HOP:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Tong so sinh vien diem danh: {total_students}\n")
                f.write(f"Sinh vien da xac thuc: {verified_students}\n")
                f.write(f"Sinh vien chua xac thuc: {unverified_students}\n")
                f.write(f"Ty le xac thuc: {(verified_students/total_students*100):.1f}%\n\n")
                
                f.write("CHI TIET DIEM DANH:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'STT':<5} {'Ten sinh vien':<20} {'Thoi gian':<12} {'Tin cay':<10} {'Xac thuc':<15}\n")
                f.write("-" * 80 + "\n")
                
                for i, record in enumerate(self.attendance_records):
                    f.write(f"{i+1:<5} {record['name']:<20} {record['time']:<12} "
                            f"{record['confidence']:<10.2f} {record['verification']:<15}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("LUU Y:\n")
                f.write("- XAC THUC: Sinh vien da vuot qua kiem tra chong gia mao\n")
                f.write("- KHONG XAC THUC: Sinh vien chua vuot qua kiem tra chong gia mao\n")
                f.write("- Do tin cay: Muc do tin cay cua mo hinh nhan dien (0.0 - 1.0)\n")
            
            dpg.set_value("status_bar", f"Da xuat bao cao: {filename}")
            
        except Exception as e:
            dpg.set_value("status_bar", f"Loi khi xuat bao cao: {str(e)}")

    def run(self):
        """Chạy ứng dụng"""
        dpg.setup_dearpygui()
        dpg.show_viewport()
        
        try:
            while dpg.is_dearpygui_running():
                dpg.render_dearpygui_frame()
        finally:
            self.cleanup()

    def cleanup(self):
        """Dọn dẹp tài nguyên"""
        self.stop_camera()
        dpg.destroy_context()

def main():
   """Hàm main"""
   app = AttendanceGUI()
   app.run()

if __name__ == "__main__":
   main()