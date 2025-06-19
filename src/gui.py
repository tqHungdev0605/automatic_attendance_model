#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI cho he thong diem danh bang nhan dang khuon mat
Su dung DearPyGui
"""

import dearpygui.dearpygui as dpg
import os
import sys
from datetime import datetime
import threading
import time
from attendance import AttendanceSystem

class AttendanceGUI:
    def __init__(self):
        self.system = None
        self.is_attending = False
        self.attendance_thread = None
        
        # Tao context DearPyGui
        dpg.create_context()
        
        # Thiet lap fonts (tuy chon)
        self.setup_fonts()
        
        # Tao cua so chinh
        self.create_main_window()
        
        # Thiet lap viewport
        dpg.create_viewport(title="he thong diem danh khuon mat", width=1000, height=700)
        dpg.setup_dearpygui()
        
    def setup_fonts(self):
        """Thiet lap fonts (tuy chon)"""
    
    def create_main_window(self):
        """Tao cua so chinh"""
        with dpg.window(label="he thong diem danh", tag="main_window"):
            # Header
            dpg.add_text("he thong diem danh khuon mat", color=[255, 255, 0])
            dpg.add_separator()
            
            # Khoi tao he thong
            with dpg.group(horizontal=True):
                dpg.add_button(label="khoi tao he thong", callback=self.init_system)
                dpg.add_text("trang thai: chua khoi tao", tag="system_status", color=[255, 100, 100])
            
            dpg.add_separator()
            
            # Tab bar
            with dpg.tab_bar():
                # Tab quan ly sinh vien
                with dpg.tab(label="quan ly sinh vien"):
                    self.create_student_management_tab()
                
                # Tab diem danh
                with dpg.tab(label="diem danh"):
                    self.create_attendance_tab()
                
                # Tab xem du lieu
                with dpg.tab(label="xem du lieu"):
                    self.create_view_data_tab()
                
                # Tab cai dat
                with dpg.tab(label="cai dat"):
                    self.create_settings_tab()
    
    def create_student_management_tab(self):
        """Tab quan ly sinh vien"""
        dpg.add_text("quan ly sinh vien", color=[0, 255, 255])
        dpg.add_separator()
        
        # Them sinh vien tu anh
        with dpg.collapsing_header(label="them sinh vien tu anh"):
            dpg.add_input_text(label="ma sinh vien", tag="student_id_image", width=200)
            dpg.add_input_text(label="ten sinh vien", tag="student_name_image", width=300)
            dpg.add_input_text(label="duong dan anh", tag="image_path", width=400)
            with dpg.group(horizontal=True):
                dpg.add_button(label="chon file anh", callback=self.select_image_file)
                dpg.add_button(label="them sinh vien", callback=self.add_student_from_image)
        
        # Them sinh vien tu camera
        with dpg.collapsing_header(label="them sinh vien tu camera"):
            dpg.add_input_text(label="ma sinh vien", tag="student_id_camera", width=200)
            dpg.add_input_text(label="ten sinh vien", tag="student_name_camera", width=300)
            dpg.add_button(label="them tu camera", callback=self.add_student_from_camera)
        
        # Xoa sinh vien
        with dpg.collapsing_header(label="xoa sinh vien"):
            dpg.add_input_text(label="ma sinh vien can xoa", tag="delete_student_id", width=200)
            dpg.add_button(label="xoa sinh vien", callback=self.delete_student)
        
        dpg.add_separator()
        
        # Danh sach sinh vien
        dpg.add_text("danh sach sinh vien")
        with dpg.group(horizontal=True):
            dpg.add_button(label="lam moi danh sach", callback=self.refresh_student_list)
        
        # Bang danh sach sinh vien
        with dpg.table(header_row=True, tag="student_table", 
                      resizable=True, policy=dpg.mvTable_SizingStretchProp):
            dpg.add_table_column(label="stt", width_fixed=True, init_width_or_weight=50)
            dpg.add_table_column(label="ma sv", width_fixed=True, init_width_or_weight=120)
            dpg.add_table_column(label="ten sinh vien", width_stretch=True)
            dpg.add_table_column(label="ngay tao", width_fixed=True, init_width_or_weight=150)
    
    def create_attendance_tab(self):
        """Tab diem danh"""
        dpg.add_text("diem danh", color=[0, 255, 255])
        dpg.add_separator()
        
        # Trang thai diem danh
        with dpg.group(horizontal=True):
            dpg.add_button(label="bat dau diem danh", tag="start_attendance_btn", 
                          callback=self.start_attendance)
            dpg.add_button(label="dung diem danh", tag="stop_attendance_btn", 
                          callback=self.stop_attendance, enabled=False)
            dpg.add_text("trang thai: chua bat dau", tag="attendance_status", color=[255, 255, 255])
        
        dpg.add_separator()
        
        # Thong tin diem danh trong phien hien tai
        dpg.add_text("thong tin phien diem danh")
        dpg.add_text("so sinh vien da diem danh: 0", tag="current_attendance_count")
        
        # Danh sach sinh vien vua diem danh
        dpg.add_text("danh sach sinh vien vua diem danh:")
        with dpg.child_window(height=200, tag="current_attendance_list"):
            dpg.add_text("chua co sinh vien nao diem danh")
    
    def create_view_data_tab(self):
        """Tab xem du lieu"""
        dpg.add_text("xem du lieu diem danh", color=[0, 255, 255])
        dpg.add_separator()
        
        # Chon ngay xem
        with dpg.group(horizontal=True):
            dpg.add_input_text(label="ngay (yyyy-mm-dd)", tag="view_date", 
                              default_value=datetime.now().strftime("%Y-%m-%d"), width=150)
            dpg.add_button(label="xem diem danh", callback=self.view_attendance)
            dpg.add_button(label="xuat csv", callback=self.export_attendance)
        
        # Bang du lieu diem danh
        with dpg.table(header_row=True, tag="attendance_table", 
                      resizable=True, policy=dpg.mvTable_SizingStretchProp):
            dpg.add_table_column(label="stt", width_fixed=True, init_width_or_weight=50)
            dpg.add_table_column(label="ma sv", width_fixed=True, init_width_or_weight=100)
            dpg.add_table_column(label="ten", width_stretch=True)
            dpg.add_table_column(label="gio", width_fixed=True, init_width_or_weight=80)
            dpg.add_table_column(label="do chinh xac", width_fixed=True, init_width_or_weight=100)
        
        # Thong ke
        dpg.add_separator()
        dpg.add_text("thong ke tong quan")
        dpg.add_button(label="lam moi thong ke", callback=self.refresh_statistics)
        dpg.add_text("tong so sinh vien: 0", tag="total_students")
        dpg.add_text("tong so lan diem danh: 0", tag="total_attendance")
        dpg.add_text("diem danh hom nay: 0", tag="today_attendance")
    
    def create_settings_tab(self):
        """Tab cai dat"""
        dpg.add_text("cai dat he thong", color=[0, 255, 255])
        dpg.add_separator()
        
        # Cai dat nguong
        dpg.add_slider_float(label="nguong nhan dang", tag="threshold_slider", 
                           default_value=0.6, min_value=0.1, max_value=1.0, 
                           format="%.2f", callback=self.update_threshold)
        
        dpg.add_separator()
        
        # Thong tin he thong
        dpg.add_text("thong tin he thong")
        dpg.add_text("database: face_attendance.db", tag="db_info")
        dpg.add_text("model: chua tai", tag="model_info")
        dpg.add_text("thu muc anh: student_faces/", tag="image_dir_info")
        
        dpg.add_separator()
        
        # Sao luu
        dpg.add_button(label="sao luu co so du lieu", callback=self.backup_database)
    
    def init_system(self):
        """Khoi tao he thong"""
        try:
            # Tim model
            model_path = self.get_model_path()
            if not model_path:
                dpg.set_value("system_status", "loi: khong tim thay model!")
                dpg.configure_item("system_status", color=[255, 100, 100])
                return
            
            # Khoi tao AttendanceSystem
            self.system = AttendanceSystem(model_path)
            
            dpg.set_value("system_status", "da khoi tao thanh cong")
            dpg.configure_item("system_status", color=[100, 255, 100])
            dpg.set_value("model_info", f"model: {model_path}")
            
            # Lam moi danh sach sinh vien
            self.refresh_student_list()
            self.refresh_statistics()
            
        except Exception as e:
            dpg.set_value("system_status", f"loi: {str(e)}")
            dpg.configure_item("system_status", color=[255, 100, 100])
    
    def get_model_path(self):
        """Tim duong dan model"""
        possible_paths = [
            "models/face_recognition_model_compatible.tflite",
            "face_recognition_model_compatible.tflite",
            "model.tflite"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def select_image_file(self):
        """Chon file anh"""
        def file_callback(sender, app_data):
            dpg.set_value("image_path", app_data["file_path_name"])
        
        with dpg.file_dialog(directory_selector=False, show=True, 
                           callback=file_callback, width=700, height=400):
            dpg.add_file_extension(".*")
            dpg.add_file_extension(".jpg", color=[255, 255, 0, 255])
            dpg.add_file_extension(".png", color=[255, 255, 0, 255])
            dpg.add_file_extension(".jpeg", color=[255, 255, 0, 255])
    
    def add_student_from_image(self):
        """Them sinh vien tu anh"""
        if not self.system:
            self.show_message("loi", "hay khoi tao he thong truoc!")
            return
        
        student_id = dpg.get_value("student_id_image").strip()
        name = dpg.get_value("student_name_image").strip()
        image_path = dpg.get_value("image_path").strip()
        
        if not student_id or not name or not image_path:
            self.show_message("loi", "vui long dien day du thong tin!")
            return
        
        try:
            success, message = self.system.add_student_from_image(image_path, student_id, name)
            if success:
                self.show_message("thanh cong", message)
                # Xoa form
                dpg.set_value("student_id_image", "")
                dpg.set_value("student_name_image", "")
                dpg.set_value("image_path", "")
                # Lam moi danh sach
                self.refresh_student_list()
            else:
                self.show_message("loi", message)
        except Exception as e:
            self.show_message("loi", f"loi khong mong muon: {str(e)}")
    
    def add_student_from_camera(self):
        """Them sinh vien tu camera"""
        if not self.system:
            self.show_message("loi", "hay khoi tao he thong truoc!")
            return
        
        student_id = dpg.get_value("student_id_camera").strip()
        name = dpg.get_value("student_name_camera").strip()
        
        if not student_id or not name:
            self.show_message("loi", "vui long dien day du thong tin!")
            return
        
        def camera_thread():
            try:
                success, message = self.system.add_student_from_camera(student_id, name)
                if success:
                    self.show_message("thanh cong", message)
                    # Xoa form
                    dpg.set_value("student_id_camera", "")
                    dpg.set_value("student_name_camera", "")
                    # Lam moi danh sach
                    self.refresh_student_list()
                else:
                    self.show_message("loi", message)
            except Exception as e:
                self.show_message("loi", f"loi khong mong muon: {str(e)}")
        
        threading.Thread(target=camera_thread, daemon=True).start()
    
    def delete_student(self):
        """Xoa sinh vien"""
        if not self.system:
            self.show_message("loi", "hay khoi tao he thong truoc!")
            return
        
        student_id = dpg.get_value("delete_student_id").strip()
        if not student_id:
            self.show_message("loi", "vui long nhap ma sinh vien!")
            return
        
        # Hien thi dialog xac nhan
        self.show_confirm_dialog(f"ban co chac chan muon xoa sinh vien {student_id}?", 
                               lambda: self.do_delete_student(student_id))
    
    def do_delete_student(self, student_id):
        """Thuc hien xoa sinh vien"""
        try:
            success, message = self.system.delete_student(student_id)
            if success:
                self.show_message("thanh cong", message)
                dpg.set_value("delete_student_id", "")
                self.refresh_student_list()
            else:
                self.show_message("loi", message)
        except Exception as e:
            self.show_message("loi", f"loi khong mong muon: {str(e)}")
    
    def refresh_student_list(self):
        """Lam moi danh sach sinh vien"""
        if not self.system:
            return
        
        try:
            students = self.system.db.get_all_students()
            
            # Xoa cac hang cu
            if dpg.does_item_exist("student_table"):
                children = dpg.get_item_children("student_table", slot=1)
                for child in children:
                    dpg.delete_item(child)
            
            # Them du lieu moi
            for i, student in enumerate(students, 1):
                with dpg.table_row(parent="student_table"):
                    dpg.add_text(str(i))
                    dpg.add_text(student['id'])
                    dpg.add_text(student['name'])
                    dpg.add_text("n/a")  # Co the them created_at sau
                    
        except Exception as e:
            print(f"loi khi lam moi danh sach sinh vien: {e}")
    
    def start_attendance(self):
        """Bat dau diem danh"""
        if not self.system:
            self.show_message("loi", "hay khoi tao he thong truoc!")
            return
        
        if self.is_attending:
            return
        
        self.is_attending = True
        dpg.configure_item("start_attendance_btn", enabled=False)
        dpg.configure_item("stop_attendance_btn", enabled=True)
        dpg.set_value("attendance_status", "trang thai: dang diem danh")
        dpg.configure_item("attendance_status", color=[100, 255, 100])
        
        def attendance_thread():
            try:
                success, message, attended = self.system.run_attendance()
                
                # Cap nhat giao dien
                self.is_attending = False
                dpg.configure_item("start_attendance_btn", enabled=True)
                dpg.configure_item("stop_attendance_btn", enabled=False)
                dpg.set_value("attendance_status", "trang thai: da dung")
                dpg.configure_item("attendance_status", color=[255, 255, 255])
                
                if success and attended:
                    # Cap nhat danh sach sinh vien vua diem danh
                    self.update_current_attendance_list(attended)
                    dpg.set_value("current_attendance_count", 
                                f"so sinh vien da diem danh: {len(attended)}")
                    
                    self.show_message("hoan thanh", f"{message}\nda diem danh {len(attended)} sinh vien")
                else:
                    self.show_message("thong bao", message)
                    
            except Exception as e:
                self.is_attending = False
                dpg.configure_item("start_attendance_btn", enabled=True)
                dpg.configure_item("stop_attendance_btn", enabled=False)
                dpg.set_value("attendance_status", "trang thai: loi")
                dpg.configure_item("attendance_status", color=[255, 100, 100])
                self.show_message("loi", f"loi trong qua trinh diem danh: {str(e)}")
        
        self.attendance_thread = threading.Thread(target=attendance_thread, daemon=True)
        self.attendance_thread.start()
    
    def stop_attendance(self):
        """Dung diem danh"""
        self.is_attending = False
        dpg.configure_item("start_attendance_btn", enabled=True)
        dpg.configure_item("stop_attendance_btn", enabled=False)
        dpg.set_value("attendance_status", "trang thai: da dung")
        dpg.configure_item("attendance_status", color=[255, 255, 255])
    
    def update_current_attendance_list(self, attended):
        """Cap nhat danh sach sinh vien vua diem danh"""
        # Xoa noi dung cu
        children = dpg.get_item_children("current_attendance_list", slot=1)
        for child in children:
            dpg.delete_item(child)
        
        # Them danh sach moi
        if attended:
            for i, student in enumerate(attended, 1):  # Bắt đầu đếm từ 1
                dpg.add_text(f"{i}. {student['name']} ({student['id']}) - {student['time']}", 
                        parent="current_attendance_list")
        else:
            dpg.add_text("chua co sinh vien nao diem danh", parent="current_attendance_list")
    
    def view_attendance(self):
        """Xem danh sach diem danh"""
        if not self.system:
            self.show_message("loi", "hay khoi tao he thong truoc!")
            return
        
        date = dpg.get_value("view_date")
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            records, message = self.system.view_attendance(date)
            
            # Xoa du lieu cu
            children = dpg.get_item_children("attendance_table", slot=1)
            for child in children:
                dpg.delete_item(child)
            
            # Them du lieu moi
            if records:
                for i, record in enumerate(records, 1):
                    with dpg.table_row(parent="attendance_table"):
                        dpg.add_text(str(i))
                        dpg.add_text(record['student_id'])
                        dpg.add_text(record['name'])
                        dpg.add_text(record['time'])
                        similarity = f"{record['similarity']:.2f}" if record['similarity'] else "n/a"
                        dpg.add_text(similarity)
            
            self.show_message("thong tin", message)
            
        except Exception as e:
            self.show_message("loi", f"loi khi xem du lieu: {str(e)}")
    
    def export_attendance(self):
        """Xuat du lieu diem danh"""
        if not self.system:
            self.show_message("loi", "hay khoi tao he thong truoc!")
            return
        
        date = dpg.get_value("view_date")
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Tao ten file
        filename = f"attendance_{date}.csv"
        os.makedirs("exports", exist_ok=True)
        output_path = os.path.join("exports", filename)
        
        try:
            success, message = self.system.export_attendance_to_csv(output_path, date)
            self.show_message("thong bao", message)
        except Exception as e:
            self.show_message("loi", f"loi khi xuat du lieu: {str(e)}")
    
    def refresh_statistics(self):
        """Lam moi thong ke"""
        if not self.system:
            return
        
        try:
            stats = self.system.get_statistics()
            dpg.set_value("total_students", f"tong so sinh vien: {stats['total_students']}")
            dpg.set_value("total_attendance", f"tong so lan diem danh: {stats['total_attendance_records']}")
            dpg.set_value("today_attendance", f"diem danh hom nay: {stats['today_attendance']}")
        except Exception as e:
            print(f"loi khi lam moi thong ke: {e}")
    
    def update_threshold(self):
        """Cap nhat nguong nhan dang"""
        if self.system:
            threshold = dpg.get_value("threshold_slider")
            self.system.threshold = threshold
    
    def backup_database(self):
        """Sao luu co so du lieu"""
        if not self.system:
            self.show_message("loi", "hay khoi tao he thong truoc!")
            return
        
        try:
            import shutil
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            shutil.copy2(self.system.db.db_path, backup_name)
            self.show_message("thanh cong", f"da sao luu database thanh {backup_name}")
        except Exception as e:
            self.show_message("loi", f"loi sao luu: {str(e)}")
    
    def show_message(self, title, message):
        """Hien thi thong bao"""
        with dpg.window(label=title, modal=True, show=True, tag="message_window", 
                       width=400, height=150, pos=[300, 300]):
            dpg.add_text(message)
            dpg.add_separator()
            dpg.add_button(label="ok", callback=lambda: dpg.delete_item("message_window"))
    
    def show_confirm_dialog(self, message, callback):
        """Hien thi dialog xac nhan"""
        with dpg.window(label="xac nhan", modal=True, show=True, tag="confirm_window", 
                       width=400, height=150, pos=[300, 300]):
            dpg.add_text(message)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                def yes_callback():
                    dpg.delete_item("confirm_window")
                    callback()
                def no_callback():
                    dpg.delete_item("confirm_window")
                
                dpg.add_button(label="co", callback=yes_callback)
                dpg.add_button(label="khong", callback=no_callback)
    
    def run(self):
        """Chay ung dung"""
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
        
        # Main loop
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
        
        dpg.destroy_context()

def main():
    """Ham main de chay GUI"""
    app = AttendanceGUI()
    app.run()

if __name__ == "__main__":
    main()