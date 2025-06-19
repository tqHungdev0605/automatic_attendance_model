#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Attendance System - Main Interface
Hệ thống điểm danh bằng nhận dạng khuôn mặt
"""

import os
import sys
from datetime import datetime
from attendance import AttendanceSystem

def print_header():
    """In header của chương trình"""
    print("\n" + "="*60)
    print("          HỆ THỐNG ĐIỂM DANH KHUÔN MẶT")
    print("          Face Attendance System")
    print("="*60)

def print_menu():
    """In menu chính"""
    print("\n" + "="*50)
    print("                 MENU CHÍNH")
    print("="*50)
    print("1. Thêm sinh viên từ ảnh")
    print("2. Thêm sinh viên từ camera")
    print("3. Xóa sinh viên")
    print("4. Xem danh sách sinh viên")
    print("5. Bắt đầu điểm danh")
    print("6. Xem danh sách điểm danh")
    print("7. Xuất dữ liệu điểm danh (CSV)")
    print("8. Thống kê tổng quan")
    print("9. Cài đặt")
    print("0. Thoát")
    print("="*50)

def print_settings_menu():
    """In menu cài đặt"""
    print("\n" + "="*40)
    print("           CÀI ĐẶT")
    print("="*40)
    print("1. Thay đổi ngưỡng nhận dạng")
    print("2. Xem thông tin hệ thống")
    print("3. Sao lưu dữ liệu")
    print("0. Quay lại menu chính")
    print("="*40)

def get_model_path():
    """Tìm đường dẫn model"""
    possible_paths = [
        "models/face_recognition_model_compatible (1).tflite",
        "face_recognition_model_compatibl.tflite",
        "model.tflite"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    print("⚠️  Không tìm thấy file model!")
    print("Vui lòng đặt file model vào một trong các vị trí sau:")
    for path in possible_paths:
        print(f"  - {path}")
    
    custom_path = input("Hoặc nhập đường dẫn model: ").strip()
    if custom_path and os.path.exists(custom_path):
        return custom_path
    
    return None

def add_student_from_image(system):
    """Thêm sinh viên từ ảnh"""
    print("\n--- THÊM SINH VIÊN TỪ ẢNH ---")
    
    student_id = input("Nhập mã sinh viên: ").strip()
    if not student_id:
        print("❌ Mã sinh viên không được để trống!")
        return
    
    name = input("Nhập tên sinh viên: ").strip()
    if not name:
        print("❌ Tên sinh viên không được để trống!")
        return
    
    image_path = input("Nhập đường dẫn ảnh: ").strip()
    if not image_path or not os.path.exists(image_path):
        print("❌ Đường dẫn ảnh không hợp lệ!")
        return
    
    print("Đang xử lý...")
    success, message = system.add_student_from_image(image_path, student_id, name)
    
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")

def add_student_from_camera(system):
    """Thêm sinh viên từ camera"""
    print("\n--- THÊM SINH VIÊN TỪ CAMERA ---")
    
    student_id = input("Nhập mã sinh viên: ").strip()
    if not student_id:
        print("❌ Mã sinh viên không được để trống!")
        return
    
    name = input("Nhập tên sinh viên: ").strip()
    if not name:
        print("❌ Tên sinh viên không được để trống!")
        return
    
    print("Chuẩn bị camera...")
    success, message = system.add_student_from_camera(student_id, name)
    
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")

def delete_student(system):
    """Xóa sinh viên"""
    print("\n--- XÓA SINH VIÊN ---")
    
    # Hiển thị danh sách sinh viên trước
    system.print_student_list()
    
    student_id = input("Nhập mã sinh viên cần xóa: ").strip()
    if not student_id:
        print("❌ Mã sinh viên không được để trống!")
        return
    
    confirm = input(f"Bạn có chắc chắn muốn xóa sinh viên {student_id}? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Đã hủy thao tác xóa.")
        return
    
    success, message = system.delete_student(student_id)
    
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")

def view_attendance(system):
    """Xem danh sách điểm danh"""
    print("\n--- XEM DANH SÁCH ĐIỂM DANH ---")
    
    date_input = input("Nhập ngày (YYYY-MM-DD) hoặc Enter để xem hôm nay: ").strip()
    date = date_input if date_input else None
    
    if date:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            print("❌ Định dạng ngày không hợp lệ! Sử dụng định dạng YYYY-MM-DD")
            return
    
    system.print_attendance_list(date)

def export_attendance(system):
    """Xuất dữ liệu điểm danh"""
    print("\n--- XUẤT DỮ LIỆU ĐIỂM DANH ---")
    
    date_input = input("Nhập ngày (YYYY-MM-DD) hoặc Enter để xuất hôm nay: ").strip()
    date = date_input if date_input else None
    
    if date:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            print("❌ Định dạng ngày không hợp lệ! Sử dụng định dạng YYYY-MM-DD")
            return
    
    # Tạo tên file mặc định
    export_date = date if date else datetime.now().strftime("%Y-%m-%d")
    default_filename = f"attendance_{export_date}.csv"
    
    filename = input(f"Nhập tên file (mặc định: {default_filename}): ").strip()
    if not filename:
        filename = default_filename
    
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    # Tạo thư mục exports nếu chưa có
    os.makedirs("exports", exist_ok=True)
    output_path = os.path.join("exports", filename)
    
    success, message = system.export_attendance_to_csv(output_path, date)
    
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")

def show_statistics(system):
    """Hiển thị thống kê"""
    print("\n--- THỐNG KÊ TỔNG QUAN ---")
    
    stats = system.get_statistics()
    
    print("="*50)
    print(f"Tổng số sinh viên:           {stats['total_students']}")
    print(f"Tổng số lần điểm danh:       {stats['total_attendance_records']}")
    print(f"Điểm danh hôm nay:           {stats['today_attendance']}")
    print("="*50)

def settings_menu(system):
    """Menu cài đặt"""
    while True:
        print_settings_menu()
        choice = input("Chọn chức năng: ").strip()
        
        if choice == '1':
            print(f"\nNgưỡng hiện tại: {system.threshold}")
            try:
                new_threshold = float(input("Nhập ngưỡng mới (0.0-1.0): "))
                if 0.0 <= new_threshold <= 1.0:
                    system.threshold = new_threshold
                    print(f"✅ Đã cập nhật ngưỡng thành {new_threshold}")
                else:
                    print("❌ Ngưỡng phải trong khoảng 0.0-1.0!")
            except ValueError:
                print("❌ Vui lòng nhập số hợp lệ!")
        
        elif choice == '2':
            print("\n--- THÔNG TIN HỆ THỐNG ---")
            print(f"Database: {system.db.db_path}")
            print(f"Ngưỡng nhận dạng: {system.threshold}")
            print(f"Thư mục ảnh sinh viên: student_faces/")
            print(f"Thư mục xuất dữ liệu: exports/")
        
        elif choice == '3':
            print("\n--- SAO LƯU DỮ LIỆU ---")
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            try:
                import shutil
                shutil.copy2(system.db.db_path, backup_name)
                print(f"✅ Đã sao lưu database thành {backup_name}")
            except Exception as e:
                print(f"❌ Lỗi sao lưu: {str(e)}")
        
        elif choice == '0':
            break
        
        else:
            print("❌ Lựa chọn không hợp lệ!")

def main():
    """Hàm main"""
    print_header()
    
    # Kiểm tra model
    model_path = get_model_path()
    if not model_path:
        print("❌ Không thể khởi tạo hệ thống без model!")
        sys.exit(1)
    
    print(f"📁 Sử dụng model: {model_path}")
    
    # Khởi tạo hệ thống
    try:
        system = AttendanceSystem(model_path)
        print("✅ Khởi tạo hệ thống thành công!")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo hệ thống: {str(e)}")
        sys.exit(1)
    
    # Main loop
    while True:
        try:
            print_menu()
            choice = input("Chọn chức năng: ").strip()
            
            if choice == '1':
                add_student_from_image(system)
            
            elif choice == '2':
                add_student_from_camera(system)
            
            elif choice == '3':
                delete_student(system)
            
            elif choice == '4':
                system.print_student_list()
            
            elif choice == '5':
                print("\n--- BẮT ĐẦU ĐIỂM DANH ---")
                print("Chuẩn bị camera...")
                success, message, attended = system.run_attendance()
                
                if success:
                    print(f"\n✅ {message}")
                    if attended:
                        print(f"Đã điểm danh {len(attended)} sinh viên:")
                        for student in attended:
                            print(f"  - {student['name']} ({student['id']}) - {student['time']}")
                else:
                    print(f"❌ {message}")
            
            elif choice == '6':
                view_attendance(system)
            
            elif choice == '7':
                export_attendance(system)
            
            elif choice == '8':
                show_statistics(system)
            
            elif choice == '9':
                settings_menu(system)
            
            elif choice == '0':
                print("\n👋 Cảm ơn bạn đã sử dụng hệ thống!")
                break
            
            else:
                print("❌ Lựa chọn không hợp lệ! Vui lòng chọn từ 0-9.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"\n❌ Lỗi không mong muốn: {str(e)}")
            print("Hệ thống sẽ tiếp tục hoạt động...")

if __name__ == "__main__":
    main()