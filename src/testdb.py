# query_database.py - Tool để truy vấn và xem cơ sở dữ liệu điểm danh
import sqlite3
import json
from datetime import datetime
import pandas as pd

class DatabaseViewer:
    def __init__(self, db_path="face_attendance.db"):
        self.db_path = db_path
    
    def view_all_students(self):
        """Xem tất cả sinh viên"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name, photo_path, created_at FROM students")
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            print("Không có sinh viên nào trong cơ sở dữ liệu.")
            return
        
        print("\n=== DANH SÁCH SINH VIÊN ===")
        print(f"{'ID':<10} {'Tên':<25} {'Ảnh':<20} {'Ngày tạo':<20}")
        print("-" * 75)
        
        for row in results:
            id, name, photo_path, created_at = row
            photo = photo_path if photo_path else "Không có"
            print(f"{id:<10} {name:<25} {photo:<20} {created_at:<20}")
    
    def view_attendance_by_date(self, date=None):
        """Xem điểm danh theo ngày"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT a.student_id, s.name, a.time, a.status, a.similarity
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE a.date = ?
            ORDER BY a.time
        """, (date,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            print(f"Không có dữ liệu điểm danh cho ngày {date}")
            return
        
        print(f"\n=== ĐIỂM DANH NGÀY {date} ===")
        print(f"{'ID SV':<10} {'Tên':<25} {'Giờ':<10} {'Trạng thái':<12} {'Độ tương đồng':<15}")
        print("-" * 72)
        
        for row in results:
            student_id, name, time, status, similarity = row
            sim_str = f"{similarity:.3f}" if similarity else "N/A"
            print(f"{student_id:<10} {name:<25} {time:<10} {status:<12} {sim_str:<15}")
    
    def view_all_attendance(self):
        """Xem tất cả dữ liệu điểm danh"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT a.student_id, s.name, a.date, a.time, a.status, a.similarity
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            ORDER BY a.date DESC, a.time DESC
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            print("Không có dữ liệu điểm danh.")
            return
        
        print("\n=== TẤT CẢ DỮ LIỆU ĐIỂM DANH ===")
        print(f"{'ID SV':<10} {'Tên':<25} {'Ngày':<12} {'Giờ':<10} {'Trạng thái':<12} {'Độ tương đồng':<15}")
        print("-" * 84)
        
        for row in results:
            student_id, name, date, time, status, similarity = row
            sim_str = f"{similarity:.3f}" if similarity else "N/A"
            print(f"{student_id:<10} {name:<25} {date:<12} {time:<10} {status:<12} {sim_str:<15}")
    
    def search_student(self, keyword):
        """Tìm kiếm sinh viên theo tên hoặc ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, photo_path, created_at 
            FROM students 
            WHERE id LIKE ? OR name LIKE ?
        """, (f"%{keyword}%", f"%{keyword}%"))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            print(f"Không tìm thấy sinh viên nào với từ khóa: {keyword}")
            return
        
        print(f"\n=== KẾT QUẢ TÌM KIẾM: '{keyword}' ===")
        print(f"{'ID':<10} {'Tên':<25} {'Ảnh':<20} {'Ngày tạo':<20}")
        print("-" * 75)
        
        for row in results:
            id, name, photo_path, created_at = row
            photo = photo_path if photo_path else "Không có"
            print(f"{id:<10} {name:<25} {photo:<20} {created_at:<20}")
    
    def get_student_attendance_history(self, student_id):
        """Xem lịch sử điểm danh của một sinh viên"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Lấy thông tin sinh viên
        cursor.execute("SELECT name FROM students WHERE id = ?", (student_id,))
        student = cursor.fetchone()
        
        if not student:
            print(f"Không tìm thấy sinh viên với ID: {student_id}")
            conn.close()
            return
        
        student_name = student[0]
        
        # Lấy lịch sử điểm danh
        cursor.execute("""
            SELECT date, time, status, similarity
            FROM attendance
            WHERE student_id = ?
            ORDER BY date DESC, time DESC
        """, (student_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            print(f"Sinh viên {student_name} ({student_id}) chưa có lịch sử điểm danh.")
            return
        
        print(f"\n=== LỊCH SỬ ĐIỂM DANH - {student_name} ({student_id}) ===")
        print(f"{'Ngày':<12} {'Giờ':<10} {'Trạng thái':<12} {'Độ tương đồng':<15}")
        print("-" * 49)
        
        for row in results:
            date, time, status, similarity = row
            sim_str = f"{similarity:.3f}" if similarity else "N/A"
            print(f"{date:<12} {time:<10} {status:<12} {sim_str:<15}")
    
    def get_statistics(self):
        """Xem thống kê tổng quan"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Đếm số sinh viên
        cursor.execute("SELECT COUNT(*) FROM students")
        total_students = cursor.fetchone()[0]
        
        # Đếm số lượt điểm danh
        cursor.execute("SELECT COUNT(*) FROM attendance")
        total_attendance = cursor.fetchone()[0]
        
        # Điểm danh hôm nay
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute("SELECT COUNT(*) FROM attendance WHERE date = ?", (today,))
        today_attendance = cursor.fetchone()[0]
        
        # Ngày có điểm danh gần nhất
        cursor.execute("SELECT date FROM attendance ORDER BY date DESC LIMIT 1")
        latest_date = cursor.fetchone()
        latest_date = latest_date[0] if latest_date else "Chưa có"
        
        conn.close()
        
        print("\n=== THỐNG KÊ TỔNG QUAN ===")
        print(f"Tổng số sinh viên: {total_students}")
        print(f"Tổng số lượt điểm danh: {total_attendance}")
        print(f"Điểm danh hôm nay ({today}): {today_attendance}")
        print(f"Ngày điểm danh gần nhất: {latest_date}")
    
    def export_to_csv(self, table_name="all_attendance", filename=None):
        """Xuất dữ liệu ra file CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{table_name}_{timestamp}.csv"
        
        conn = sqlite3.connect(self.db_path)
        
        if table_name == "students":
            query = "SELECT id, name, photo_path, created_at FROM students"
        elif table_name == "all_attendance":
            query = """
                SELECT a.student_id, s.name, a.date, a.time, a.status, a.similarity
                FROM attendance a
                JOIN students s ON a.student_id = s.id
                ORDER BY a.date DESC, a.time DESC
            """
        else:
            print("Tên bảng không hợp lệ. Chọn 'students' hoặc 'all_attendance'")
            conn.close()
            return
        
        try:
            df = pd.read_sql_query(query, conn)
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"Đã xuất dữ liệu ra file: {filename}")
        except Exception as e:
            print(f"Lỗi khi xuất file: {e}")
        finally:
            conn.close()

# Hàm chính để sử dụng
def main():
    viewer = DatabaseViewer()
    
    print("=== CÔNG CỤ TRUY VẤN CƠ SỞ DỮ LIỆU ĐIỂM DANH ===")
    
    while True:
        print("\nChọn chức năng:")
        print("1. Xem tất cả sinh viên")
        print("2. Xem điểm danh hôm nay")
        print("3. Xem điểm danh theo ngày")
        print("4. Xem tất cả điểm danh")
        print("5. Tìm kiếm sinh viên")
        print("6. Xem lịch sử điểm danh của sinh viên")
        print("7. Xem thống kê")
        print("8. Xuất dữ liệu ra CSV")
        print("0. Thoát")
        
        choice = input("\nNhập lựa chọn (0-8): ").strip()
        
        if choice == "1":
            viewer.view_all_students()
        
        elif choice == "2":
            viewer.view_attendance_by_date()
        
        elif choice == "3":
            date = input("Nhập ngày (YYYY-MM-DD) hoặc Enter để chọn hôm nay: ").strip()
            if not date:
                date = None
            viewer.view_attendance_by_date(date)
        
        elif choice == "4":
            viewer.view_all_attendance()
        
        elif choice == "5":
            keyword = input("Nhập từ khóa tìm kiếm (tên hoặc ID): ").strip()
            if keyword:
                viewer.search_student(keyword)
            else:
                print("Vui lòng nhập từ khóa!")
        
        elif choice == "6":
            student_id = input("Nhập ID sinh viên: ").strip()
            if student_id:
                viewer.get_student_attendance_history(student_id)
            else:
                print("Vui lòng nhập ID sinh viên!")
        
        elif choice == "7":
            viewer.get_statistics()
        
        elif choice == "8":
            print("Chọn loại dữ liệu để xuất:")
            print("1. Danh sách sinh viên")
            print("2. Tất cả điểm danh")
            export_choice = input("Nhập lựa chọn (1-2): ").strip()
            
            if export_choice == "1":
                viewer.export_to_csv("students")
            elif export_choice == "2":
                viewer.export_to_csv("all_attendance")
            else:
                print("Lựa chọn không hợp lệ!")
        
        elif choice == "0":
            print("Tạm biệt!")
            break
        
        else:
            print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()