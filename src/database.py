import sqlite3
import os
import pickle
from datetime import datetime

class SimpleDatabase:
    def __init__(self, db_path="data/students.db"):
        self.db_path = db_path
        
        # Tạo thư mục nếu chưa có
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        
        self.create_tables()
    
    def get_connection(self):
        """Tạo kết nối database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def create_tables(self):
        """Tạo bảng cần thiết"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Bảng sinh viên
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                class_name TEXT NOT NULL,
                embedding_path TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Bảng điểm danh
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                student_name TEXT,
                attendance_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                FOREIGN KEY (student_id) REFERENCES students (student_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # ========== QUẢN LÝ SINH VIÊN ==========
    def add_student(self, student_id, full_name, class_name, embedding_path=None):
        """Thêm sinh viên mới"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO students (student_id, full_name, class_name, embedding_path)
                VALUES (?, ?, ?, ?)
            ''', (student_id, full_name, class_name, embedding_path))
            
            conn.commit()
            conn.close()
            print(f"Thêm sinh viên {student_id} - {full_name} thành công!")
            return True
            
        except sqlite3.IntegrityError:
            conn.close()
            print(f"Lỗi: Sinh viên {student_id} đã tồn tại!")
            return False
        except Exception as e:
            conn.close()
            print(f"Lỗi thêm sinh viên: {e}")
            return False
    
    def delete_student(self, student_id):
        """Xóa sinh viên"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Lấy thông tin sinh viên trước khi xóa
            cursor.execute('SELECT * FROM students WHERE student_id = ?', (student_id,))
            student = cursor.fetchone()
            
            if not student:
                conn.close()
                print(f"Không tìm thấy sinh viên {student_id}")
                return False
            
            # Xóa file embedding nếu có
            if student['embedding_path'] and os.path.exists(student['embedding_path']):
                os.remove(student['embedding_path'])
                print(f"Đã xóa file embedding: {student['embedding_path']}")
            
            # Xóa bản ghi điểm danh
            cursor.execute('DELETE FROM attendance WHERE student_id = ?', (student_id,))
            
            # Xóa sinh viên
            cursor.execute('DELETE FROM students WHERE student_id = ?', (student_id,))
            
            conn.commit()
            conn.close()
            print(f"Đã xóa sinh viên {student_id} - {student['full_name']}")
            return True
            
        except Exception as e:
            conn.close()
            print(f"Lỗi xóa sinh viên: {e}")
            return False
    
    def get_student(self, student_id):
        """Lấy thông tin sinh viên"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM students WHERE student_id = ?', (student_id,))
        student = cursor.fetchone()
        conn.close()
        
        if student:
            return dict(student)
        return None
    
    def get_all_students(self):
        """Lấy tất cả sinh viên"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM students ORDER BY full_name')
        students = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return students
    
    def update_student_embedding(self, student_id, embedding_path):
        """Cập nhật embedding cho sinh viên"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE students 
            SET embedding_path = ?
            WHERE student_id = ?
        ''', (embedding_path, student_id))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if success:
            print(f"Cập nhật embedding cho {student_id} thành công!")
        return success
    
    # ========== ĐIỂM DANH ==========
    def mark_attendance(self, student_id, confidence=None):
        """Ghi nhận điểm danh"""
        try:
            # Lấy thông tin sinh viên
            student = self.get_student(student_id)
            if not student:
                print(f"Không tìm thấy sinh viên {student_id}")
                return False
            
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO attendance (student_id, student_name, confidence)
                VALUES (?, ?, ?)
            ''', (student_id, student['full_name'], confidence))
            
            conn.commit()
            conn.close()
            
            print(f"Điểm danh thành công: {student['full_name']} (ID: {student_id})")
            if confidence:
                print(f"Độ tin cậy: {confidence:.2f}")
            
            return True
            
        except Exception as e:
            print(f"Lỗi ghi điểm danh: {e}")
            return False
    
    def get_attendance_today(self):
        """Lấy danh sách điểm danh hôm nay"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        today = datetime.now().date()
        cursor.execute('''
            SELECT * FROM attendance 
            WHERE DATE(attendance_time) = ?
            ORDER BY attendance_time DESC
        ''', (today,))
        
        attendance_list = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return attendance_list
    
    def get_attendance_history(self, student_id=None, limit=50):
        """Lấy lịch sử điểm danh"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if student_id:
            cursor.execute('''
                SELECT * FROM attendance 
                WHERE student_id = ?
                ORDER BY attendance_time DESC
                LIMIT ?
            ''', (student_id, limit))
        else:
            cursor.execute('''
                SELECT * FROM attendance 
                ORDER BY attendance_time DESC
                LIMIT ?
            ''', (limit,))
        
        history = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return history


# Test đơn giản
def test_database():
    print("=== TEST DATABASE ===")
    
    db = SimpleDatabase("data/test_students.db")
    
    # Test thêm sinh viên
    db.add_student("SV001", "Nguyen Van A", "CNTT01")
    db.add_student("SV002", "Tran Thi B", "CNTT01")
    
    # Test lấy danh sách
    students = db.get_all_students()
    print(f"Có {len(students)} sinh viên:")
    for s in students:
        print(f"- {s['student_id']}: {s['full_name']} - {s['class_name']}")
    
    # Test điểm danh
    db.mark_attendance("SV001", 0.95)
    db.mark_attendance("SV002", 0.87)
    
    # Test lấy điểm danh hôm nay
    today_attendance = db.get_attendance_today()
    print(f"\nĐiểm danh hôm nay ({len(today_attendance)} người):")
    for a in today_attendance:
        print(f"- {a['student_name']} ({a['student_id']}) - {a['attendance_time']}")
    
    # Test xóa sinh viên
    db.delete_student("SV002")
    
    print("\nTest hoàn thành!")

if __name__ == "__main__":
    test_database()