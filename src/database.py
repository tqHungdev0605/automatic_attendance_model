# database.py - Simplified version
import sqlite3
import numpy as np
import json
import os
from datetime import datetime

class FaceDatabase:
    def __init__(self, db_path="face_attendance.db"):
        self.db_path = db_path
        self.create_tables()
    
    def create_tables(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bảng lưu thông tin sinh viên (đã loại bỏ class_id)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            embedding TEXT NOT NULL,
            photo_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Bảng lưu thông tin điểm danh (đã đơn giản hóa)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            date DATE NOT NULL,
            time TIME NOT NULL,
            status TEXT DEFAULT 'present',
            similarity REAL,
            FOREIGN KEY (student_id) REFERENCES students(id),
            UNIQUE(student_id, date)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def embedding_to_db(self, embedding):
        """Chuyển đổi embedding numpy array thành chuỗi để lưu vào DB"""
        return json.dumps(embedding.tolist())
    
    def embedding_from_db(self, embedding_str):
        """Chuyển đổi chuỗi từ DB thành numpy array"""
        return np.array(json.loads(embedding_str))
    
    def add_student(self, student_id, name, embedding, photo_path=None):
        """Thêm sinh viên mới vào cơ sở dữ liệu"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_str = self.embedding_to_db(embedding)
        
        try:
            cursor.execute(
                "INSERT INTO students (id, name, embedding, photo_path) VALUES (?, ?, ?, ?)",
                (student_id, name, embedding_str, photo_path)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Sinh viên đã tồn tại
            return False
        finally:
            conn.close()
    
    def update_student(self, student_id, name=None, embedding=None, photo_path=None):
        """Cập nhật thông tin sinh viên"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Lấy thông tin hiện tại của sinh viên
        cursor.execute("SELECT name, embedding, photo_path FROM students WHERE id=?", (student_id,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return False  # Sinh viên không tồn tại
        
        current_name, current_embedding, current_photo_path = result
        
        # Cập nhật các trường nếu được cung cấp
        new_name = name if name is not None else current_name
        new_embedding_str = self.embedding_to_db(embedding) if embedding is not None else current_embedding
        new_photo_path = photo_path if photo_path is not None else current_photo_path
        
        cursor.execute(
            "UPDATE students SET name=?, embedding=?, photo_path=? WHERE id=?",
            (new_name, new_embedding_str, new_photo_path, student_id)
        )
        
        conn.commit()
        conn.close()
        return True
    
    def delete_student(self, student_id):
        """Xóa sinh viên khỏi cơ sở dữ liệu"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM students WHERE id=?", (student_id,))
        affected_rows = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return affected_rows > 0
    
    def get_student(self, student_id):
        """Lấy thông tin sinh viên theo ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name, embedding, photo_path FROM students WHERE id=?", (student_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            id, name, embedding_str, photo_path = result
            return {
                'id': id,
                'name': name,
                'embedding': embedding_str,
                'photo_path': photo_path
            }
        return None
    
    def get_all_students(self):
        """Lấy danh sách tất cả sinh viên"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name, embedding, photo_path FROM students")
        results = cursor.fetchall()
        conn.close()
        
        students = []
        for result in results:
            id, name, embedding_str, photo_path = result
            students.append({
                'id': id,
                'name': name,
                'embedding': embedding_str,
                'photo_path': photo_path
            })
        
        return students
    
    def mark_attendance(self, student_id, similarity=None, status="present", date=None):
        """Đánh dấu sinh viên có mặt - CHỈ THÊM MỚI, KHÔNG GHI ĐÈ"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        time = datetime.now().strftime("%H:%M:%S")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Kiểm tra xem sinh viên đã điểm danh hôm nay chưa
            cursor.execute(
                "SELECT id FROM attendance WHERE student_id = ? AND date = ?",
                (student_id, date)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Sinh viên đã điểm danh rồi, không thêm nữa
                conn.close()
                return False
            
            # Thêm bản ghi điểm danh mới
            cursor.execute(
                "INSERT INTO attendance (student_id, date, time, status, similarity) VALUES (?, ?, ?, ?, ?)",
                (student_id, date, time, status, similarity)
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error marking attendance: {e}")
            return False
        finally:
            conn.close()
    
    def get_attendance(self, date=None):
        """Lấy danh sách điểm danh theo ngày"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Để kết quả trả về dưới dạng dictionary
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT a.id, a.student_id, s.name, a.date, a.time, a.status, a.similarity
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE a.date = ?
            ORDER BY a.time
        """, (date,))
        
        results = cursor.fetchall()
        
        attendance_records = []
        for row in results:
            attendance_records.append(dict(row))
        
        conn.close()
        return attendance_records
    
    def get_all_attendance(self):
        """Lấy tất cả dữ liệu điểm danh"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT a.id, a.student_id, s.name, a.date, a.time, a.status, a.similarity
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            ORDER BY a.date DESC, a.time DESC
        """)
        
        results = cursor.fetchall()
        
        attendance_records = []
        for row in results:
            attendance_records.append(dict(row))
        
        conn.close()
        return attendance_records
    
    def is_student_attended_today(self, student_id, date=None):
        """Kiểm tra sinh viên đã điểm danh hôm nay chưa"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id FROM attendance WHERE student_id = ? AND date = ?",
            (student_id, date)
        )
        result = cursor.fetchone()
        conn.close()
        
        return result is not None