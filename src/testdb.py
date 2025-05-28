import unittest
import os
import tempfile
import shutil
from datetime import datetime, timedelta
import sys

# Import database class (giả sử file database.py ở cùng thư mục)
from database import SimpleDatabase

class TestSimpleDatabase(unittest.TestCase):
    
    def setUp(self):
        """Thiết lập trước mỗi test - tạo database tạm thời"""
        # Tạo thư mục tạm thời cho test
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_students.db")
        self.db = SimpleDatabase(self.db_path)
        
        # Dữ liệu test mẫu
        self.sample_students = [
            ("SV001", "Nguyễn Văn An", "CNTT01"),
            ("SV002", "Trần Thị Bình", "CNTT01"), 
            ("SV003", "Lê Hoàng Cường", "CNTT02"),
            ("SV004", "Phạm Thị Dung", "CNTT02"),
            ("SV005", "Hoàng Văn Em", "CNTT01")
        ]
    
    def tearDown(self):
        """Dọn dẹp sau mỗi test - xóa database tạm thời"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    # ========== TEST QUẢN LÝ SINH VIÊN ==========
    
    def test_add_student_success(self):
        """Test thêm sinh viên thành công"""
        result = self.db.add_student("SV001", "Nguyễn Văn An", "CNTT01")
        self.assertTrue(result)
        
        # Kiểm tra sinh viên đã được thêm
        student = self.db.get_student("SV001")
        self.assertIsNotNone(student)
        self.assertEqual(student['student_id'], "SV001")
        self.assertEqual(student['full_name'], "Nguyễn Văn An")
        self.assertEqual(student['class_name'], "CNTT01")
    
    def test_add_duplicate_student(self):
        """Test thêm sinh viên trùng ID"""
        # Thêm sinh viên đầu tiên
        self.db.add_student("SV001", "Nguyễn Văn An", "CNTT01")
        
        # Thêm sinh viên trùng ID
        result = self.db.add_student("SV001", "Trần Văn Bình", "CNTT02")
        self.assertFalse(result)
    
    def test_get_student_exists(self):
        """Test lấy thông tin sinh viên tồn tại"""
        self.db.add_student("SV001", "Nguyễn Văn An", "CNTT01")
        student = self.db.get_student("SV001")
        
        self.assertIsNotNone(student)
        self.assertEqual(student['student_id'], "SV001")
        self.assertEqual(student['full_name'], "Nguyễn Văn An")
    
    def test_get_student_not_exists(self):
        """Test lấy thông tin sinh viên không tồn tại"""
        student = self.db.get_student("SV999")
        self.assertIsNone(student)
    
    def test_get_all_students(self):
        """Test lấy tất cả sinh viên"""
        # Thêm nhiều sinh viên
        for student_id, name, class_name in self.sample_students:
            self.db.add_student(student_id, name, class_name)
        
        students = self.db.get_all_students()
        self.assertEqual(len(students), len(self.sample_students))
        
        # Kiểm tra sắp xếp theo tên
        names = [s['full_name'] for s in students]
        self.assertEqual(names, sorted(names))
    
    def test_update_student_embedding(self):
        """Test cập nhật embedding cho sinh viên"""
        self.db.add_student("SV001", "Nguyễn Văn An", "CNTT01")
        
        embedding_path = "embeddings/SV001.pkl"
        result = self.db.update_student_embedding("SV001", embedding_path)
        self.assertTrue(result)
        
        # Kiểm tra đã cập nhật
        student = self.db.get_student("SV001")
        self.assertEqual(student['embedding_path'], embedding_path)
    
    def test_update_embedding_nonexistent_student(self):
        """Test cập nhật embedding cho sinh viên không tồn tại"""
        result = self.db.update_student_embedding("SV999", "some/path.pkl")
        self.assertFalse(result)
    
    def test_delete_student_success(self):
        """Test xóa sinh viên thành công"""
        self.db.add_student("SV001", "Nguyễn Văn An", "CNTT01")
        
        # Thêm điểm danh cho sinh viên
        self.db.mark_attendance("SV001", 0.95)
        
        # Xóa sinh viên
        result = self.db.delete_student("SV001")
        self.assertTrue(result)
        
        # Kiểm tra sinh viên đã bị xóa
        student = self.db.get_student("SV001")
        self.assertIsNone(student)
        
        # Kiểm tra điểm danh cũng bị xóa
        history = self.db.get_attendance_history()
        attendance_for_student = [a for a in history if a['student_id'] == "SV001"]
        self.assertEqual(len(attendance_for_student), 0)
    
    def test_delete_nonexistent_student(self):
        """Test xóa sinh viên không tồn tại"""
        result = self.db.delete_student("SV999")
        self.assertFalse(result)
    
    # ========== TEST ĐIỂM DANH ==========
    
    def test_mark_attendance_success(self):
        """Test điểm danh thành công"""
        self.db.add_student("SV001", "Nguyễn Văn An", "CNTT01")
        
        result = self.db.mark_attendance("SV001", 0.95)
        self.assertTrue(result)
        
        # Kiểm tra điểm danh đã được ghi
        today_attendance = self.db.get_attendance_today()
        self.assertEqual(len(today_attendance), 1)
        self.assertEqual(today_attendance[0]['student_id'], "SV001")
        self.assertEqual(today_attendance[0]['confidence'], 0.95)
    
    def test_mark_attendance_nonexistent_student(self):
        """Test điểm danh sinh viên không tồn tại"""
        result = self.db.mark_attendance("SV999", 0.95)
        self.assertFalse(result)
    
    def test_mark_attendance_without_confidence(self):
        """Test điểm danh không có confidence"""
        self.db.add_student("SV001", "Nguyễn Văn An", "CNTT01")
        
        result = self.db.mark_attendance("SV001")
        self.assertTrue(result)
        
        today_attendance = self.db.get_attendance_today()
        self.assertEqual(len(today_attendance), 1)
        self.assertIsNone(today_attendance[0]['confidence'])
    
    def test_get_attendance_today(self):
        """Test lấy điểm danh hôm nay"""
        # Thêm sinh viên
        for student_id, name, class_name in self.sample_students[:3]:
            self.db.add_student(student_id, name, class_name)
        
        # Điểm danh
        self.db.mark_attendance("SV001", 0.95)
        self.db.mark_attendance("SV002", 0.87)
        self.db.mark_attendance("SV003", 0.92)
        
        today_attendance = self.db.get_attendance_today()
        self.assertEqual(len(today_attendance), 3)
        
        # Kiểm tra sắp xếp theo thời gian giảm dần
        times = [a['attendance_time'] for a in today_attendance]
        self.assertEqual(times, sorted(times, reverse=True))
    
    def test_get_attendance_history_all(self):
        """Test lấy lịch sử điểm danh tất cả"""
        # Thêm sinh viên và điểm danh
        self.db.add_student("SV001", "Nguyễn Văn An", "CNTT01")
        self.db.add_student("SV002", "Trần Thị Bình", "CNTT01")
        
        self.db.mark_attendance("SV001", 0.95)
        self.db.mark_attendance("SV002", 0.87)
        self.db.mark_attendance("SV001", 0.92)
        
        history = self.db.get_attendance_history()
        self.assertEqual(len(history), 3)
    
    def test_get_attendance_history_by_student(self):
        """Test lấy lịch sử điểm danh theo sinh viên"""
        self.db.add_student("SV001", "Nguyễn Văn An", "CNTT01")
        self.db.add_student("SV002", "Trần Thị Bình", "CNTT01")
        
        self.db.mark_attendance("SV001", 0.95)
        self.db.mark_attendance("SV002", 0.87)
        self.db.mark_attendance("SV001", 0.92)
        
        history_sv001 = self.db.get_attendance_history("SV001")
        self.assertEqual(len(history_sv001), 2)
        
        for record in history_sv001:
            self.assertEqual(record['student_id'], "SV001")
    
    def test_get_attendance_history_with_limit(self):
        """Test lấy lịch sử điểm danh với giới hạn"""
        self.db.add_student("SV001", "Nguyễn Văn An", "CNTT01")
        
        # Tạo nhiều bản ghi điểm danh
        for i in range(10):
            self.db.mark_attendance("SV001", 0.9 + i * 0.01)
        
        history = self.db.get_attendance_history(limit=5)
        self.assertEqual(len(history), 5)
    
    # ========== TEST TÍCH HỢP ==========
    
    def test_full_workflow(self):
        """Test quy trình đầy đủ"""
        # 1. Thêm sinh viên
        self.db.add_student("SV001", "Nguyễn Văn An", "CNTT01")
        self.db.add_student("SV002", "Trần Thị Bình", "CNTT01")
        
        # 2. Cập nhật embedding
        self.db.update_student_embedding("SV001", "embeddings/SV001.pkl")
        
        # 3. Điểm danh
        self.db.mark_attendance("SV001", 0.95)
        self.db.mark_attendance("SV002", 0.87)
        
        # 4. Kiểm tra kết quả
        students = self.db.get_all_students()
        self.assertEqual(len(students), 2)
        
        today_attendance = self.db.get_attendance_today()
        self.assertEqual(len(today_attendance), 2)
        
        # 5. Xóa một sinh viên
        self.db.delete_student("SV002")
        
        students_after_delete = self.db.get_all_students()
        self.assertEqual(len(students_after_delete), 1)
        self.assertEqual(students_after_delete[0]['student_id'], "SV001")


def run_manual_tests():
    """Chạy test thủ công để quan sát output"""
    print("=" * 50)
    print("CHẠY TEST THỦ CÔNG")
    print("=" * 50)
    
    # Tạo database test
    test_db = SimpleDatabase("data/manual_test.db")
    
    print("\n1. Test thêm sinh viên:")
    test_db.add_student("SV001", "Nguyễn Văn An", "CNTT01")
    test_db.add_student("SV002", "Trần Thị Bình", "CNTT01")
    test_db.add_student("SV003", "Lê Hoàng Cường", "CNTT02")
    
    print("\n2. Danh sách sinh viên:")
    students = test_db.get_all_students()
    for s in students:
        print(f"   - {s['student_id']}: {s['full_name']} ({s['class_name']})")
    
    print("\n3. Test điểm danh:")
    test_db.mark_attendance("SV001", 0.95)
    test_db.mark_attendance("SV002", 0.87)
    
    print("\n4. Điểm danh hôm nay:")
    today = test_db.get_attendance_today()
    for a in today:
        print(f"   - {a['student_name']} (ID: {a['student_id']}) - Confidence: {a['confidence']}")
    
    print("\n5. Test cập nhật embedding:")
    test_db.update_student_embedding("SV001", "embeddings/SV001.pkl")
    updated_student = test_db.get_student("SV001")
    print(f"   Embedding path cho SV001: {updated_student['embedding_path']}")
    
    print("\n6. Test xóa sinh viên:")
    test_db.delete_student("SV003")
    
    print("\n7. Danh sách sinh viên sau khi xóa:")
    students_after = test_db.get_all_students()
    for s in students_after:
        print(f"   - {s['student_id']}: {s['full_name']} ({s['class_name']})")
    
    print("\n" + "=" * 50)
    print("HOÀN THÀNH TEST THỦ CÔNG")
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test database cho hệ thống điểm danh")
    parser.add_argument("--manual", action="store_true", help="Chạy test thủ công")
    parser.add_argument("--unit", action="store_true", help="Chạy unit test")
    parser.add_argument("--all", action="store_true", help="Chạy tất cả test")
    
    args = parser.parse_args()
    
    if args.all or (not args.manual and not args.unit):
        # Mặc định chạy tất cả
        print("Chạy Unit Tests...")
        unittest.main(argv=[''], exit=False, verbosity=2)
        
        print("\n" + "="*50)
        run_manual_tests()
        
    elif args.unit:
        unittest.main(verbosity=2)
        
    elif args.manual:
        run_manual_tests()