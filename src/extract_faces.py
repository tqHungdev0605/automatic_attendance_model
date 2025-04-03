import cv2
import os
import numpy as np
import shutil
from tqdm import tqdm
import random
import mediapipe as mp

# Khởi tạo MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def ensure_dir(directory):
    """Đảm bảo thư mục tồn tại, nếu không tạo mới"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def detect_and_crop_face(image, padding=30):
    """
    Phát hiện và cắt khuôn mặt từ ảnh
    
    Args:
        image: Ảnh đầu vào
        padding: Padding thêm vào xung quanh khuôn mặt
        
    Returns:
        Ảnh khuôn mặt đã cắt hoặc None nếu không phát hiện được khuôn mặt
    """
    # Chuyển đổi sang RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Phát hiện khuôn mặt
    results = face_detection.process(image_rgb)
    
    # Kiểm tra nếu không phát hiện được khuôn mặt
    if not results.detections:
        return None
    
    # Lấy khuôn mặt đầu tiên phát hiện được
    detection = results.detections[0]
    bboxC = detection.location_data.relative_bounding_box
    
    # Lấy kích thước ảnh
    ih, iw, _ = image.shape
    
    # Chuyển đổi tọa độ tương đối sang tọa độ tuyệt đối
    x = int(bboxC.xmin * iw)
    y = int(bboxC.ymin * ih)
    w = int(bboxC.width * iw)
    h = int(bboxC.height * ih)
    
    # Thêm padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(iw - x, w + 2 * padding)
    h = min(ih - y, h + 2 * padding)
    
    # Cắt khuôn mặt
    face_image = image[y:y+h, x:x+w]
    
    return face_image

def extract_frames_from_video(video_path, output_folder, num_frames=180, min_face_size=(100, 100)):
    """
    Trích xuất và cắt khuôn mặt từ video
    
    Args:
        video_path: Đường dẫn đến file video
        output_folder: Thư mục đầu ra để lưu các frame
        num_frames: Số lượng frame cần trích xuất
        min_face_size: Kích thước tối thiểu của khuôn mặt (width, height)
    
    Returns:
        Số lượng frame đã lưu thành công
    """
    # Đảm bảo thư mục đầu ra tồn tại
    ensure_dir(output_folder)
    
    # Đọc video
    video = cv2.VideoCapture(video_path)
    
    # Kiểm tra video đã mở thành công chưa
    if not video.isOpened():
        print(f"Error: Không thể mở video {video_path}")
        return 0
    
    # Lấy tổng số frame trong video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Tính toán khoảng cách giữa các frame để lấy mẫu
    # Chúng ta muốn lấy num_frames frame từ video
    interval = max(1, total_frames // num_frames)
    
    # Counter cho số frame đã lưu
    saved_count = 0
    
    # Duyệt qua các frame để lấy mẫu
    with tqdm(total=num_frames, desc=f"Extracting frames from {os.path.basename(video_path)}") as pbar:
        for frame_idx in range(0, total_frames, interval):
            # Đặt vị trí đọc frame
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # Đọc frame
            success, frame = video.read()
            
            # Kiểm tra đọc frame thành công
            if not success:
                continue
            
            # Phát hiện và cắt khuôn mặt
            face_image = detect_and_crop_face(frame)
            
            # Kiểm tra nếu phát hiện được khuôn mặt và kích thước thoả mãn điều kiện
            if face_image is not None and face_image.shape[0] >= min_face_size[1] and face_image.shape[1] >= min_face_size[0]:
                # Tạo tên file
                filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
                
                # Lưu ảnh
                cv2.imwrite(filename, face_image)
                saved_count += 1
                pbar.update(1)
                
                # Kiểm tra nếu đã lưu đủ số lượng frame
                if saved_count >= num_frames:
                    break
    
    # Giải phóng tài nguyên
    video.release()
    
    print(f"Successfully saved {saved_count} face frames from {os.path.basename(video_path)}")
    return saved_count

def split_data(raw_folder, train_folder, test_folder, val_folder, train_ratio=0.7, test_ratio=0.15):
    """
    Chia dữ liệu vào các thư mục train, test, validation
    
    Args:
        raw_folder: Thư mục chứa dữ liệu gốc
        train_folder: Thư mục dữ liệu huấn luyện
        test_folder: Thư mục dữ liệu kiểm thử
        val_folder: Thư mục dữ liệu xác thực
        train_ratio: Tỷ lệ dữ liệu huấn luyện
        test_ratio: Tỷ lệ dữ liệu kiểm thử (val_ratio = 1 - train_ratio - test_ratio)
    """
    # Đảm bảo các thư mục tồn tại
    ensure_dir(train_folder)
    ensure_dir(test_folder)
    ensure_dir(val_folder)
    
    # Lấy danh sách thư mục con trong raw_folder (mỗi thư mục là một student)
    student_folders = [d for d in os.listdir(raw_folder) if os.path.isdir(os.path.join(raw_folder, d))]
    
    for student_folder in student_folders:
        # Tạo thư mục tương ứng trong train, test, val
        student_id = student_folder
        
        train_student_folder = os.path.join(train_folder, student_id)
        test_student_folder = os.path.join(test_folder, student_id)
        val_student_folder = os.path.join(val_folder, student_id)
        
        ensure_dir(train_student_folder)
        ensure_dir(test_student_folder)
        ensure_dir(val_student_folder)
        
        # Lấy danh sách hình ảnh trong thư mục raw của sinh viên
        raw_student_folder = os.path.join(raw_folder, student_folder)
        images = [f for f in os.listdir(raw_student_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Xáo trộn danh sách ảnh
        random.shuffle(images)
        
        # Tính số lượng ảnh cho mỗi tập
        num_images = len(images)
        num_train = int(num_images * train_ratio)
        num_test = int(num_images * test_ratio)
        
        # Chia thành các tập
        train_images = images[:num_train]
        test_images = images[num_train:num_train+num_test]
        val_images = images[num_train+num_test:]
        
        # Sao chép ảnh vào các thư mục tương ứng
        for img in train_images:
            shutil.copy2(os.path.join(raw_student_folder, img), os.path.join(train_student_folder, img))
        
        for img in test_images:
            shutil.copy2(os.path.join(raw_student_folder, img), os.path.join(test_student_folder, img))
        
        for img in val_images:
            shutil.copy2(os.path.join(raw_student_folder, img), os.path.join(val_student_folder, img))
        
        print(f"Student {student_id}: {len(train_images)} training, {len(test_images)} testing, {len(val_images)} validation images")

def process_all_videos(video_folder, raw_folder, train_folder, test_folder, val_folder, frames_per_video=180):
    """
    Xử lý tất cả các video trong thư mục
    
    Args:
        video_folder: Thư mục chứa các video
        raw_folder: Thư mục đầu ra cho ảnh raw
        train_folder: Thư mục dữ liệu huấn luyện
        test_folder: Thư mục dữ liệu kiểm thử
        val_folder: Thư mục dữ liệu xác thực
        frames_per_video: Số lượng frame cần trích xuất từ mỗi video
    """
    # Đảm bảo các thư mục tồn tại
    ensure_dir(raw_folder)
    ensure_dir(train_folder)
    ensure_dir(test_folder)
    ensure_dir(val_folder)
    
    # Lấy danh sách video trong thư mục
    videos = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    for video in videos:
        video_path = os.path.join(video_folder, video)
        
        # Lấy tên sinh viên từ tên video (bỏ đuôi file)
        student_id = os.path.splitext(video)[0]
        
        # Tạo thư mục cho sinh viên trong raw
        raw_student_folder = os.path.join(raw_folder, student_id)
        ensure_dir(raw_student_folder)
        
        # Trích xuất frame từ video
        extract_frames_from_video(video_path, raw_student_folder, num_frames=frames_per_video)
    
    # Chia dữ liệu vào các thư mục train, test, val
    split_data(raw_folder, train_folder, test_folder, val_folder)

if __name__ == "__main__":
    # Đường dẫn đến các thư mục
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # KLTN/
    video_folder = os.path.join(project_dir, "data", "videos")
    raw_folder = os.path.join(project_dir, "data", "raw")
    train_folder = os.path.join(project_dir, "data", "train")
    test_folder = os.path.join(project_dir, "data", "test")
    val_folder = os.path.join(project_dir, "data", "validation")
    
    # Số lượng frame cần trích xuất từ mỗi video
    frames_per_video = 180  # Khoảng 3 frame mỗi giây cho video 60s
    
    # Xử lý tất cả các video
    process_all_videos(video_folder, raw_folder, train_folder, test_folder, val_folder, frames_per_video)