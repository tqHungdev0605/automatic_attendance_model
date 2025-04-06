import cv2
import os
import numpy as np
import shutil
from tqdm import tqdm
import random
import mediapipe as mp
import math

# Khởi tạo MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def ensure_dir(directory):
    """Đảm bảo thư mục tồn tại, nếu không tạo mới"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def detect_and_crop_face(image, target_size=224, padding_percent=0.2):
    """
    Phát hiện, cắt khuôn mặt thành hình vuông và resize về kích thước cố định
    
    Args:
        image: Ảnh đầu vào
        target_size: Kích thước đích của ảnh đầu ra (pixel)
        padding_percent: Phần trăm padding thêm vào xung quanh khuôn mặt (0.2 = 20%)
        
    Returns:
        Ảnh khuôn mặt đã cắt thành hình vuông và resize hoặc None nếu không phát hiện được khuôn mặt
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
    
    # Tính toán padding dựa trên phần trăm kích thước khuôn mặt
    padding_x = int(w * padding_percent)
    padding_y = int(h * padding_percent)
    
    # Tính toán tâm của khuôn mặt
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Xác định kích thước lớn nhất giữa chiều rộng và chiều cao (với padding)
    size = max(w + 2 * padding_x, h + 2 * padding_y)
    
    # Tính toán tọa độ mới để cắt hình vuông từ tâm
    new_x = max(0, center_x - size // 2)
    new_y = max(0, center_y - size // 2)
    
    # Đảm bảo khuôn mặt nằm trong ảnh
    if new_x + size > iw:
        new_x = iw - size
    if new_y + size > ih:
        new_y = ih - size
    
    # Đảm bảo không vượt quá kích thước ảnh
    if new_x < 0:
        new_x = 0
    if new_y < 0:
        new_y = 0
    
    size = min(size, iw - new_x, ih - new_y)
    
    # Cắt khuôn mặt thành hình vuông
    face_image = image[new_y:new_y+size, new_x:new_x+size]
    
    # Resize về kích thước cố định
    face_image = cv2.resize(face_image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    return face_image

def calculate_frame_difference(frame1, frame2):
    """
    Tính toán sự khác biệt giữa hai frame
    
    Args:
        frame1: Frame thứ nhất
        frame2: Frame thứ hai
        
    Returns:
        Điểm số khác biệt (càng cao càng khác biệt)
    """
    # Chuyển đổi sang grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Tính toán sự khác biệt
    diff = cv2.absdiff(gray1, gray2)
    score = np.sum(diff) / (diff.shape[0] * diff.shape[1])
    
    return score

def extract_frames_from_video(video_path, output_folder, num_frames=100, target_size=224, diff_threshold=5.0):
    """
    Trích xuất và cắt khuôn mặt đa dạng từ video
    
    Args:
        video_path: Đường dẫn đến file video
        output_folder: Thư mục đầu ra để lưu các frame
        num_frames: Số lượng frame cần trích xuất
        target_size: Kích thước đích của ảnh đầu ra (pixel)
        diff_threshold: Ngưỡng khác biệt tối thiểu giữa các frame liên tiếp
    
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
    
    # Lấy tổng số frame và FPS
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps  # Thời lượng video (giây)
    
    print(f"Video info: {total_frames} frames, {fps} fps, {duration:.2f} seconds")
    
    # Chia video thành num_segments phân đoạn và lấy frames từ mỗi phân đoạn
    num_segments = min(num_frames, 20)  # Chia tối đa 20 phân đoạn
    frames_per_segment = num_frames // num_segments
    segment_length = total_frames // num_segments
    
    # Danh sách lưu các frame đã trích xuất và điểm khác biệt
    candidate_frames = []
    last_saved_face = None
    
    # Duyệt qua các phân đoạn
    with tqdm(total=num_frames, desc=f"Extracting frames from {os.path.basename(video_path)}") as pbar:
        for segment in range(num_segments):
            # Vị trí bắt đầu và kết thúc của phân đoạn
            start_frame = segment * segment_length
            end_frame = min(start_frame + segment_length, total_frames)
            
            # Lấy nhiều frame từ phân đoạn để chọn frame tốt nhất
            sample_indices = np.linspace(start_frame, end_frame - 1, frames_per_segment * 3, dtype=int)
            
            segment_candidates = []
            
            for frame_idx in sample_indices:
                # Đặt vị trí đọc frame
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                # Đọc frame
                success, frame = video.read()
                
                # Kiểm tra đọc frame thành công
                if not success:
                    continue
                
                # Phát hiện, cắt khuôn mặt thành hình vuông và resize
                face_image = detect_and_crop_face(frame, target_size=target_size)
                
                # Kiểm tra nếu phát hiện được khuôn mặt
                if face_image is not None:
                    # Tính toán điểm khác biệt với frame cuối cùng đã lưu
                    diff_score = 0
                    if last_saved_face is not None:
                        # Resize để so sánh cùng kích thước
                        resized_face = cv2.resize(face_image, (last_saved_face.shape[1], last_saved_face.shape[0]))
                        diff_score = calculate_frame_difference(resized_face, last_saved_face)
                    
                    # Lưu frame và điểm số
                    segment_candidates.append({
                        'frame_idx': frame_idx,
                        'face_image': face_image,
                        'diff_score': diff_score
                    })
            
            # Sắp xếp các frame theo điểm khác biệt (từ cao xuống thấp)
            segment_candidates.sort(key=lambda x: x['diff_score'], reverse=True)
            
            # Chọn các frame tốt nhất từ phân đoạn này
            selected_candidates = segment_candidates[:frames_per_segment]
            
            # Thêm vào danh sách ứng viên chung
            candidate_frames.extend(selected_candidates)
            
            # Cập nhật frame cuối cùng được lưu
            if selected_candidates:
                last_saved_face = selected_candidates[0]['face_image']
            
            pbar.update(len(selected_candidates))
    
    # Sắp xếp lại tất cả các frame theo thứ tự video
    candidate_frames.sort(key=lambda x: x['frame_idx'])
    
    # Lưu các frame đã chọn
    saved_count = 0
    for i, candidate in enumerate(candidate_frames):
        if i >= num_frames:
            break
            
        # Tạo tên file
        filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
        
        # Lưu ảnh
        cv2.imwrite(filename, candidate['face_image'])
        saved_count += 1
    
    # Giải phóng tài nguyên
    video.release()
    
    print(f"Successfully saved {saved_count} diverse face frames from {os.path.basename(video_path)}")
    return saved_count

def process_all_videos(video_folder, raw_folder, frames_per_video=100):
    """
    Xử lý tất cả các video trong thư mục
    
    Args:
        video_folder: Thư mục chứa các video
        raw_folder: Thư mục đầu ra cho ảnh raw
        frames_per_video: Số lượng frame cần trích xuất từ mỗi video
    """
    # Đảm bảo các thư mục tồn tại
    ensure_dir(raw_folder)
    
    # Lấy danh sách video trong thư mục
    videos = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    for video in videos:
        video_path = os.path.join(video_folder, video)
        
        # Lấy tên sinh viên từ tên video
        student_id = os.path.splitext(video)[0]
        
        # Tạo thư mục cho sinh viên trong raw
        raw_student_folder = os.path.join(raw_folder, student_id)
        ensure_dir(raw_student_folder)
        
        # Trích xuất frame từ video
        extract_frames_from_video(video_path, raw_student_folder, num_frames=frames_per_video)
    
    print(f"Đã hoàn thành trích xuất ảnh từ video, lưu trong thư mục {raw_folder}")

if __name__ == "__main__":
    # Đường dẫn đến các thư mục
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Graduate_thesis/
    video_folder = os.path.join(project_dir, "data", "videos")
    raw_folder = os.path.join(project_dir, "data", "raw")
    
    # Cấu hình
    frames_per_video = 100 
    image_size = 224
    
    print(f"Trích xuất {frames_per_video} ảnh từ mỗi video, kích thước {image_size}x{image_size} pixel")
    
    # Xử lý tất cả các video
    process_all_videos(video_folder, raw_folder, frames_per_video)