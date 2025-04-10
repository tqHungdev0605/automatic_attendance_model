import cv2
import os
import numpy as np
import shutil
from tqdm import tqdm
import random
import mediapipe as mp
from scipy.spatial import distance

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
    
    # Lấy khuôn mặt đầu tiên có độ tin cậy cao nhất
    detection = results.detections[0]
    bboxC = detection.location_data.relative_bounding_box
    
    # Lấy kích thước ảnh
    ih, iw, _ = image.shape
    
    # Chuyển đổi tọa độ tương đối sang tọa độ tuyệt đối
    x = int(bboxC.xmin * iw)
    y = int(bboxC.ymin * ih)
    w = int(bboxC.width * iw)
    h = int(bboxC.height * ih)
    
    # Kiểm tra nếu khuôn mặt quá nhỏ
    min_face_size = 50
    if w < min_face_size or h < min_face_size:
        return None
    
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
    
    # Resize để đảm bảo cùng kích thước
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    
    # Tính toán sự khác biệt sử dụng MSE (Mean Squared Error)
    mse = np.mean((gray1.astype(np.float32) - gray2.astype(np.float32)) ** 2)
    
    # Tính toán histogram khoảng cách
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    # Chuẩn hóa histogram
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    # Tính khoảng cách histogram
    hist_dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    
    # Kết hợp các đo lường khác biệt
    difference_score = 0.5 * mse + 0.5 * hist_dist * 1000
    
    return difference_score

def segment_video(video_path, num_segments=12):
    """
    Phân đoạn video thành các đoạn có độ dài bằng nhau
    
    Args:
        video_path: Đường dẫn tới file video
        num_segments: Số lượng đoạn cần chia
        
    Returns:
        Danh sách các đoạn (segment), mỗi đoạn là một khoảng frame [start, end]
    """
    # Đọc video
    video = cv2.VideoCapture(video_path)
    
    # Lấy tổng số frame
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Giải phóng tài nguyên
    video.release()
    
    # Tính kích thước của mỗi đoạn
    segment_size = total_frames // num_segments
    
    # Tạo danh sách các đoạn
    segments = []
    for i in range(num_segments):
        start_frame = i * segment_size
        end_frame = min((i + 1) * segment_size - 1, total_frames - 1)
        segments.append((start_frame, end_frame))
    
    return segments

def extract_diverse_frames_from_video(video_path, output_folder, num_segments=12, frames_per_segment=8, target_size=224):
    """
    Trích xuất và cắt các khung hình đa dạng từ video
    
    Args:
        video_path: Đường dẫn đến file video
        output_folder: Thư mục đầu ra để lưu các frame
        num_segments: Số lượng đoạn video được chia
        frames_per_segment: Số frame trích xuất từ mỗi đoạn
        target_size: Kích thước ảnh đầu ra (pixel)
    
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
    
    # Lấy thông tin video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"Video info: {total_frames} frames, {fps} fps, {duration:.2f} seconds")
    
    # Phân đoạn video
    segments = segment_video(video_path, num_segments)
    
    # Tạo danh sách để lưu các frame đã trích xuất
    saved_frames = []
    
    # Duyệt qua từng đoạn để trích xuất frame
    with tqdm(total=num_segments * frames_per_segment, desc=f"Extracting frames from {os.path.basename(video_path)}") as pbar:
        for segment_idx, (start_frame, end_frame) in enumerate(segments):
            # Lấy một số frame từ segment để tìm kiếm sự đa dạng
            sample_frames = []
            
            # Xác định số lượng frame cần lấy mẫu (khoảng 3 lần số lượng frames_per_segment)
            num_samples = min(frames_per_segment * 3, end_frame - start_frame + 1)
            
            # Lấy mẫu các frame cách đều nhau trong đoạn
            sample_indices = np.linspace(start_frame, end_frame, num_samples, dtype=int)
            
            # Duyệt qua các frame mẫu để đọc và kiểm tra khuôn mặt
            for frame_idx in sample_indices:
                # Đặt vị trí đọc frame
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                # Đọc frame
                success, frame = video.read()
                
                # Kiểm tra đọc frame thành công
                if not success:
                    continue
                
                # Phát hiện và cắt khuôn mặt
                face_image = detect_and_crop_face(frame, target_size=target_size)
                
                # Kiểm tra nếu phát hiện được khuôn mặt
                if face_image is not None:
                    sample_frames.append((frame_idx, face_image))
            
            # Nếu không có đủ frame với khuôn mặt, bỏ qua segment này
            if len(sample_frames) < frames_per_segment:
                for _ in range(len(sample_frames)):
                    pbar.update(1)
                continue
            
            # Thuật toán lựa chọn frame đa dạng
            selected_frames = []
            
            # Chọn frame đầu tiên (có thể là frame giữa segment)
            middle_idx = len(sample_frames) // 2
            selected_frames.append(sample_frames[middle_idx])
            
            # Loại bỏ frame đã chọn khỏi danh sách mẫu
            remaining_frames = sample_frames[:middle_idx] + sample_frames[middle_idx+1:]
            
            # Lựa chọn các frame còn lại dựa trên độ khác biệt
            while len(selected_frames) < frames_per_segment and remaining_frames:
                # Tính độ khác biệt của mỗi frame còn lại so với các frame đã chọn
                max_diff_score = -1
                max_diff_idx = -1
                
                for i, (frame_idx, frame) in enumerate(remaining_frames):
                    # Tính tổng độ khác biệt với tất cả các frame đã chọn
                    total_diff = sum(calculate_frame_difference(frame, selected_frame[1]) for selected_frame in selected_frames)
                    
                    # Lấy frame có tổng độ khác biệt lớn nhất
                    if total_diff > max_diff_score:
                        max_diff_score = total_diff
                        max_diff_idx = i
                
                # Thêm frame có độ khác biệt lớn nhất vào danh sách đã chọn
                if max_diff_idx >= 0:
                    selected_frames.append(remaining_frames[max_diff_idx])
                    remaining_frames.pop(max_diff_idx)
                else:
                    break
            
            # Lưu các frame đã chọn
            for i, (frame_idx, face_image) in enumerate(selected_frames):
                # Tạo tên file
                filename = os.path.join(output_folder, f"segment_{segment_idx:02d}_frame_{i:02d}.jpg")
                
                # Lưu ảnh
                cv2.imwrite(filename, face_image)
                
                # Thêm vào danh sách đã lưu
                saved_frames.append((frame_idx, filename))
                
                # Cập nhật thanh tiến trình
                pbar.update(1)
    
    # Giải phóng tài nguyên
    video.release()
    
    print(f"Successfully saved {len(saved_frames)} diverse face frames from {os.path.basename(video_path)}")
    return len(saved_frames)

def extract_faces_from_all_videos(video_folder, raw_folder, num_segments=12, frames_per_segment=8, target_size=224):
    """
    Xử lý tất cả các video trong thư mục, trích xuất khuôn mặt
    
    Args:
        video_folder: Thư mục chứa các video
        raw_folder: Thư mục đầu ra cho ảnh raw
        num_segments: Số đoạn cần chia cho mỗi video
        frames_per_segment: Số frame trích xuất từ mỗi đoạn
        target_size: Kích thước ảnh đầu ra
    """
    # Đảm bảo thư mục raw tồn tại
    ensure_dir(raw_folder)
    
    # Lấy danh sách video trong thư mục
    videos = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    for video in videos:
        video_path = os.path.join(video_folder, video)
        
        # Lấy tên sinh viên từ tên video (bỏ đuôi file)
        student_id = os.path.splitext(video)[0]
        
        # Tạo thư mục cho sinh viên trong raw
        raw_student_folder = os.path.join(raw_folder, student_id)
        ensure_dir(raw_student_folder)
        
        # Trích xuất frame đa dạng từ video
        extract_diverse_frames_from_video(video_path, raw_student_folder, num_segments=num_segments, frames_per_segment=frames_per_segment, target_size=target_size)
    
    print(f"Đã hoàn thành trích xuất ảnh từ video, lưu trong thư mục {raw_folder}")

if __name__ == "__main__":
    # Đường dẫn đến các thư mục
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # KLTN/
    video_folder = os.path.join(project_dir, "data", "videos")
    raw_folder = os.path.join(project_dir, "data", "raw")
    
    # Cấu hình trích xuất
    num_segments = 12
    frames_per_segment = 8
    target_size = 224
    
    # Tổng số frame
    total_frames = num_segments * frames_per_segment
    
    print(f"Cấu hình trích xuất:")
    print(f"- Chia mỗi video thành {num_segments} đoạn")
    print(f"- Lấy {frames_per_segment} frame đa dạng từ mỗi đoạn")
    print(f"- Tổng số frame tối đa sẽ trích xuất: {total_frames}")
    print(f"- Kích thước ảnh đầu ra: {target_size}x{target_size} pixel")
    
    # Xử lý tất cả các video
    extract_faces_from_all_videos(video_folder, raw_folder, num_segments=num_segments, frames_per_segment=frames_per_segment, target_size=target_size)