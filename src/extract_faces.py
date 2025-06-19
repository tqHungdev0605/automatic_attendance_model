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



def extract_diverse_frames_from_video(video_path, output_folder, target_frames=300, target_size=224):
    """Trích xuất 150 ảnh khuôn mặt chất lượng cao nhất từ video"""
    ensure_dir(output_folder)
    
    # Khởi tạo MediaPipe Face Mesh để đánh giá chất lượng
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"❌ Không thể mở video {video_path}")
        return 0
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎯 Mục tiêu: {target_frames} ảnh chất lượng cao từ {total_frames} frames")
    
    # Lấy mẫu frame cách đều từ video (lấy nhiều để chọn lọc)
    sample_indices = np.linspace(0, total_frames-1, min(total_frames, target_frames*3), dtype=int)
    
    face_candidates = []
    
    print("🔍 Phân tích chất lượng khuôn mặt...")
    for frame_idx in tqdm(sample_indices):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()
        if not ret:
            continue
            
        # Phát hiện khuôn mặt
        face_image = detect_and_crop_face(frame, target_size)
        if face_image is None:
            continue
            
        # Đánh giá chất lượng bằng MediaPipe
        quality_score = evaluate_face_quality_mediapipe(face_image, face_mesh)
        
        face_candidates.append((frame_idx, face_image, quality_score))
    
    video.release()
    face_mesh.close()
    
    if not face_candidates:
        print("❌ Không tìm thấy khuôn mặt nào!")
        return 0
    
    # Sắp xếp theo chất lượng và lấy top 150
    face_candidates.sort(key=lambda x: x[2], reverse=True)
    selected_faces = face_candidates[:target_frames]
    
    print(f"💾 Lưu {len(selected_faces)} ảnh chất lượng cao nhất...")
    for i, (frame_idx, face_image, quality) in enumerate(selected_faces):
        filename = os.path.join(output_folder, f"face_{i:03d}_q{quality:.2f}.jpg")
        cv2.imwrite(filename, face_image)
    
    print(f"✅ Đã lưu {len(selected_faces)} ảnh khuôn mặt chất lượng cao")
    return len(selected_faces)

def evaluate_face_quality_mediapipe(face_image, face_mesh):
    """Đánh giá chất lượng khuôn mặt bằng MediaPipe"""
    # Chuyển RGB cho MediaPipe
    rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    # Điểm cơ bản từ độ sắc nét và sáng
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    contrast = gray.std()
    
    # Điểm thưởng nếu phát hiện được face landmarks
    landmark_bonus = 0
    if results.multi_face_landmarks:
        landmark_bonus = 0.5  # Có landmarks = tốt hơn
    
    # Điểm sáng tối ưu (128 = lý tưởng)
    brightness_score = 1 - abs(brightness - 128) / 128
    
    # Tổng điểm
    quality = sharpness/1000 + contrast/100 + brightness_score + landmark_bonus
    return quality

def extract_faces_from_all_videos(video_folder, raw_folder, target_frames=300, target_size=224):
    """Xử lý tất cả video, trích xuất 150 ảnh chất lượng cao nhất"""
    ensure_dir(raw_folder)
    videos = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    print(f"🎯 Tìm thấy {len(videos)} video, mỗi video sẽ tạo {target_frames} ảnh chất lượng cao")
    
    for video in videos:
        video_path = os.path.join(video_folder, video)
        student_id = os.path.splitext(video)[0]
        raw_student_folder = os.path.join(raw_folder, student_id)
        ensure_dir(raw_student_folder)
        
        print(f"\n📹 Xử lý: {video}")
        num_extracted = extract_diverse_frames_from_video(video_path, raw_student_folder, target_frames, target_size)
        print(f"✅ {student_id}: {num_extracted} ảnh")
    
    print(f"\n🎉 Hoàn thành! Tạo ~{target_frames} ảnh chất lượng cao cho {len(videos)} sinh viên")

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_folder = os.path.join(project_dir, "data", "videos")
    raw_folder = os.path.join(project_dir, "data", "raw")
    
    print("🎯 TRÍCH XUẤT 150 ẢNH CHẤT LƯỢNG CAO/VIDEO")
    print("=" * 50)
    
    if not os.path.exists(video_folder):
        print(f"❌ Không tìm thấy thư mục: {video_folder}")
    else:
        extract_faces_from_all_videos(video_folder, raw_folder, target_frames=300)