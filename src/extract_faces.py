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
        
        # Xáo trộn danh ssách ảnh
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

def classify_face_orientation(face_image, face_mesh):
    """
    Phân loại góc nhìn của khuôn mặt với ngưỡng cải tiến
    
    Args:
        face_image: Ảnh khuôn mặt đã cắt
        face_mesh: Đối tượng FaceMesh của MediaPipe
        
    Returns:
        Nhãn góc nhìn: "front", "left", "right", "up", "down" hoặc "other"
    """
    # Chuyển đổi sang RGB
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Phát hiện các điểm mốc trên khuôn mặt
    results = face_mesh.process(face_rgb)
    
    if not results.multi_face_landmarks:
        return "other"
    
    # Lấy kích thước ảnh
    h, w, _ = face_image.shape
    
    # Dùng set cố định các điểm mốc để đảm bảo ổn định
    # MediaPipe Face Mesh có 468 điểm mốc
    # Điểm mốc quan trọng:
    # - Mũi: điểm 1
    # - Mắt trái: điểm 33, 133
    # - Mắt phải: điểm 263, 362
    # - Miệng: điểm 61, 291
    # - Cằm: điểm 199
    # - Trán: điểm 10
    try:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Trích xuất các điểm mốc quan trọng
        nose_tip = (landmarks[1].x, landmarks[1].y, landmarks[1].z)
        left_eye = (landmarks[33].x, landmarks[33].y, landmarks[33].z) 
        right_eye = (landmarks[263].x, landmarks[263].y, landmarks[263].z)
        forehead = (landmarks[10].x, landmarks[10].y, landmarks[10].z)
        chin = (landmarks[199].x, landmarks[199].y, landmarks[199].z)
        left_mouth = (landmarks[61].x, landmarks[61].y, landmarks[61].z)
        right_mouth = (landmarks[291].x, landmarks[291].y, landmarks[291].z)
        
        # Tính tâm của mặt
        face_center_x = (left_eye[0] + right_eye[0]) / 2
        face_center_y = (forehead[1] + chin[1]) / 2
        
        # Tính khoảng cách giữa hai mắt (để chuẩn hóa)
        eye_distance = abs(right_eye[0] - left_eye[0])
        
        # 1. Đánh giá góc nghiêng trái/phải (yaw)
        # Tính độ lệch của mũi so với tâm mặt, chuẩn hóa theo khoảng cách mắt
        nose_offset_x = (nose_tip[0] - face_center_x) / eye_distance
        
        # 2. Đánh giá góc ngửa/cúi (pitch)
        # Tính tỷ lệ khoảng cách từ mũi đến cằm so với mũi đến trán
        vertical_ratio = abs(nose_tip[1] - chin[1]) / abs(nose_tip[1] - forehead[1])
        
        # Phân loại dựa trên các thông số đã tính
        # Góc nghiêng trái/phải
        if abs(nose_offset_x) < 0.1:  # Mũi gần với tâm mặt
            # Góc ngửa/cúi
            if vertical_ratio < 0.8:  # Khoảng cách mũi-cằm ngắn hơn mũi-trán -> ngửa lên
                return "up"
            elif vertical_ratio > 1.2:  # Khoảng cách mũi-cằm dài hơn mũi-trán -> cúi xuống
                return "down"
            else:
                return "front"
        elif nose_offset_x < -0.1:  # Mũi lệch về bên trái so với tâm
            return "left"
        else:  # Mũi lệch về bên phải so với tâm
            return "right"
            
    except (IndexError, AttributeError) as e:
        print(f"Lỗi khi trích xuất điểm mốc: {e}")
        return "other"

def split_data_stratified(raw_folder, train_folder, test_folder, val_folder, train_ratio=0.7, test_ratio=0.15):
    """
    Chia dữ liệu vào các thư mục train, test, validation với sự cân bằng theo góc nhìn
    
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
    
    # Khởi tạo MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5)
    
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
        
        # Phân loại ảnh theo góc nhìn
        orientation_groups = {
            "front": [],
            "left": [],
            "right": [],
            "up": [],
            "down": [],
            "other": []
        }
        
        print(f"Phân loại góc nhìn cho sinh viên {student_id}...")
        for img_file in tqdm(images):
            img_path = os.path.join(raw_student_folder, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Phân loại góc nhìn với phương pháp cải tiến
            orientation = classify_face_orientation(img, face_mesh)
            orientation_groups[orientation].append(img_file)
        
        # In thống kê
        for orientation, imgs in orientation_groups.items():
            print(f"  - {orientation}: {len(imgs)} ảnh")
        
        # Chia mỗi nhóm góc nhìn vào train, test, val theo tỷ lệ
        train_images = []
        test_images = []
        val_images = []
        
        for orientation, group_images in orientation_groups.items():
            # Xáo trộn ngẫu nhiên trong mỗi nhóm
            random.shuffle(group_images)
            
            # Tính số lượng ảnh cho mỗi tập từ nhóm này
            num_images = len(group_images)
            num_train = int(num_images * train_ratio)
            num_test = int(num_images * test_ratio)
            
            # Chia thành các tập
            train_images.extend(group_images[:num_train])
            test_images.extend(group_images[num_train:num_train+num_test])
            val_images.extend(group_images[num_train+num_test:])
        
        # Sao chép ảnh vào các thư mục tương ứng
        for img in train_images:
            shutil.copy2(os.path.join(raw_student_folder, img), os.path.join(train_student_folder, img))
        
        for img in test_images:
            shutil.copy2(os.path.join(raw_student_folder, img), os.path.join(test_student_folder, img))
        
        for img in val_images:
            shutil.copy2(os.path.join(raw_student_folder, img), os.path.join(val_student_folder, img))
        
        print(f"Sinh viên {student_id}: {len(train_images)} training, {len(test_images)} testing, {len(val_images)} validation images")
    
    # Giải phóng tài nguyên
    face_mesh.close()

def process_all_videos(video_folder, raw_folder, train_folder, test_folder, val_folder, frames_per_video=100):
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
    
    # Thay đổi từ split_data sang split_data_stratified
    split_data_stratified(raw_folder, train_folder, test_folder, val_folder)

if __name__ == "__main__":
    # Đường dẫn đến các thư mục
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # KLTN/
    video_folder = os.path.join(project_dir, "data", "videos")
    raw_folder = os.path.join(project_dir, "data", "raw")
    train_folder = os.path.join(project_dir, "data", "train")
    test_folder = os.path.join(project_dir, "data", "test")
    val_folder = os.path.join(project_dir, "data", "validation")
    
    # Cấu hình
    frames_per_video = 100  # Số lượng frame cần trích xuất từ mỗi video
    image_size = 224  # Kích thước ảnh đầu ra (224x224 là kích thước phổ biến cho nhiều mô hình CNN)
    
    print(f"Trích xuất {frames_per_video} ảnh từ mỗi video, kích thước {image_size}x{image_size} pixel")
    
    # Xử lý tất cả các video
    process_all_videos(video_folder, raw_folder, train_folder, test_folder, val_folder, frames_per_video)