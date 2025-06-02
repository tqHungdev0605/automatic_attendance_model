import os
import shutil
import random
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import mediapipe as mp

def ensure_dir(directory):
    """Đảm bảo thư mục tồn tại"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_face_pose_features(image):
    """
    Trích xuất đặc trưng góc nhìn khuôn mặt từ ảnh
    Trả về vector đặc trưng 6D: [yaw, pitch, roll, nose_x, nose_y, face_width]
    """
    # Khởi tạo MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    # Đọc ảnh
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image.copy()
    
    if img is None:
        return None
    
    # Chuyển BGR sang RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Phát hiện landmarks
    results = face_mesh.process(rgb_img)
    
    if not results.multi_face_landmarks:
        face_mesh.close()
        return None
    
    landmarks = results.multi_face_landmarks[0]
    h, w = img.shape[:2]
    
    # Chuyển landmarks sang tọa độ pixel
    points_3d = []
    points_2d = []
    
    # Các điểm quan trọng để tính góc nhìn
    key_points = [
        1,    # Nose tip
        33,   # Left eye outer corner
        263,  # Right eye outer corner
        61,   # Left mouth corner
        291,  # Right mouth corner
        199,  # Lower lip center
    ]
    
    for idx in key_points:
        landmark = landmarks.landmark[idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        z = landmark.z * w  # Ước lượng độ sâu
        
        points_2d.append([x, y])
        points_3d.append([x, y, z])
    
    # Chuyển sang numpy array
    points_2d = np.array(points_2d, dtype=np.float64)
    points_3d = np.array(points_3d, dtype=np.float64)
    
    # Tham số camera (ước lượng)
    focal_length = w
    center = (w // 2, h // 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Hệ số biến dạng (giả định không có)
    dist_coeffs = np.zeros((4, 1))
    
    # Mô hình 3D khuôn mặt chuẩn
    model_points = np.array([
        [0.0, 0.0, 0.0],          # Nose tip
        [-30.0, -65.0, -5.0],     # Left eye left corner
        [30.0, -65.0, -5.0],      # Right eye right corner
        [-20.0, 50.0, -30.0],     # Left Mouth corner
        [20.0, 50.0, -30.0],      # Right mouth corner
        [0.0, 110.0, -30.0],      # Lower lip center
    ], dtype=np.float64)
    
    # Giải PnP để tìm góc quay
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, points_2d, camera_matrix, dist_coeffs
    )
    
    if not success:
        face_mesh.close()
        return None
    
    # Chuyển rotation vector sang góc Euler
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Tính góc Euler (yaw, pitch, roll)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # roll
        y = np.arctan2(-rotation_matrix[2, 0], sy)                   # pitch
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) # yaw
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]) # roll
        y = np.arctan2(-rotation_matrix[2, 0], sy)                   # pitch
        z = 0                                                        # yaw
    
    # Chuyển từ radian sang độ
    yaw = np.degrees(z)
    pitch = np.degrees(y)
    roll = np.degrees(x)
    
    # Thêm các đặc trưng khác
    nose_tip = landmarks.landmark[1]
    nose_x = nose_tip.x  # Vị trí mũi theo trục X (trái/phải)
    nose_y = nose_tip.y  # Vị trí mũi theo trục Y (trên/dưới)
    
    # Tính độ rộng khuôn mặt
    left_face = landmarks.landmark[234]
    right_face = landmarks.landmark[454]
    face_width = abs(right_face.x - left_face.x)
    
    face_mesh.close()
    
    # Vector đặc trưng 6D
    features = np.array([yaw, pitch, roll, nose_x, nose_y, face_width])
    return features

def cluster_images_by_pose(image_paths, n_clusters=3):
    """
    Phân cụm ảnh theo góc nhìn sử dụng K-means
    """
    print(f"🔍 Phân tích góc nhìn cho {len(image_paths)} ảnh...")
    
    features_list = []
    valid_images = []
    
    for img_path in tqdm(image_paths, desc="Trích xuất đặc trưng"):
        features = extract_face_pose_features(img_path)
        if features is not None:
            features_list.append(features)
            valid_images.append(img_path)
    
    if len(features_list) < n_clusters:
        print(f"⚠️ Chỉ có {len(features_list)} ảnh hợp lệ, không đủ để phân {n_clusters} cụm")
        return {0: valid_images}
    
    # Chuẩn hóa đặc trưng
    features_array = np.array(features_list)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_array)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_normalized)
    
    # Nhóm ảnh theo cụm
    clusters = {}
    for i in range(n_clusters):
        clusters[i] = []
    
    for img_path, label in zip(valid_images, cluster_labels):
        clusters[label].append(img_path)
    
    # In thống kê cụm
    print(f"📊 Phân cụm hoàn thành:")
    for i, imgs in clusters.items():
        if len(imgs) > 0:
            # Tính góc trung bình của cụm
            cluster_features = [features_list[j] for j, img in enumerate(valid_images) if img in imgs]
            avg_yaw = np.mean([f[0] for f in cluster_features])
            avg_pitch = np.mean([f[1] for f in cluster_features])
            avg_roll = np.mean([f[2] for f in cluster_features])
            print(f"   Cụm {i}: {len(imgs)} ảnh (Yaw: {avg_yaw:.1f}°, Pitch: {avg_pitch:.1f}°, Roll: {avg_roll:.1f}°)")
    
    return clusters

def split_data_by_pose(raw_folder, train_folder, test_folder, val_folder, n_clusters=3):
    """
    Chia dữ liệu dựa trên góc nhìn với K-means clustering
    """
    ensure_dir(train_folder)
    ensure_dir(test_folder) 
    ensure_dir(val_folder)
    
    students = [d for d in os.listdir(raw_folder) if os.path.isdir(os.path.join(raw_folder, d))]
    
    print(f"🎯 Chia {len(students)} sinh viên dựa trên góc nhìn (K-means với {n_clusters} cụm)")
    
    total_stats = {'train': 0, 'test': 0, 'val': 0}
    
    for student in tqdm(students, desc="Xử lý sinh viên"):
        # Tạo thư mục cho sinh viên
        ensure_dir(os.path.join(train_folder, student))
        ensure_dir(os.path.join(test_folder, student))
        ensure_dir(os.path.join(val_folder, student))
        
        # Lấy tất cả ảnh
        student_folder = os.path.join(raw_folder, student)
        images = [f for f in os.listdir(student_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            continue
        
        # Đường dẫn đầy đủ của ảnh
        image_paths = [os.path.join(student_folder, img) for img in images]
        
        # Phân cụm theo góc nhìn
        clusters = cluster_images_by_pose(image_paths, n_clusters)
        
        # Chia mỗi cụm theo tỷ lệ 70/15/15
        for cluster_id, cluster_images in clusters.items():
            if not cluster_images:
                continue
                
            random.shuffle(cluster_images)
            n = len(cluster_images)
            
            train_n = int(n * 0.7)
            test_n = int(n * 0.15)
            
            train_imgs = cluster_images[:train_n]
            test_imgs = cluster_images[train_n:train_n+test_n]
            val_imgs = cluster_images[train_n+test_n:]
            
            # Copy ảnh với tên mới (bao gồm cluster_id)
            for img_path in train_imgs:
                img_name = os.path.basename(img_path)
                name, ext = os.path.splitext(img_name)
                new_name = f"{name}_c{cluster_id}{ext}"
                shutil.copy2(img_path, os.path.join(train_folder, student, new_name))
            
            for img_path in test_imgs:
                img_name = os.path.basename(img_path)
                name, ext = os.path.splitext(img_name)
                new_name = f"{name}_c{cluster_id}{ext}"
                shutil.copy2(img_path, os.path.join(test_folder, student, new_name))
            
            for img_path in val_imgs:
                img_name = os.path.basename(img_path)
                name, ext = os.path.splitext(img_name)
                new_name = f"{name}_c{cluster_id}{ext}"
                shutil.copy2(img_path, os.path.join(val_folder, student, new_name))
            
            total_stats['train'] += len(train_imgs)
            total_stats['test'] += len(test_imgs)
            total_stats['val'] += len(val_imgs)
    
    print(f"✅ Hoàn thành chia dữ liệu dựa trên góc nhìn!")
    print(f"📊 Train: {total_stats['train']} | Test: {total_stats['test']} | Val: {total_stats['val']}")

# Hàm chia dữ liệu truyền thống (giữ lại để backup)
def split_data_traditional(raw_folder, train_folder, test_folder, val_folder):
    """Chia dữ liệu ngẫu nhiên theo tỷ lệ 70/15/15 (phiên bản gốc)"""
    
    ensure_dir(train_folder)
    ensure_dir(test_folder) 
    ensure_dir(val_folder)
    
    students = [d for d in os.listdir(raw_folder) if os.path.isdir(os.path.join(raw_folder, d))]
    
    print(f"🎯 Chia {len(students)} sinh viên theo tỷ lệ 70/15/15 (ngẫu nhiên)")
    
    total_stats = {'train': 0, 'test': 0, 'val': 0}
    
    for student in tqdm(students, desc="Chia dữ liệu"):
        # Tạo thư mục cho sinh viên
        ensure_dir(os.path.join(train_folder, student))
        ensure_dir(os.path.join(test_folder, student))
        ensure_dir(os.path.join(val_folder, student))
        
        # Lấy tất cả ảnh
        student_folder = os.path.join(raw_folder, student)
        images = [f for f in os.listdir(student_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            continue
            
        # Xáo trộn và chia
        random.shuffle(images)
        n = len(images)
        
        train_n = int(n * 0.7)
        test_n = int(n * 0.15)
        
        train_imgs = images[:train_n]
        test_imgs = images[train_n:train_n+test_n]
        val_imgs = images[train_n+test_n:]
        
        # Copy ảnh
        for img in train_imgs:
            shutil.copy2(os.path.join(student_folder, img), 
                        os.path.join(train_folder, student, img))
        
        for img in test_imgs:
            shutil.copy2(os.path.join(student_folder, img), 
                        os.path.join(test_folder, student, img))
        
        for img in val_imgs:
            shutil.copy2(os.path.join(student_folder, img), 
                        os.path.join(val_folder, student, img))
        
        total_stats['train'] += len(train_imgs)
        total_stats['test'] += len(test_imgs)
        total_stats['val'] += len(val_imgs)
    
    print(f"✅ Hoàn thành!")
    print(f"📊 Train: {total_stats['train']} | Test: {total_stats['test']} | Val: {total_stats['val']}")

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_folder = os.path.join(project_dir, "data", "raw")
    train_folder = os.path.join(project_dir, "data", "train")
    test_folder = os.path.join(project_dir, "data", "test")
    val_folder = os.path.join(project_dir, "data", "validation")
    
    random.seed(42)  # Để kết quả lặp lại được
    
    print("🚀 CHỌN PHƯƠNG PHÁP CHIA DỮ LIỆU:")
    print("1. Chia dựa trên góc nhìn (K-means clustering) - Khuyến nghị")
    print("2. Chia ngẫu nhiên (phương pháp truyền thống)")
    
    choice = input("Nhập lựa chọn (1 hoặc 2): ").strip()
    
    if choice == "1":
        n_clusters = int(input("Số cụm góc nhìn (khuyến nghị 3-5): ").strip() or "3")
        split_data_by_pose(raw_folder, train_folder, test_folder, val_folder, n_clusters)
    else:
        split_data_traditional(raw_folder, train_folder, test_folder, val_folder)