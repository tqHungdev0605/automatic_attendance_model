import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
import random
import mediapipe as mp
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import defaultdict

def ensure_dir(directory):
    """Đảm bảo thư mục tồn tại, nếu không tạo mới"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_face_embedding(face_image, face_model):
    """
    Trích xuất embedding vector từ khuôn mặt sử dụng mô hình face embedding
    Sử dụng MediaPipe Face Mesh để lấy các điểm đặc trưng làm embedding đơn giản
    """
    # Chuyển đổi sang RGB
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Phát hiện các điểm mốc trên khuôn mặt
    results = face_model.process(face_rgb)
    
    if not results.multi_face_landmarks:
        return None
    
    # Lấy các điểm mốc quan trọng và chuyển thành vector
    landmarks = results.multi_face_landmarks[0].landmark
    
    # Trích xuất tọa độ của 20 điểm mốc quan trọng
    key_points = [0, 4, 8, 33, 61, 133, 155, 234, 263, 273, 291, 323, 356, 362, 385, 387, 397, 454, 460, 467]
    embedding = []
    
    for idx in key_points:
        if idx < len(landmarks):
            point = landmarks[idx]
            embedding.extend([point.x, point.y, point.z])
    
    return np.array(embedding)

def cluster_face_images(image_paths, face_model, n_clusters=4):
    """
    Phân cụm các ảnh khuôn mặt thành các nhóm dựa trên embedding
    Giúp nhóm các ảnh tương tự nhau (cùng góc nhìn, biểu cảm, v.v.)
    """
    # Tạo danh sách lưu embedding và đường dẫn
    embeddings = []
    valid_paths = []
    
    # Trích xuất embedding cho mỗi ảnh
    for img_path in tqdm(image_paths, desc="Trích xuất embedding"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        embedding = get_face_embedding(img, face_model)
        if embedding is not None:
            embeddings.append(embedding)
            valid_paths.append(img_path)
    
    if len(embeddings) == 0:
        return None, None
    
    # Chuyển thành mảng numpy
    embeddings_array = np.array(embeddings)
    
    # Cluster các embedding
    kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42)
    clusters = kmeans.fit_predict(embeddings_array)
    
    # Tạo dict lưu ảnh theo cluster
    clustered_images = defaultdict(list)
    for i, cluster_id in enumerate(clusters):
        clustered_images[cluster_id].append(valid_paths[i])
    
    return clustered_images, kmeans

def visualize_clusters(clustered_images, output_dir, max_images_per_cluster=5):
    """Hiển thị một số ảnh từ mỗi cluster để kiểm tra"""
    ensure_dir(output_dir)
    
    for cluster_id, image_paths in clustered_images.items():
        plt.figure(figsize=(15, 3))
        plt.suptitle(f"Cluster {cluster_id} - {len(image_paths)} images")
        
        # Chọn tối đa max_images_per_cluster ảnh để hiển thị
        sample_paths = random.sample(image_paths, min(max_images_per_cluster, len(image_paths)))
        
        for i, img_path in enumerate(sample_paths):
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(1, len(sample_paths), i+1)
            plt.imshow(img_rgb)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"cluster_{cluster_id}.jpg"))
        plt.close()

def improved_data_split(raw_folder, train_folder, test_folder, val_folder, train_ratio=0.7, test_ratio=0.15):
    """
    Cải thiện phương pháp chia dữ liệu đảm bảo tính đại diện
    Sử dụng clustering để phân cụm các ảnh tương tự, sau đó chia dữ liệu dựa trên clusters
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
    
    # Thư mục để lưu kết quả visualize clusters
    clusters_viz_dir = os.path.join(os.path.dirname(train_folder), "cluster_visualization")
    ensure_dir(clusters_viz_dir)
    
    # Lấy danh sách thư mục con trong raw_folder (mỗi thư mục là một student)
    student_folders = [d for d in os.listdir(raw_folder) if os.path.isdir(os.path.join(raw_folder, d))]
    
    for student_folder in student_folders:
        # Tạo thư mục tương ứng trong train, test, val
        student_id = student_folder
        print(f"\nXử lý sinh viên: {student_id}")
        
        train_student_folder = os.path.join(train_folder, student_id)
        test_student_folder = os.path.join(test_folder, student_id)
        val_student_folder = os.path.join(val_folder, student_id)
        
        ensure_dir(train_student_folder)
        ensure_dir(test_student_folder)
        ensure_dir(val_student_folder)
        
        # Lấy danh sách hình ảnh trong thư mục raw của sinh viên
        raw_student_folder = os.path.join(raw_folder, student_folder)
        images = [os.path.join(raw_student_folder, f) for f in os.listdir(raw_student_folder) 
                 if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) == 0:
            print(f"Không tìm thấy ảnh nào cho sinh viên {student_id}!")
            continue
        
        # Số lượng clusters dựa trên số lượng ảnh
        n_clusters = min(4, len(images) // 10)  # Ít nhất 10 ảnh/cluster
        if n_clusters < 2:
            n_clusters = 2
        
        # Phân cụm ảnh
        print(f"Phân cụm ảnh khuôn mặt của sinh viên {student_id} thành {n_clusters} cụm...")
        clustered_images, kmeans = cluster_face_images(images, face_mesh, n_clusters)
        
        if clustered_images is None:
            print(f"Không thể trích xuất embedding cho sinh viên {student_id}!")
            # Chia ngẫu nhiên nếu không thể phân cụm
            random.shuffle(images)
            num_train = int(len(images) * train_ratio)
            num_test = int(len(images) * test_ratio)
            
            train_images = images[:num_train]
            test_images = images[num_train:num_train+num_test]
            val_images = images[num_train+num_test:]
        else:
            # Visualize các cluster để kiểm tra
            student_viz_dir = os.path.join(clusters_viz_dir, student_id)
            visualize_clusters(clustered_images, student_viz_dir)
            
            # Chia dữ liệu từ mỗi cluster theo tỷ lệ
            train_images = []
            test_images = []
            val_images = []
            
            for cluster_id, cluster_images in clustered_images.items():
                # Xáo trộn ngẫu nhiên trong mỗi cluster
                random.shuffle(cluster_images)
                
                # Tính số lượng ảnh cho mỗi tập từ cluster này
                num_images = len(cluster_images)
                num_train = int(num_images * train_ratio)
                num_test = int(num_images * test_ratio)
                
                # Chia thành các tập
                train_images.extend(cluster_images[:num_train])
                test_images.extend(cluster_images[num_train:num_train+num_test])
                val_images.extend(cluster_images[num_train+num_test:])
        
        # Sao chép ảnh vào các thư mục tương ứng
        for src_path in train_images:
            dest_path = os.path.join(train_student_folder, os.path.basename(src_path))
            shutil.copy2(src_path, dest_path)
        
        for src_path in test_images:
            dest_path = os.path.join(test_student_folder, os.path.basename(src_path))
            shutil.copy2(src_path, dest_path)
        
        for src_path in val_images:
            dest_path = os.path.join(val_student_folder, os.path.basename(src_path))
            shutil.copy2(src_path, dest_path)
        
        print(f"Sinh viên {student_id}: {len(train_images)} training, {len(test_images)} testing, {len(val_images)} validation images")
    
    # Giải phóng tài nguyên
    face_mesh.close()
    
    print("\nHoàn thành việc chia dữ liệu!")
    print(f"Bạn có thể xem kết quả phân cụm tại: {clusters_viz_dir}")

def remove_similar_images(folder, similarity_threshold=0.95):
    """
    Loại bỏ các ảnh quá giống nhau trong một thư mục
    Sử dụng histogram similarity để so sánh
    """
    def get_image_histogram(image):
        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Tính histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        # Chuẩn hóa
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    
    def compare_histograms(hist1, hist2):
        # So sánh hai histogram sử dụng correlation
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # Lấy danh sách subfolders (sinh viên)
    student_folders = [os.path.join(folder, d) for d in os.listdir(folder) 
                       if os.path.isdir(os.path.join(folder, d))]
    
    total_removed = 0
    
    for student_folder in student_folders:
        student_id = os.path.basename(student_folder)
        print(f"Kiểm tra ảnh trùng lặp cho sinh viên {student_id}...")
        
        # Lấy danh sách ảnh
        image_paths = [os.path.join(student_folder, f) for f in os.listdir(student_folder) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_paths) <= 1:
            continue
        
        # Đọc ảnh và tính histogram
        images_with_hist = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is not None:
                hist = get_image_histogram(img)
                images_with_hist.append((img_path, hist))
        
        # So sánh và đánh dấu ảnh cần xóa
        to_remove = set()
        n = len(images_with_hist)
        
        for i in range(n):
            if images_with_hist[i][0] in to_remove:
                continue
                
            for j in range(i+1, n):
                if images_with_hist[j][0] in to_remove:
                    continue
                    
                # So sánh histogram
                similarity = compare_histograms(images_with_hist[i][1], images_with_hist[j][1])
                
                if similarity > similarity_threshold:
                    # Ảnh quá giống nhau, đánh dấu xóa ảnh thứ hai
                    to_remove.add(images_with_hist[j][0])
        
        # Xóa các ảnh trùng lặp
        for img_path in to_remove:
            os.remove(img_path)
        
        print(f"  Đã xóa {len(to_remove)} ảnh trùng lặp từ {len(image_paths)} ảnh")
        total_removed += len(to_remove)
    
    print(f"Tổng cộng đã xóa {total_removed} ảnh trùng lặp")

def main():
    # Đường dẫn đến các thư mục
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # KLTN/
    raw_folder = os.path.join(project_dir, "data", "raw")
    train_folder = os.path.join(project_dir, "data", "train")
    test_folder = os.path.join(project_dir, "data", "test")
    val_folder = os.path.join(project_dir, "data", "validation")
    
    # 1. Xóa ảnh trùng lặp từ thư mục raw (tùy chọn)
    remove_similar_images(raw_folder, similarity_threshold=0.98)
    
    # 2. Chia dữ liệu theo phương pháp cải tiến
    improved_data_split(raw_folder, train_folder, test_folder, val_folder)

if __name__ == "__main__":
    main()