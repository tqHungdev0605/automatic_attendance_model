import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
import random
import mediapipe as mp
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import time
from scipy.spatial import distance

def ensure_dir(directory):
    """Đảm bảo thư mục tồn tại, nếu không tạo mới"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def remove_similar_images(folder, similarity_threshold=0.92):
    """
    Loại bỏ các ảnh quá giống nhau trong một thư mục
    Sử dụng kết hợp histogram và perceptual hash để so sánh
    
    Args:
        folder: Thư mục chứa ảnh cần kiểm tra
        similarity_threshold: Ngưỡng giống nhau để loại bỏ ảnh (0.92 = 92%)
    
    Returns:
        Số lượng ảnh đã xóa
    """
    def get_image_histogram(image):
        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Tính histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        # Chuẩn hóa
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    
    def get_perceptual_hash(image, hash_size=8):
        # Resize ảnh thành kích thước nhỏ
        resized = cv2.resize(image, (hash_size + 1, hash_size))
        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Tính diff 
        diff = gray[:, 1:] > gray[:, :-1]
        # Chuyển đổi thành hash
        return diff.flatten()
    
    def compare_images(img1, img2):
        # So sánh histogram
        hist1 = get_image_histogram(img1)
        hist2 = get_image_histogram(img2)
        hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # So sánh perceptual hash
        hash1 = get_perceptual_hash(img1)
        hash2 = get_perceptual_hash(img2)
        hash_dist = np.count_nonzero(hash1 != hash2) / len(hash1)
        hash_sim = 1 - hash_dist
        
        # Kết hợp cả hai thông số (trọng số 0.7-0.3)
        combined_similarity = 0.7 * hist_corr + 0.3 * hash_sim
        
        return combined_similarity
    
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
            print(f"  Chỉ có {len(image_paths)} ảnh, bỏ qua")
            continue
        
        # Đọc ảnh
        images = []
        valid_paths = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                valid_paths.append(img_path)
        
        # So sánh và đánh dấu ảnh cần xóa
        to_remove = set()
        n = len(images)
        
        print(f"  So sánh {n} ảnh...")
        for i in tqdm(range(n), leave=False):
            if valid_paths[i] in to_remove:
                continue
                
            for j in range(i+1, n):
                if valid_paths[j] in to_remove:
                    continue
                    
                # So sánh hai ảnh
                similarity = compare_images(images[i], images[j])
                
                if similarity > similarity_threshold:
                    # Ảnh quá giống nhau, đánh dấu xóa ảnh thứ hai
                    to_remove.add(valid_paths[j])
        
        # Xóa các ảnh trùng lặp
        for img_path in to_remove:
            os.remove(img_path)
        
        print(f"  Đã xóa {len(to_remove)} ảnh trùng lặp từ {len(valid_paths)} ảnh")
        total_removed += len(to_remove)
    
    print(f"Tổng cộng đã xóa {total_removed} ảnh trùng lặp")
    return total_removed

def evaluate_face_image_quality(face_image):
    """
    Đánh giá chất lượng ảnh khuôn mặt
    
    Args:
        face_image: Ảnh khuôn mặt
    
    Returns:
        Điểm chất lượng (cao hơn = tốt hơn)
    """
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    # 1. Đánh giá độ sắc nét bằng Laplacian
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Tính tương phản
    contrast = gray.std()
    
    # 3. Kiểm tra độ sáng
    brightness = np.mean(gray)
    
    # Điểm cho độ sáng (cao nhất khi độ sáng ở mức trung bình)
    brightness_score = 1 - abs((brightness - 128) / 128)
    
    # Kết hợp các thông số (trọng số có thể điều chỉnh)
    quality_score = (0.5 * laplacian_var/100 + 0.3 * contrast/50 + 0.2 * brightness_score)
    
    return quality_score

def get_face_embedding(face_image, face_mesh_model):
    """
    Trích xuất embedding vector từ khuôn mặt sử dụng MediaPipe Face Mesh
    
    Args:
        face_image: Ảnh khuôn mặt
        face_mesh_model: Đối tượng MediaPipe Face Mesh
    
    Returns:
        Embedding vector hoặc None nếu không tìm thấy khuôn mặt
    """
    # Chuyển đổi sang RGB
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Phát hiện các điểm mốc trên khuôn mặt
    results = face_mesh_model.process(face_rgb)
    
    if not results.multi_face_landmarks:
        return None
    
    # Lấy các điểm mốc quan trọng và chuyển thành vector
    landmarks = results.multi_face_landmarks[0].landmark
    
    # Danh sách các điểm mốc quan trọng
    # Bao gồm: mắt, mũi, miệng, cằm, lông mày, đường viền mặt
    key_indices = [
        # Mắt trái
        33, 7, 163, 144, 145, 153, 154, 155, 133, 
        # Mắt phải
        263, 249, 390, 373, 374, 380, 381, 382, 362,
        # Mũi
        1, 2, 3, 4, 5, 6, 
        # Miệng
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        # Đường viền mặt
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152
    ]
    
    # Trích xuất tọa độ từ các điểm mốc quan trọng
    embedding = []
    h, w, _ = face_image.shape
    
    for idx in key_indices:
        if idx < len(landmarks):
            landmark = landmarks[idx]
            # Chuyển đổi tọa độ từ tương đối sang tuyệt đối và chuẩn hóa
            x, y, z = landmark.x, landmark.y, landmark.z
            embedding.extend([x, y, z])
    
    if not embedding:
        return None
    
    return np.array(embedding)

def cluster_face_images(images_folder, embedding_cache_path=None):
    """
    Phân cụm các ảnh khuôn mặt dựa trên embedding
    
    Args:
        images_folder: Thư mục chứa ảnh khuôn mặt
        embedding_cache_path: Đường dẫn để lưu/đọc cache embedding
    
    Returns:
        Dictionary chứa thông tin phân cụm, mỗi cluster là một list các đường dẫn ảnh
    """
    # Khởi tạo MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    )
    
    # Lấy danh sách ảnh
    image_paths = [os.path.join(images_folder, f) for f in os.listdir(images_folder) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Các dictionary để lưu trữ thông tin
    embeddings = {}
    quality_scores = {}
    file_paths = {}
    
    # Kiểm tra cache
    if embedding_cache_path and os.path.exists(embedding_cache_path):
        try:
            print(f"Đọc cache embedding từ {embedding_cache_path}")
            with open(embedding_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                embeddings = cache_data.get('embeddings', {})
                quality_scores = cache_data.get('quality_scores', {})
                file_paths = cache_data.get('file_paths', {})
        except Exception as e:
            print(f"Lỗi khi đọc cache: {e}")
            embeddings = {}
            quality_scores = {}
            file_paths = {}
    
    # Xác định những ảnh cần trích xuất embedding
    images_to_process = []
    for i, img_path in enumerate(image_paths):
        img_id = os.path.basename(img_path)
        if img_id not in embeddings:
            images_to_process.append((i, img_path, img_id))
    
    # Trích xuất embedding cho những ảnh mới
    if images_to_process:
        print(f"Trích xuất embedding cho {len(images_to_process)} ảnh mới...")
        for i, img_path, img_id in tqdm(images_to_process):
            # Đọc ảnh
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Tính điểm chất lượng
            quality_score = evaluate_face_image_quality(img)
            
            # Trích xuất embedding
            embedding = get_face_embedding(img, face_mesh)
            if embedding is not None:
                embeddings[img_id] = embedding
                quality_scores[img_id] = quality_score
                file_paths[img_id] = img_path
    
    # Lưu cache nếu có đường dẫn
    if embedding_cache_path:
        cache_dir = os.path.dirname(embedding_cache_path)
        ensure_dir(cache_dir)
        with open(embedding_cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'quality_scores': quality_scores,
                'file_paths': file_paths
            }, f)
    
    # Chuẩn bị dữ liệu cho clustering
    valid_embeddings = []
    valid_ids = []
    
    for img_id, embedding in embeddings.items():
        valid_embeddings.append(embedding)
        valid_ids.append(img_id)
    
    if not valid_embeddings:
        print("Không có embedding hợp lệ để phân cụm!")
        return {}
    
    # Chuyển thành mảng numpy
    embeddings_array = np.array(valid_embeddings)
    
    # Xác định số lượng cluster phù hợp dựa trên kích thước dữ liệu
    n_samples = len(embeddings_array)
    if n_samples <= 10:
        n_clusters = min(2, n_samples)
    elif n_samples <= 30:
        n_clusters = 3
    elif n_samples <= 60:
        n_clusters = 4
    else:
        n_clusters = 5
    
    print(f"Phân cụm {n_samples} embedding thành {n_clusters} nhóm...")
    
    # Thực hiện clustering với K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings_array)
    
    # Thử với DBSCAN nếu số mẫu đủ lớn
    if n_samples >= 20:
        try:
            # Tính toán epsilon thích hợp (khoảng cách trung bình đến k láng giềng gần nhất)
            from sklearn.neighbors import NearestNeighbors
            k = min(10, n_samples // 2)
            nbrs = NearestNeighbors(n_neighbors=k).fit(embeddings_array)
            distances, indices = nbrs.kneighbors(embeddings_array)
            dist_desc = sorted(distances[:, k-1], reverse=True)
            
            # Tìm "elbow point" - điểm gấp khúc trên đồ thị khoảng cách
            max_diff = 0
            elbow_idx = 0
            for i in range(1, len(dist_desc)-1):
                diff = dist_desc[i-1] - dist_desc[i]
                if diff > max_diff:
                    max_diff = diff
                    elbow_idx = i
            
            epsilon = dist_desc[elbow_idx]
            min_samples = max(3, n_samples // 20)  # Ít nhất 3 mẫu hoặc 5% tổng số mẫu
            
            # Áp dụng DBSCAN
            dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
            dbscan_clusters = dbscan.fit_predict(embeddings_array)
            
            # Kiểm tra số lượng cluster
            unique_clusters = np.unique(dbscan_clusters)
            n_dbscan_clusters = len([c for c in unique_clusters if c != -1])
            
            # Nếu DBSCAN cho kết quả tốt, sử dụng nó
            if n_dbscan_clusters >= 2 and n_dbscan_clusters <= max(n_clusters+2, 7):
                print(f"Sử dụng DBSCAN với {n_dbscan_clusters} cluster")
                clusters = dbscan_clusters
            else:
                print(f"DBSCAN không hiệu quả ({n_dbscan_clusters} clusters), quay lại K-means")
        except Exception as e:
            print(f"Lỗi khi thử DBSCAN: {e}")
    
    # Tạo dictionary lưu ảnh theo cluster
    clustered_images = defaultdict(list)
    for i, cluster_id in enumerate(clusters):
        if cluster_id != -1:  # Bỏ qua noise trong DBSCAN
            img_id = valid_ids[i]
            file_path = file_paths[img_id]
            quality = quality_scores.get(img_id, 0)
            clustered_images[cluster_id].append((file_path, quality))
    
    # Sắp xếp ảnh trong mỗi cluster theo điểm chất lượng
    for cluster_id in clustered_images:
        clustered_images[cluster_id].sort(key=lambda x: x[1], reverse=True)
    
    # Giải phóng tài nguyên
    face_mesh.close()
    
    return {k: [file for file, _ in v] for k, v in clustered_images.items()}

def visualize_clusters(clustered_images, output_dir, max_images_per_cluster=5):
    """Hiển thị một số ảnh từ mỗi cluster để kiểm tra"""
    ensure_dir(output_dir)
    
    for cluster_id, image_paths in clustered_images.items():
        plt.figure(figsize=(15, 3))
        plt.suptitle(f"Cluster {cluster_id} - {len(image_paths)} images")
        
        # Chọn tối đa max_images_per_cluster ảnh để hiển thị
        sample_paths = image_paths[:min(max_images_per_cluster, len(image_paths))]
        
        for i, img_path in enumerate(sample_paths):
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(1, len(sample_paths), i+1)
            plt.imshow(img_rgb)
            plt.title(os.path.basename(img_path), fontsize=8)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"cluster_{cluster_id}.jpg"))
        plt.close()

def split_data_balanced(raw_folder, train_folder, test_folder, val_folder, train_ratio=0.7, test_ratio=0.15):
    """
    Phân chia dữ liệu vào các tập train, test, validation đảm bảo cân bằng
    
    Args:
        raw_folder: Thư mục chứa dữ liệu gốc
        train_folder: Thư mục dữ liệu huấn luyện
        test_folder: Thư mục dữ liệu kiểm thử
        val_folder: Thư mục dữ liệu xác thực
        train_ratio: Tỷ lệ dữ liệu huấn luyện
        test_ratio: Tỷ lệ dữ liệu kiểm thử
    """
    # Đảm bảo các thư mục tồn tại
    ensure_dir(train_folder)
    ensure_dir(test_folder)
    ensure_dir(val_folder)
    
    # Thư mục để lưu kết quả visualize clusters
    clusters_viz_dir = os.path.join(os.path.dirname(train_folder), "cluster_visualization")
    ensure_dir(clusters_viz_dir)
    
    # Thư mục cache cho embedding
    cache_dir = os.path.join(os.path.dirname(train_folder), "embedding_cache")
    ensure_dir(cache_dir)
    
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
        
        # Đường dẫn đến thư mục raw của sinh viên
        raw_student_folder = os.path.join(raw_folder, student_folder)
        
        # Đường dẫn đến file cache embedding
        embedding_cache_path = os.path.join(cache_dir, f"{student_id}_embedding.pkl")
        
        # Phân cụm ảnh
        clustered_images = cluster_face_images(raw_student_folder, embedding_cache_path)
        
        if not clustered_images:
            print(f"Không thể phân cụm ảnh cho sinh viên {student_id}!")
            # Chia ngẫu nhiên nếu không thể phân cụm
            all_images = [os.path.join(raw_student_folder, f) for f in os.listdir(raw_student_folder) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
            random.shuffle(all_images)
            
            # Chia theo tỷ lệ
            num_images = len(all_images)
            num_train = int(num_images * train_ratio)
            num_test = int(num_images * test_ratio)
            
            train_images = all_images[:num_train]
            test_images = all_images[num_train:num_train+num_test]
            val_images = all_images[num_train+num_test:]
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
                # (đã sắp xếp theo chất lượng trong hàm cluster_face_images)
                
                # Tính số lượng ảnh cho mỗi tập từ cluster này
                num_images = len(cluster_images)
                num_train = max(int(num_images * train_ratio), 1)  # Ít nhất 1 ảnh cho train
                num_test = max(int(num_images * test_ratio), 1)    # Ít nhất 1 ảnh cho test
                
                # Đảm bảo có đủ ảnh cho validation
                remaining = num_images - num_train - num_test
                if remaining <= 0:
                    # Nếu không đủ ảnh, điều chỉnh lại
                    if num_images <= 3:
                        # Nếu chỉ có 1-3 ảnh, ưu tiên cho train
                        if num_images == 1:
                            num_train, num_test = 1, 0
                        elif num_images == 2:
                            num_train, num_test = 1, 1
                        else:  # num_images == 3
                            num_train, num_test = 2, 1
                    else:
                        # Nếu có nhiều ảnh hơn, chia theo tỷ lệ 2:1:1
                        num_train = num_images // 2
                        num_test = num_images // 4
                
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

def check_dataset_stats(folder):
    """Hiển thị thống kê về tập dữ liệu"""
    if not os.path.exists(folder):
        print(f"Thư mục {folder} không tồn tại!")
        return
    
    student_folders = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    
    if not student_folders:
        print(f"Không tìm thấy thư mục sinh viên nào trong {folder}!")
        return
    
    counts = []
    for student in student_folders:
        student_dir = os.path.join(folder, student)
        images = [f for f in os.listdir(student_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        counts.append((student, len(images)))
    
    # Sắp xếp theo số lượng ảnh
    counts.sort(key=lambda x: x[1], reverse=True)
    
    # Tính thống kê
    images_count = [c for _, c in counts]
    total_images = sum(images_count)
    avg_images = total_images / len(counts) if counts else 0
    min_images = min(images_count) if images_count else 0
    max_images = max(images_count) if images_count else 0
    
    print(f"\nThống kê thư mục {folder}:")
    print(f"Tổng số sinh viên: {len(counts)}")
    print(f"Tổng số ảnh: {total_images}")
    print(f"Trung bình: {avg_images:.1f} ảnh/sinh viên")
    print(f"Ít nhất: {min_images} ảnh")
    print(f"Nhiều nhất: {max_images} ảnh")
    
    # Hiển thị phân phối
    print("\nPhân phối số lượng ảnh:")
    for student, count in counts:
        bar = "#" * (count * 50 // max_images if max_images > 0 else 0)
        print(f"{student}: {count} {bar}")

def main():
    # Đường dẫn đến các thư mục
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # KLTN/
    raw_folder = os.path.join(project_dir, "data", "raw")
    train_folder = os.path.join(project_dir, "data", "train")
    test_folder = os.path.join(project_dir, "data", "test")
    val_folder = os.path.join(project_dir, "data", "validation")
    
    # 1. Kiểm tra thống kê ban đầu
    print("Kiểm tra thống kê ban đầu của tập dữ liệu raw:")
    check_dataset_stats(raw_folder)
    
    # 2. Xóa ảnh trùng lặp
    print("\nXóa ảnh trùng lặp từ thư mục raw...")
    num_removed = remove_similar_images(raw_folder, similarity_threshold=0.92)
    
    if num_removed > 0:
        print("\nKiểm tra thống kê sau khi xóa ảnh trùng lặp:")
        check_dataset_stats(raw_folder)
    
    # 3. Chia dữ liệu theo phương pháp cân bằng
    print("\nChia dữ liệu vào các tập train, test, validation...")
    split_data_balanced(raw_folder, train_folder, test_folder, val_folder)
    
    # 4. Kiểm tra thống kê cuối cùng
    print("\nKiểm tra thống kê cuối cùng:")
    check_dataset_stats(train_folder)
    check_dataset_stats(test_folder)
    check_dataset_stats(val_folder)
    
    print("\nHoàn thành xử lý và chia dữ liệu!")
    print("Bạn có thể huấn luyện mô hình với dữ liệu đã được chuẩn bị.")

if __name__ == "__main__":
    main()