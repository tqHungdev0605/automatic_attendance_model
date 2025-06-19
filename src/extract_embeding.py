import cv2
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine
import time
import os
import mediapipe as mp

class FaceRecognition:
    def __init__(self, model_path, threshold=0.6):
        self.threshold = threshold
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]
    
    def preprocess_image(self, image):
        if image.shape[0] != self.input_shape[0] or image.shape[1] != self.input_shape[1]:
            image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        return np.expand_dims(image.astype(np.float32) / 255.0, axis=0)
    
    def extract_embedding(self, image):
        preprocessed_image = self.preprocess_image(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_image)
        self.interpreter.invoke()
        embedding = self.interpreter.get_tensor(self.output_details[0]['index'])
        return embedding[0] / np.linalg.norm(embedding)
    
    def compare_faces(self, image1, image2):
        embedding1 = self.extract_embedding(image1)
        embedding2 = self.extract_embedding(image2)
        distance = cosine(embedding1, embedding2)
        similarity = 1 - distance
        return {
            'similarity': similarity,
            'is_same_person': similarity >= self.threshold,
            'embedding1': embedding1,
            'embedding2': embedding2
        }
    
    def detect_face(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            width = min(int(bbox.width * w), w - x)
            height = min(int(bbox.height * h), h - y)
            face_img = image[y:y+height, x:x+width]
            if face_img.size == 0 or width <= 0 or height <= 0:
                return False, None, (0, 0, 0, 0)
            return True, face_img, (x, y, width, height)
        return False, None, (0, 0, 0, 0)
    
    def run_camera_demo(self):
        cap = cv2.VideoCapture(0)
        stored_embedding = None
        stored_face_image = None
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            display_frame = frame.copy()
            face_detected, face_image, (x, y, w, h) = self.detect_face(frame)
            
            if face_detected:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                current_embedding = self.extract_embedding(face_image)
                cv2.putText(display_frame, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if stored_embedding is not None:
                    distance = cosine(current_embedding, stored_embedding)
                    similarity = 1 - distance
                    match_text = f"Match: {similarity:.2f}"
                    color = (0, 255, 0) if similarity >= self.threshold else (0, 0, 255)
                    cv2.putText(display_frame, match_text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if face_image is not None and display_frame.shape[0] >= 110 and display_frame.shape[1] >= 110:
                    display_frame[10:110, 10:110] = cv2.resize(face_image, (100, 100))
            
            if stored_face_image is not None and display_frame.shape[0] >= 110 and display_frame.shape[1] >= 220:
                display_frame[10:110, 120:220] = cv2.resize(stored_face_image, (100, 100))
                cv2.putText(display_frame, "Stored Face", (120, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            cv2.putText(display_frame, "s: store face, c: clear face, q: quit", (10, display_frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Face Recognition Demo', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('s') and face_detected:
                stored_embedding = current_embedding
                stored_face_image = face_image.copy()
            elif key == ord('c'):
                stored_embedding = None
                stored_face_image = None
        
        self.face_detection.close()
        cap.release()
        cv2.destroyAllWindows()
    
    def compare_image_files(self, image_path1, image_path2):
        img1, img2 = cv2.imread(image_path1), cv2.imread(image_path2)
        if img1 is None or img2 is None: return None
        
        face1_detected, face1, (x1, y1, w1, h1) = self.detect_face(img1)
        if not face1_detected: return None
        
        face2_detected, face2, (x2, y2, w2, h2) = self.detect_face(img2)
        if not face2_detected: return None
        
        result = self.compare_faces(face1, face2)
        
        img1_with_rect, img2_with_rect = img1.copy(), img2.copy()
        cv2.rectangle(img1_with_rect, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)
        cv2.rectangle(img2_with_rect, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2)
        
        max_height = max(img1_with_rect.shape[0], img2_with_rect.shape[0])
        img1_resized = cv2.resize(img1_with_rect, (int(img1_with_rect.shape[1] * max_height / img1_with_rect.shape[0]), max_height))
        img2_resized = cv2.resize(img2_with_rect, (int(img2_with_rect.shape[1] * max_height / img2_with_rect.shape[0]), max_height))
        
        face1_display = cv2.resize(face1, (224, 224))
        face2_display = cv2.resize(face2, (224, 224))
        combined_faces = np.hstack((face1_display, face2_display))
        
        result_img = np.ones((100, combined_faces.shape[1], 3), dtype=np.uint8) * 255
        cv2.putText(result_img, f"Similarity: {result['similarity']:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        match_text = "Same Person" if result['is_same_person'] else "Different Person"
        match_color = (0, 128, 0) if result['is_same_person'] else (0, 0, 255)
        cv2.putText(result_img, match_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2)
        
        combined_result = np.vstack((combined_faces, result_img))
        # cv2.imshow("Image 1", img1_resized)
        # cv2.imshow("Image 2", img2_resized)
        cv2.imshow("Face Comparison Result", combined_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return result


if __name__ == "__main__":
    model_path = "models/face_recognition_model_compatible (1).tflite"
    face_recognizer = FaceRecognition(model_path, threshold=0.6)
    
    choice = input("Chọn chế độ (1: So sánh hai ảnh, 2: Demo với camera): ")
    
    if choice == "1":
        image_path1 = input("Nhập đường dẫn ảnh thứ nhất: ")
        image_path2 = input("Nhập đường dẫn ảnh thứ hai: ")
        result = face_recognizer.compare_image_files(image_path1, image_path2)
        if result:
            print(f"Similarity: {result['similarity']:.4f}, Same person: {result['is_same_person']}")
    elif choice == "2":
        face_recognizer.run_camera_demo()