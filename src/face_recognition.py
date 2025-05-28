import os
import cv2
import numpy as np
import tensorflow as tf
import json
import mediapipe as mp

class FaceRecognition:
   def __init__(self, model_path, class_indices_path, confidence_threshold=0.7):
       self.confidence_threshold = confidence_threshold
       
       # Load TFLite model
       self.interpreter = tf.lite.Interpreter(model_path=model_path)
       self.interpreter.allocate_tensors()
       self.input_details = self.interpreter.get_input_details()
       self.output_details = self.interpreter.get_output_details()
       
       # Load class names
       with open(class_indices_path, 'r') as f:
           class_indices = json.load(f)
       self.idx_to_class = {v: k for k, v in class_indices.items()}
       
       # MediaPipe face detection
       self.mp_face_detection = mp.solutions.face_detection
       self.face_detector = self.mp_face_detection.FaceDetection(
           model_selection=1, min_detection_confidence=0.6
       )
   
   def detect_faces(self, frame):
       rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       results = self.face_detector.process(rgb_frame)
       
       faces = []
       if results.detections:
           h, w = frame.shape[:2]
           for detection in results.detections:
               bbox = detection.location_data.relative_bounding_box
               x = int(bbox.xmin * w)
               y = int(bbox.ymin * h)
               width = int(bbox.width * w)
               height = int(bbox.height * h)
               
               if width > 50 and height > 50:
                   faces.append((x, y, width, height))
       return faces
   
   def preprocess_face(self, face_img):
       face_img = cv2.resize(face_img, (224, 224))
       face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
       face_img = face_img.astype(np.float32) / 255.0
       face_img = np.expand_dims(face_img, axis=0)
       return face_img
   
   def recognize_face(self, face_img):
       processed_img = self.preprocess_face(face_img)
       
       self.interpreter.set_tensor(self.input_details[0]['index'], processed_img)
       self.interpreter.invoke()
       predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
       
       top_idx = np.argmax(predictions)
       confidence = predictions[top_idx]
       
       class_name = self.idx_to_class.get(top_idx, "Unknown")
       if confidence < self.confidence_threshold:
           class_name = "Unknown"
       
       return class_name, confidence
   
   def process_frame(self, frame):
       faces = self.detect_faces(frame)
       results = []
       
       for (x, y, w, h) in faces:
           face_img = frame[y:y+h, x:x+w]
           class_name, confidence = self.recognize_face(face_img)
           results.append((x, y, w, h, class_name, confidence))
           
           # Draw result
           color = (0, 255, 0) if class_name != "Unknown" else (0, 0, 255)
           cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
           
           label = f"{class_name}: {confidence:.2f}"
           cv2.putText(frame, label, (x, y-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
       
       return frame, results
   
   def run_webcam(self, camera_id=0):
       cap = cv2.VideoCapture(camera_id)
       
       while True:
           ret, frame = cap.read()
           if not ret:
               break
           
           frame = cv2.flip(frame, 1)
           result_frame, faces = self.process_frame(frame)
           
           cv2.imshow("Face Recognition", result_frame)
           
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break
       
       cap.release()
       cv2.destroyAllWindows()

def main():
   model_path = "models/model (4).tflite"
   class_indices_path = "models/class_indices.json"
   
   face_system = FaceRecognition(model_path, class_indices_path)
   face_system.run_webcam()

if __name__ == "__main__":
   main()