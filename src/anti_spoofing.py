import random
import time
import cv2
import mediapipe as mp
import numpy as np


class AntiSpoofing:
    def __init__(self):
        """Khoi tao module chong gia mao khuon mat dua tren thu thach ngau nhien"""
        # Khoi tao MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Cac thong so
        self.is_live = False
        self.frame_count = 0
        self.initialization_frames = 10
        self.initialization_done = False
        
        # Luu tru landmark
        self.initial_landmarks = None
        self.blink_history = []
        self.head_pose_history = []

        # Cac diem landmark cho mat
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # Cac diem landmark cho xoay dau
        self.NOSE_TIP = 4
        self.LEFT_FACE = 234
        self.RIGHT_FACE = 454

        # Tao thu thach ngau nhien
        self.challenges = [
            ("Nhay mat 2 lan", "Quay trai"),
            ("Nhay mat 2 lan", "Quay phai"),
            ("Nhay mat 3 lan", "Quay trai"),
            ("Nhay mat 3 lan", "Quay phai"),
            ("Quay trai", "Nhay mat 2 lan"),
            ("Quay phai", "Nhay mat 2 lan"),
            ("Quay trai", "Nhay mat 3 lan"),
            ("Quay phai", "Nhay mat 3 lan"),
        ]

        # Theo doi mat khuon mat
        self.no_face_counter = 0
        self.max_no_face_frames = 30  # 30 frames khong co mat thi reset
        
        # Khoi tao thu thach dau tien
        self._reset_challenge()

    def _reset_challenge(self):
        """Reset va tao thu thach moi"""
        self.current_challenge = random.choice(self.challenges)
        self.challenge_status = [False, False]
        self.blink_count = 0
        self.head_direction = None
        self.start_time = None
        self.last_blink_time = 0
        self.is_live = False
        self.frame_count = 0
        self.initialization_done = False
        self.initial_landmarks = None
        self.blink_history = []
        self.head_pose_history = []
        print(f"Thu thach moi: {self.current_challenge[0]} + {self.current_challenge[1]}")

    def check_liveness(self, image):
        """Kiem tra tinh sinh dong dua tren thu thach ngau nhien"""
        self.frame_count += 1
        processed_image = image.copy()
        h, w, _ = image.shape

        # Chuyen sang RGB cho MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        # Phat hien khuon mat
        if results.multi_face_landmarks:
            # Reset bo dem khong co mat
            self.no_face_counter = 0
            
            landmarks = results.multi_face_landmarks[0].landmark

            # Khoi tao cac landmark ban dau
            if not self.initialization_done:
                if self.frame_count <= self.initialization_frames:
                    cv2.putText(
                        processed_image,
                        f"Initializing... {self.frame_count}/{self.initialization_frames}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                    )

                    if self.frame_count == self.initialization_frames:
                        self.initial_landmarks = landmarks
                        self.initialization_done = True
                        self.start_time = time.time()
                else:
                    self.initialization_done = True
                    self.start_time = time.time()
            else:
                # Da khoi tao xong, kiem tra thu thach
                blink_detected = self._detect_blink(landmarks)
                if blink_detected:
                    current_time = time.time()
                    if current_time - self.last_blink_time > 0.5:
                        self.blink_count += 1
                        self.last_blink_time = current_time

                # Kiem tra huong dau
                head_direction = self._detect_head_direction(landmarks)
                if head_direction:
                    self.head_direction = head_direction

                # Kiem tra hoan thanh thu thach
                self._check_challenge_completion()

                # Ve trang thai
                status = "REAL FACE" if self.is_live else "VERIFYING..."
                color = (0, 255, 0) if self.is_live else (0, 165, 255)
                cv2.putText(processed_image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # Hien thi thong tin thu thach
                self._draw_challenge_info(processed_image)

                # Hien thi thoi gian
                if self.start_time is not None:
                    elapsed = time.time() - self.start_time
                    cv2.putText(
                        processed_image,
                        f"Time: {elapsed:.1f}s",
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
        else:
            # Khong tim thay khuon mat
            self.no_face_counter += 1
            
            # Hien thi thong bao
            cv2.putText(
                processed_image,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            
            # Hien thi so frame khong co mat
            cv2.putText(
                processed_image,
                f"No face: {self.no_face_counter}/{self.max_no_face_frames}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
            
            # Neu qua lau khong co mat thi tao thu thach moi
            if self.no_face_counter >= self.max_no_face_frames:
                cv2.putText(
                    processed_image,
                    "Creating new challenge...",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                self._reset_challenge()

        return self.is_live, processed_image

    def _detect_blink(self, landmarks):
        """Phat hien nhay mat dua tren Eye Aspect Ratio (EAR)"""
        if landmarks is None:
            return False

        def calc_ear(eye_points):
            if any(idx >= len(landmarks) for idx in eye_points):
                return 1.0

            points = [landmarks[idx] for idx in eye_points]
            v1 = np.sqrt((points[1].x - points[5].x) ** 2 + (points[1].y - points[5].y) ** 2)
            v2 = np.sqrt((points[2].x - points[4].x) ** 2 + (points[2].y - points[4].y) ** 2)
            h = np.sqrt((points[0].x - points[3].x) ** 2 + (points[0].y - points[3].y) ** 2)

            if h == 0:
                return 1.0
            return (v1 + v2) / (2.0 * h)

        left_ear = calc_ear(self.LEFT_EYE)
        right_ear = calc_ear(self.RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        self.blink_history.append(avg_ear)
        if len(self.blink_history) > 20:
            self.blink_history.pop(0)

        if len(self.blink_history) < 10:
            return False

        # Tim mau nhay mat: ear ban dau lon, giam, roi tang lai
        for i in range(3, len(self.blink_history) - 3):
            if (
                self.blink_history[i - 3] > 0.2
                and self.blink_history[i] < 0.15
                and self.blink_history[i + 3] > 0.2
            ):
                return True
        return False

    def _detect_head_direction(self, landmarks):
        """Phat hien huong xoay dau"""
        if landmarks is None or self.initial_landmarks is None:
            return None

        key_points = [self.NOSE_TIP, self.LEFT_FACE, self.RIGHT_FACE]
        if any(idx >= len(landmarks) for idx in key_points):
            return None

        nose = landmarks[self.NOSE_TIP]
        left = landmarks[self.LEFT_FACE]
        right = landmarks[self.RIGHT_FACE]

        init_nose = self.initial_landmarks[self.NOSE_TIP]
        init_left = self.initial_landmarks[self.LEFT_FACE]
        init_right = self.initial_landmarks[self.RIGHT_FACE]

        left_dist = np.sqrt((nose.x - left.x) ** 2 + (nose.y - left.y) ** 2)
        right_dist = np.sqrt((nose.x - right.x) ** 2 + (nose.y - right.y) ** 2)
        init_left_dist = np.sqrt((init_nose.x - init_left.x) ** 2 + (init_nose.y - init_left.y) ** 2)
        init_right_dist = np.sqrt((init_nose.x - init_right.x) ** 2 + (init_nose.y - init_right.y) ** 2)

        left_ratio = left_dist / init_left_dist if init_left_dist > 0 else 1.0
        right_ratio = right_dist / init_right_dist if init_right_dist > 0 else 1.0

        self.head_pose_history.append((left_ratio, right_ratio))
        if len(self.head_pose_history) > 10:
            self.head_pose_history.pop(0)

        recent_poses = self.head_pose_history[-5:] if len(self.head_pose_history) >= 5 else self.head_pose_history
        avg_left_ratio = np.mean([pose[0] for pose in recent_poses])
        avg_right_ratio = np.mean([pose[1] for pose in recent_poses])

        threshold = 1.15

        if avg_left_ratio > threshold and avg_right_ratio < 1.0:
            return "left"
        elif avg_right_ratio > threshold and avg_left_ratio < 1.0:
            return "right"
        return None

    def _check_challenge_completion(self):
        """Kiem tra hoan thanh thu thach"""
        # Kiem tra hoan thanh thu thach 1
        if "Nhay mat 2 lan" in self.current_challenge[0] and self.blink_count >= 2:
            self.challenge_status[0] = True
        elif "Nhay mat 3 lan" in self.current_challenge[0] and self.blink_count >= 3:
            self.challenge_status[0] = True
        elif "Quay trai" in self.current_challenge[0] and self.head_direction == "left":
            self.challenge_status[0] = True
        elif "Quay phai" in self.current_challenge[0] and self.head_direction == "right":
            self.challenge_status[0] = True

        # Kiem tra hoan thanh thu thach 2
        if "Nhay mat 2 lan" in self.current_challenge[1] and self.blink_count >= 2:
            self.challenge_status[1] = True
        elif "Nhay mat 3 lan" in self.current_challenge[1] and self.blink_count >= 3:
            self.challenge_status[1] = True
        elif "Quay trai" in self.current_challenge[1] and self.head_direction == "left":
            self.challenge_status[1] = True
        elif "Quay phai" in self.current_challenge[1] and self.head_direction == "right":
            self.challenge_status[1] = True

        # Neu da hoan thanh ca hai thu thach
        if all(self.challenge_status):
            self.is_live = True

    def _draw_challenge_info(self, processed_image):
        """Ve thong tin thu thach"""
        # Xác định số lần nhay mat cần thiết
        blinks_required = 0
        for challenge in self.current_challenge:
            if "Nhay mat 2 lan" in challenge:
                blinks_required = max(blinks_required, 2)
            elif "Nhay mat 3 lan" in challenge:
                blinks_required = max(blinks_required, 3)

        # Kiểm tra trạng thái hoàn thành cho phần hướng đầu
        head_complete = False
        for i, challenge in enumerate(self.current_challenge):
            is_head_challenge = "Quay trai" in challenge or "Quay phai" in challenge
            if is_head_challenge and self.challenge_status[i]:
                head_complete = True

        # Hiển thị số lần nhay mat (chưa đủ thì hiển thị)
        if self.blink_count < blinks_required:
            cv2.putText(
                processed_image,
                f"Blinks: {self.blink_count}/{blinks_required}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Hiển thị hướng đầu (chưa hoàn thành thì màu đỏ)
        if not head_complete:
            head_status = self.head_direction if self.head_direction else "center"

            # Xác định hướng đầu được yêu cầu từ thử thách
            required_direction = ""
            for challenge in self.current_challenge:
                if "Quay trai" in challenge:
                    required_direction = "left"
                    break
                elif "Quay phai" in challenge:
                    required_direction = "right"
                    break

            direction_text = f"Head: {head_status} -> {required_direction}"
            cv2.putText(
                processed_image,
                direction_text,
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

    def reset(self):
        """Reset trang thai de bat dau xac thuc moi"""
        self._reset_challenge()


def anti_spoofing():
    """Test module chong gia mao dua tren thu thach ngau nhien"""
    anti_spoof = AntiSpoofing()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Khong the mo camera!")
        return

    print("Nhan 'q' de thoat hoac 'r' de reset va tao thu thach moi")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            is_live, processed_frame = anti_spoof.check_liveness(frame)
            cv2.imshow("Challenge Verification", processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                anti_spoof.reset()
                print(f"Reset voi thu thach moi: {anti_spoof.current_challenge}")
    except Exception as e:
        print(f"Loi: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    anti_spoofing()