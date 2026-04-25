"""
Camera-based face tracking — makes the character **look at the user**.

Uses MediaPipe FaceLandmarker to locate the user's face position
in the camera frame. The character's head and eyes then follow
the user's position (not the user's own gaze direction).

Returns (face_x, face_y, detected):
    face_x: user position on horizontal axis, [-1, 1]
    face_y: user position on vertical axis,   [-1, 1]
    (0, 0) = camera centre
"""

import os
import threading
import time
import urllib.request

import cv2
import numpy as np

_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)


class FaceTracker:
    """
    Detects user's face position so the character can look at the user.

    Uses the nose tip (landmark 1) for stable face-centre estimation.
    Auto-calibrates for the first ~1 s to zero out the neutral position.
    """

    def __init__(self, camera_index: int = 0, mirror: bool = True):
        self._face_x: float = 0.0
        self._face_y: float = 0.0
        self._detected: bool = False

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._mirror = mirror
        self._camera_index = camera_index
        self._last_detect_time: float = 0.0

        # Smoothing
        self._smooth_alpha = 0.18

        # Auto-calibration
        self._calib_frames = 0
        self._calib_target = 30  # ~1 second
        self._calib_sum_x = 0.0
        self._calib_sum_y = 0.0
        self._offset_x = 0.0
        self._offset_y = 0.0

        self._landmarker = None
        self._init_landmarker()

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[FaceTracker] 已启动 (camera={camera_index}, mode=look-at-user)")

    def _get_model_path(self) -> str:
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "face_landmarker.task")
        if not os.path.exists(model_path):
            print("[FaceTracker] 正在下载 FaceLandmarker 模型...")
            urllib.request.urlretrieve(_LANDMARKER_URL, model_path)
            print("[FaceTracker] 模型下载完成")
        return model_path

    def _init_landmarker(self):
        try:
            import mediapipe as mp
            model_path = self._get_model_path()
            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"[FaceTracker] FaceLandmarker 初始化失败: {e}")

    def _capture_loop(self):
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            print("[FaceTracker] 无法打开摄像头，面部追踪已禁用")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            if self._mirror:
                frame = cv2.flip(frame, 1)

            pos = self._detect(frame)

            with self._lock:
                if pos is not None:
                    rx, ry = pos

                    # Auto-calibration
                    if self._calib_frames < self._calib_target:
                        self._calib_frames += 1
                        self._calib_sum_x += rx
                        self._calib_sum_y += ry
                        if self._calib_frames == self._calib_target:
                            self._offset_x = self._calib_sum_x / self._calib_target
                            self._offset_y = self._calib_sum_y / self._calib_target
                            print(f"[FaceTracker] 校准完成: offset=({self._offset_x:+.3f},{self._offset_y:+.3f})")

                    rx -= self._offset_x
                    ry -= self._offset_y

                    a = self._smooth_alpha
                    self._face_x = self._face_x * (1 - a) + rx * a
                    self._face_y = self._face_y * (1 - a) + ry * a
                    self._detected = True
                    self._last_detect_time = time.time()
                else:
                    if time.time() - self._last_detect_time > 0.5:
                        self._detected = False

            time.sleep(0.033)

        cap.release()

    def _detect(self, frame) -> "tuple[float, float] | None":
        """Return (x, y) of user's face centre in [-1, 1], or None."""
        if self._landmarker is None:
            return None
        try:
            import mediapipe as mp
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_image)

            if not result.face_landmarks:
                return None

            # Use nose tip (landmark 1) as stable face centre
            nose = result.face_landmarks[0][1]
            x = (nose.x - 0.5) * 2.0   # [0,1] → [-1,1]
            y = (nose.y - 0.5) * 2.0
            return (x, y)

        except RuntimeError:
            return None

    def get_face_position(self) -> tuple:
        """
        Returns ``(face_x, face_y, detected)``.

        face_x/y: user's face position in [-1, 1], 0 = centre.
        detected: bool
        """
        with self._lock:
            return self._face_x, self._face_y, self._detected

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
