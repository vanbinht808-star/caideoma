import base64
import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, send_from_directory
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
MODEL_NAME = os.getenv("YOLO26_POSE_MODEL", "yolo26n-pose.pt")

app = Flask(__name__, static_folder=None)

try:
    pose_model = YOLO(MODEL_NAME)
except Exception:
    MODEL_NAME = "yolo11n-pose.pt"
    pose_model = YOLO(MODEL_NAME)

hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=os.path.join(ROOT_DIR, "hand_landmarker.task")),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
hands = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)

camera = None
camera_lock = threading.Lock()
state_lock = threading.Lock()
running = False
delay_seconds = 0.0
last_frame = None

POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 17), (12, 18),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
]


def estimated_distance(keypoints, box):
    if keypoints is not None and len(keypoints) >= 17:
        head = keypoints[0]
        ankles = keypoints[[15, 16]]
        visible_head = head[2] > 0.3
        visible_ankles = np.mean(ankles[:, 2]) > 0.3
        if visible_head and visible_ankles:
            body_height = max(abs(float(head[1]) - float(np.max(ankles[:, 1]))), 1.0)
            return 1190.0 / body_height

    x1, y1, x2, y2 = map(float, box)
    body_height = max(y2 - y1, 1.0)
    return 1190.0 / body_height


def draw_person(frame, result):
    if result.boxes is None:
        return

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
    confidences = result.keypoints.conf.cpu().numpy() if result.keypoints is not None else None

    for index, box in enumerate(boxes):
        if classes[index] != 0:
            continue

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 220, 120), 1)

        if keypoints is not None and index < len(keypoints):
            points = keypoints[index]
            scores = confidences[index] if confidences is not None and index < len(confidences) else np.ones(len(points))
            kp = np.column_stack((points, scores))
            for start, end in POSE_CONNECTIONS:
                if start < len(kp) and end < len(kp) and kp[start, 2] > 0.3 and kp[end, 2] > 0.3:
                    cv2.line(frame, tuple(kp[start, :2].astype(int)), tuple(kp[end, :2].astype(int)), (0, 255, 150), 2)
            for point in kp:
                if point[2] > 0.3:
                    cv2.circle(frame, tuple(point[:2].astype(int)), 4, (0, 220, 255), -1)

            distance = estimated_distance(kp, box)
            label_y = max(y1 - 8, 20)
            cv2.putText(frame, f"{distance:.2f} m", (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_hands(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    hand_results = hands.detect(image)

    height, width = frame.shape[:2]
    for hand_landmarks in hand_results.hand_landmarks:
        points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in hand_landmarks]

        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (255, 130, 40), 2)
        for point in points:
            cv2.circle(frame, point, 4, (255, 255, 255), -1)


def capture_frames():
    global camera, last_frame, running

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 60)

    while True:
        with state_lock:
            should_run = running
        if not should_run:
            break

        success, frame = camera.read()
        if not success:
            continue

        results = pose_model.predict(frame, imgsz=640, conf=0.35, verbose=False)[0]
        draw_person(frame, results)
        draw_hands(frame)

        cv2.putText(frame, "YOLO26 Pose + Hands + Distance", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        with camera_lock:
            last_frame = frame

        with state_lock:
            current_delay = delay_seconds
        if current_delay > 0:
            time.sleep(current_delay)

    if camera is not None:
        camera.release()
        camera = None
    with camera_lock:
        last_frame = None


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def frontend(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/api/camera/start", methods=["POST"])
def start_camera():
    global running
    with state_lock:
        if running:
            return jsonify({"running": True})
        running = True
    threading.Thread(target=capture_frames, daemon=True).start()
    return jsonify({"running": True})


@app.route("/api/camera/stop", methods=["POST"])
def stop_camera():
    global running
    with state_lock:
        running = False
    return jsonify({"running": False})


@app.route("/api/delay", methods=["POST"])
def set_delay():
    global delay_seconds
    from flask import request
    delay_seconds = float(request.get_json(silent=True).get("delay", 0.0))
    return jsonify({"delay": delay_seconds})


@app.route("/api/status")
def status():
    with state_lock:
        return jsonify({"running": running, "delay": delay_seconds, "model": MODEL_NAME})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    from flask import request

    image_data = (request.get_json(silent=True) or {}).get("image", "")
    if not image_data:
        return jsonify({"width": 640, "height": 640, "people": [], "hands": []})

    try:
        encoded = base64.b64decode(image_data)
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({"width": 640, "height": 640, "people": [], "hands": []})

    if frame is None:
        return jsonify({"width": 640, "height": 640, "people": [], "hands": []})

    height, width = frame.shape[:2]
    result = pose_model.predict(frame, imgsz=640, conf=0.35, verbose=False)[0]
    people = []

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
        confidences = result.keypoints.conf.cpu().numpy() if result.keypoints is not None else None

        for index, box in enumerate(boxes):
            if classes[index] != 0 or keypoints is None or index >= len(keypoints):
                continue

            points = keypoints[index]
            scores = confidences[index] if confidences is not None and index < len(confidences) else np.ones(len(points))
            keypoint_data = [
                {"x": float(point[0]), "y": float(point[1]), "score": float(score)}
                for point, score in zip(points, scores)
            ]
            kp = np.column_stack((points, scores))
            people.append({
                "keypoints": keypoint_data,
                "distance": estimated_distance(kp, box)
            })

    hands_data = []
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    hand_results = hands.detect(image)
    for index, hand_landmarks in enumerate(hand_results.hand_landmarks):
        hand_points = [
            {"x": float(point.x * width), "y": float(point.y * height)}
            for point in hand_landmarks
        ]

        label = "Unknown"
        if hand_results.handedness and index < len(hand_results.handedness):
            raw_label = hand_results.handedness[index][0].category_name
            label = "Left" if raw_label == "Right" else "Right"

        hands_data.append({"points": hand_points, "connections": HAND_CONNECTIONS, "label": label})

    return jsonify({"width": width, "height": height, "people": people, "hands": hands_data})


@app.route("/stream")
def stream():
    def generate():
        boundary = b"--frame\r\n"
        while True:
            with state_lock:
                should_run = running
            if not should_run:
                time.sleep(0.02)
                continue

            with camera_lock:
                frame = None if last_frame is None else last_frame.copy()
            if frame is None:
                time.sleep(0.01)
                continue

            encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + encoded + b"\r\n"

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4173, threaded=True, debug=False)
