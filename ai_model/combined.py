from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import dlib
from imutils import face_utils
from flask_cors import CORS
import os
import time
import math
import traceback
from flask import jsonify

app = Flask(__name__)
CORS(app)

# Initialize face detection models
try:
    # Use absolute path to ensure the model is found
    current_dir = os.path.dirname(os.path.abspath(__file__))
    predictor_path = os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat")
    predictor = dlib.shape_predictor(predictor_path)
    hog_detector = dlib.get_frontal_face_detector()
    print("✓ Dlib models loaded successfully")
except Exception as e:
    print(f"✗ Error loading dlib models: {e}")
    predictor = None
    hog_detector = None

# Try to initialize MediaPipe (fallback if not available)
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    MEDIAPIPE_AVAILABLE = True
    print("✓ MediaPipe loaded successfully")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠ MediaPipe not available - using dlib only mode")

# Global variables for hand tracking (from your test_hand.py)
hand_close_start_time = None
FATIGUE_TIME_THRESHOLD = 3  # Sekunden, ab wann Müdigkeit erkannt wird
MOVEMENT_THRESHOLD = 0.01   # minimale Bewegung für "ruhig"
prev_hand_pos = None
hand_positions = []

# Your original functions from main.py
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_head_pose(shape, img_shape):
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    image_points = np.array([
        shape[30],
        shape[8],
        shape[36],
        shape[45],
        shape[48],
        shape[54]
    ], dtype="double")

    size = (img_shape[0], img_shape[1])
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    rotation_mat, _ = cv2.Rodrigues(rotation_vector)

    sy = np.sqrt(rotation_mat[0, 0] * rotation_mat[0, 0] + rotation_mat[1, 0] * rotation_mat[1, 0])

    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(-rotation_mat[2, 0], sy)
        yaw = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
        roll = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
    else:
        pitch = np.arctan2(-rotation_mat[2, 0], sy)
        yaw = np.arctan2(-rotation_mat[0, 1], rotation_mat[1, 1])
        roll = 0

    pitch = np.rad2deg(pitch)
    yaw = np.rad2deg(yaw)
    roll = np.rad2deg(roll)
    
    return pitch, yaw, roll

def get_head_pose_mediapipe(landmarks, img_shape):
    """MediaPipe head pose estimation"""
    if not MEDIAPIPE_AVAILABLE:
        return 0, 0, 0
        
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nasenwurzel
        (0.0, -330.0, -65.0),        # Kinn
        (-225.0, 170.0, -135.0),     # linkes Auge links außen
        (225.0, 170.0, -135.0),      # rechtes Auge rechts außen
        (-150.0, -150.0, -125.0),    # linker Mundwinkel
        (150.0, -150.0, -125.0)      # rechter Mundwinkel
    ])

    image_points = np.array([
        landmarks[1],     # Nasenwurzel (index 1)
        landmarks[152],   # Kinn (index 152)
        landmarks[33],    # linkes Auge außen (index 33)
        landmarks[263],   # rechtes Auge außen (index 263)
        landmarks[78],    # linker Mundwinkel (index 78)
        landmarks[308],   # rechter Mundwinkel (index 308)
    ], dtype="double")

    size = (img_shape[0], img_shape[1])
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    
    sy = np.sqrt(rotation_mat[0, 0] * rotation_mat[0, 0] + rotation_mat[1, 0] * rotation_mat[1, 0])

    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(-rotation_mat[2, 0], sy)
        yaw = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
        roll = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
    else:
        pitch = np.arctan2(-rotation_mat[2, 0], sy)
        yaw = np.arctan2(-rotation_mat[0, 1], rotation_mat[1, 1])
        roll = 0

    pitch = np.rad2deg(pitch)
    yaw = np.rad2deg(yaw)
    roll = np.rad2deg(roll)

    return pitch, yaw, roll

def get_landmarks_mediapipe(image):
    """Get MediaPipe face landmarks"""
    if not MEDIAPIPE_AVAILABLE:
        return None
    
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        h, w, _ = image.shape
        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            landmarks.append((x, y))
        return np.array(landmarks)
    return None

def decode_image(base64_str):
    try:
        header, data = base64_str.split(',', 1)
        missing_padding = len(data) % 4
        if missing_padding:
            data += '=' * (4 - missing_padding)
        img_bytes = base64.b64decode(data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode gibt None zurück")
        return img
    except Exception as e:
        print("Fehler beim Bild-Decodieren:", e)
        return None

def gaze_direction(eye_points, gray_img):
    margin = 5
    min_x = np.min(eye_points[:, 0]) - margin
    max_x = np.max(eye_points[:, 0]) + margin
    min_y = np.min(eye_points[:, 1]) - margin
    max_y = np.max(eye_points[:, 1]) + margin

    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    max_x = min(max_x, gray_img.shape[1])
    max_y = min(max_y, gray_img.shape[0])

    eye_roi = gray_img[min_y:max_y, min_x:max_x]

    if eye_roi.size == 0:
        return "unknown"

    _, thresh = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown"

    max_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(max_contour)
    if M["m00"] == 0:
        return "unknown"
    cx = int(M["m10"] / M["m00"])

    eye_width = max_x - min_x
    gaze_ratio = cx / eye_width

    if gaze_ratio < 0.4:
        return "left"
    elif gaze_ratio > 0.6:
        return "right"
    else:
        return "center"

def lernfaehigkeits_score(ear, left_gaze, right_gaze, pitch, yaw):
    score = 0

    if ear >= 0.25:
        score += 40
    else:
        score += 0

    if left_gaze == "center" and right_gaze == "center":
        score += 30
    elif left_gaze == "center" or right_gaze == "center":
        score += 20
    else:
        score += 10

    if abs(yaw) <= 15 and abs(pitch) <= 15:
        score += 30
    else:
        score += 10

    return score

# Hand gesture analysis from test_hand.py
def distance_points(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)  #Normalize distance from screen)

def analyze_hand_gestures(img):
    """Analyze hand gestures for fatigue detection"""
    global hand_close_start_time, prev_hand_pos, hand_positions
    
    if not MEDIAPIPE_AVAILABLE:
        print("❌ MediaPipe not available - no hand/wrist data")
        # Simplified hand analysis based on timing and basic heuristics
        return {
            'hand_fatigue_detected': False,
            'hand_at_head': False,
            'playing_with_hair': False,
            'hand_movement': 'normal'
        }

    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process with face_mesh to get detailed facial landmarks
    results_face = face_mesh.process(frame_rgb)
    chin_pos = None
    if results_face.multi_face_landmarks:
        # The landmark for the bottom of the chin is at index 152
        chin_landmark = results_face.multi_face_landmarks[0].landmark[152]
        chin_pos = (chin_landmark.x, chin_landmark.y)
        print("✓ Face landmarks detected - chin position found")

        # Haar-Position (first estimate) landmarks 54 and 284 https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
        hair_right = results_face.multi_face_landmarks[0].landmark[54]
        hair_left = results_face.multi_face_landmarks[0].landmark[284]
        hair_offset = 0  # offset up to be adjusted
        hair_right_pos = (hair_right.x, hair_right.y + hair_offset)
        hair_left_pos = (hair_left.x, hair_left.y + hair_offset)
    else:
        print("❌ No face landmarks detected")

    results_pose = pose.process(frame_rgb)
    results_hands = hands.process(frame_rgb)

    # Check if hands were detected
    if results_hands.multi_hand_landmarks:
        num_hands = len(results_hands.multi_hand_landmarks)
        print(f"✓ Hand detection successful - {num_hands} hand(s) detected")
        
        # Check each hand for wrist data
        for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_pos = (wrist.x, wrist.y)
            print(f"✓ Hand {i+1}: Wrist data found at position ({wrist.x:.3f}, {wrist.y:.3f})")
    else:
        print("❌ No hands detected - no wrist data available")

    # Check if pose landmarks were detected
    if results_pose.pose_landmarks:
        print("✓ Pose landmarks detected")
    else:
        print("❌ No pose landmarks detected")

    fatigue_detected = False
    hand_at_head = False
    playing_with_hair = False
    hand_movement = "normal"

    # Ensure all landmarks (chin, pose, hands) were detected before proceeding
    if chin_pos and results_pose.pose_landmarks and results_hands.multi_hand_landmarks:
        print("✓ All required landmarks detected - proceeding with analysis")
        pose_landmarks = results_pose.pose_landmarks.landmark

        # Kopf-Merkmale: Nase und Mundmitte from Pose model
        nose = pose_landmarks[mp_pose.PoseLandmark.NOSE]
        mouth_left = pose_landmarks[mp_pose.PoseLandmark.MOUTH_LEFT]
        mouth_right = pose_landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT]

        mouth_center_x = (mouth_left.x + mouth_right.x) / 2
        mouth_center_y = (mouth_left.y + mouth_right.y) / 2
        mouth_pos = (mouth_center_x, mouth_center_y)
        nose_pos = (nose.x, nose.y)

        for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            # Finger-Tipps + Handgelenk prüfen
            finger_tips_ids = [
                mp_hands.HandLandmark.THUMB_TIP,
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP
            ]
            finger_tips = [
                (hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y) for i in finger_tips_ids
            ]

            #wrist
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_pos = (wrist.x, wrist.y)
            print(f"✓ Processing Hand {i+1}: Using wrist at ({wrist.x:.3f}, {wrist.y:.3f})")

            #pinky_finger_mcp_point
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
            pinky_mcp_pos = (pinky_mcp.x, pinky_mcp.y)

            #palm
            palm_outside_x = (wrist.x + pinky_mcp.x) / 2
            palm_outside_y = (wrist.y + pinky_mcp.y) / 2
            palm_pos = (palm_outside_x, palm_outside_y)
            print(f"✓ Hand {i+1}: Palm position calculated at ({palm_outside_x:.3f}, {palm_outside_y:.3f})")

            # Müdigkeit, wenn die Hand nahe am Kopf ist #vorher wrist_pos, nose_pos, now
            distance_hand_head = distance_points(palm_pos, chin_pos)
            print(f"✓ Hand {i+1}: Distance from palm to chin: {distance_hand_head:.3f}")

            if distance_hand_head < 0.2:
                print(f"⚠️ Hand {i+1}: Close to head detected (distance: {distance_hand_head:.3f})")
                if hand_close_start_time is None:
                    hand_close_start_time = time.time()
                    print(f"⏱️ Hand {i+1}: Started timer for fatigue detection")
                else:
                    elapsed_time = time.time() - hand_close_start_time
                    print(f"⏱️ Hand {i+1}: Hand close for {elapsed_time:.1f} seconds")
                    if elapsed_time > FATIGUE_TIME_THRESHOLD:
                        fatigue_detected = True
                        print(f"😴 Hand {i+1}: FATIGUE DETECTED! Hand close for {elapsed_time:.1f} seconds")
            else:
                if hand_close_start_time is not None:
                    print(f"✓ Hand {i+1}: Hand moved away from head - resetting timer")
                hand_close_start_time = None

            # Spielen mit den Haaren, wenn die Finger in der Nähe des Kopfes sind
            for tip_idx, tip in enumerate(finger_tips):
                distance_finger_head = min(distance_points(tip, hair_right_pos), distance_points(tip, hair_left_pos)) #before tip, nose_pos now tip, hair_right or hair_left
                if distance_finger_head < 0.15:
                    playing_with_hair = True
                    print(f"💇 Hand {i+1}: PLAYING WITH HAIR detected! Finger {tip_idx+1} distance: {distance_finger_head:.3f}")
                    break

            # Bewegung der Hand messen
            current_hand_pos = (wrist.x, wrist.y)
            hand_positions.append(current_hand_pos)

            if prev_hand_pos is not None:
                movement = distance_points(current_hand_pos, prev_hand_pos)
                print(f"✓ Hand {i+1}: Movement detected: {movement:.4f}")
                if movement < MOVEMENT_THRESHOLD:
                    hand_movement = "ruhig"
                    print(f"😌 Hand {i+1}: Hand movement is calm")
                else:
                    hand_movement = "normal"
                    print(f"👋 Hand {i+1}: Hand movement is normal")
            else:
                print(f"✓ Hand {i+1}: First frame - no previous position to compare")
            prev_hand_pos = current_hand_pos
    else:
        print("❌ Missing required landmarks - skipping detailed analysis")
        if not chin_pos:
            print("  - Missing chin position")
        if not results_pose.pose_landmarks:
            print("  - Missing pose landmarks")
        if not results_hands.multi_hand_landmarks:
            print("  - Missing hand landmarks")

    analysis_results = {
        'hand_fatigue_detected': fatigue_detected,
        'hand_at_head': hand_at_head,  # not implemented
        'playing_with_hair': playing_with_hair,
        'hand_movement': hand_movement
    }
    print("🔍 Hand Gesture Analysis Results:", analysis_results)
    return analysis_results


def resize_image_keep_aspect(img, max_side=640):
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h))
    return img

@app.route('/api/analyze_frame', methods=['POST'])
def analyze_frame_route():
    """Process a single frame"""
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    image_b64 = data["image"]
    img = decode_image(image_b64)
    if img is None:
        return jsonify({"error": "Decoding failed"}), 400
    
    result = analyze_frame(img)
    return jsonify(result)

# In your combined.py file, find the analyze_frame function.
# You need to change its definition to accept the 'frame' argument.

def analyze_frame(frame):
    # The rest of your existing code inside this function remains unchanged.
    # It should use the 'frame' variable that is now passed as an argument.
    img = frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Face detection using dlib
    rects = hog_detector(gray, 0) if hog_detector else []
    shape = None
    if rects:
        shape = predictor(gray, rects[0]) if predictor else None
        shape = face_utils.shape_to_np(shape) if shape else None

    # Fallback to MediaPipe face mesh if dlib fails
    if shape is None and MEDIAPIPE_AVAILABLE:
        shape = get_landmarks_mediapipe(img)
    
    if shape is None:
        return {
            "status": "kein Gesicht erkannt",
            "lernfaehigkeitsAnalyse": {
                "score": 0,
                "feedback": ["Kein Gesicht im Bild erkannt, Analyse nicht möglich."]
            }
        }

    # Eye aspect ratio
    if len(shape) == 68:
        leftEye = shape[36:42]
        rightEye = shape[42:48]
    else:
        leftEye_indices = [33, 160, 158, 133, 153, 144]
        rightEye_indices = [263, 387, 385, 362, 380, 373]
        leftEye = np.array([shape[i] for i in leftEye_indices])
        rightEye = np.array([shape[i] for i in rightEye_indices])

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0

    # Gaze direction
    left_gaze = gaze_direction(leftEye, gray)
    right_gaze = gaze_direction(rightEye, gray)

    # Head pose estimation
    if len(shape) == 68:
        pitch, yaw, roll = get_head_pose(shape, img.shape)
    else:
        pitch, yaw, roll = get_head_pose_mediapipe(shape, img.shape)

    hand_analysis = analyze_hand_gestures(img)
    
    # Learning capability score
    lernfaehigkeit = lernfaehigkeits_score(ear, left_gaze, right_gaze, pitch, yaw)

    return {
        "ear": ear,
        "left_gaze": left_gaze,
        "right_gaze": right_gaze,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "hand_analysis": hand_analysis,
        "lernfaehigkeit": lernfaehigkeit
    }

# Make sure you have 'app' or your actual Flask app instance defined.
# If your Flask app instance is named 'combined', use @combined.route(...) instead.
@app.route("/api/analyze", methods=["POST"])
def check_fatigue():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"status": "kein Bild"}), 400

        image_b64 = data.get("image")
        img = decode_image(image_b64)
        if img is None:
            return jsonify({"status": "Bild dekodieren fehlgeschlagen"}), 400

        img = resize_image_keep_aspect(img, 640)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects = hog_detector(gray, 1)

        if len(rects) > 0:
            rect = rects[0]
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            method_used = "dlib"
        else:
            # Kein Gesicht mit dlib, MediaPipe versuchen
            shape = get_landmarks_mediapipe(img)
            method_used = "mediapipe"
            if shape is None:
                return jsonify({"status": "kein Gesicht erkannt"})

        # Falls MediaPipe: keine 68 Punkte wie dlib, sondern 468. Für Augen + Head Pose nehmen wir Schlüsselindizes (Anpassung nötig)

        if method_used == "dlib":
            leftEye = shape[36:42]
            rightEye = shape[42:48]
        else:
            # MediaPipe: Augenindizes grob approximieren
            leftEye_indices = [33, 160, 158, 133, 153, 144]  # Beispiel für linkes Auge
            rightEye_indices = [263, 387, 385, 362, 380, 373]  # Beispiel für rechtes Auge
            leftEye = np.array([shape[i] for i in leftEye_indices])
            rightEye = np.array([shape[i] for i in rightEye_indices])

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        ear_percent = round(ear * 100, 1)

        status_fatigue = "tired" if ear < 0.25 else "awake"

        left_gaze = gaze_direction(leftEye, gray)
        right_gaze = gaze_direction(rightEye, gray)

        # Head pose nur bei dlib möglich, bei MediaPipe hier eine Dummy-Ausgabe (kann man später mit mp.solutions.face_mesh erweitern)
        if method_used == "dlib":
            pitch, yaw, roll = get_head_pose(shape, img.shape)
        else:
            pitch, yaw, roll = get_head_pose_mediapipe(shape, img.shape)

        attention_head_pose = "aufmerksam"
        if abs(yaw) > 20 or abs(pitch) > 15:
            attention_head_pose = "abgelenkt"

        attention_count = 0
        if ear >= 0.25:
            attention_count += 1
        if left_gaze == "center" and right_gaze == "center":
            attention_count += 1
        if attention_head_pose == "aufmerksam":
            attention_count += 1

        attention_status = "aufmerksam" if attention_count >= 2 else "abgelenkt"

        lernfaehigkeits_score_value = lernfaehigkeits_score(ear, left_gaze, right_gaze, pitch, yaw)

        print(f"Method: {method_used}, EAR: {ear_percent}, Gaze: {left_gaze}/{right_gaze}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f}")
        print(f"Attention count: {attention_count}, Status: {attention_status}, Lernfähigkeits-Score: {lernfaehigkeits_score_value}")

        return jsonify({
            "methodUsed": method_used,
            "status": status_fatigue,
            "avg EAR": ear_percent,
            "facesDetected": len(rects),
            "gazeLeft": left_gaze,
            "gazeRight": right_gaze,
            "headPose": {
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll
            },
            "attention": attention_status,
            "lernfaehigkeitsScore": lernfaehigkeits_score_value
        })
    except Exception as e:
        print(f"An exception occurred in check_fatigue: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')