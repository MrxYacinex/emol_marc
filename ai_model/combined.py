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
import logging
import uuid
import atexit
from datetime import datetime

# AI Agent Integration
try:
    from agents.analyse import LearningAnalyzer
    from gemini import GeminiAnalyzer
    AI_AGENT_AVAILABLE = True
    print("✓ AI Agent Module verfügbar")
except ImportError as e:
    AI_AGENT_AVAILABLE = False
    print(f"⚠️ AI Agent nicht verfügbar: {e}")

app = Flask(__name__)
CORS(app)

# AI Agent Globale Variablen
learning_analyzer = None
current_recommendations = None

# Global session tracking
current_session_id = None
session_start_time = None

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
# Initialize MediaPipe variables
mp_face_mesh = None
mp_pose = None
mp_hands = None
face_mesh = None
pose = None
hands = None
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    print("mp_face_mesh", mp_face_mesh)
    print("mp_pose", mp_pose)
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # Lowered detection thresholds for better hand detection
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,  # Lowered from 0.7
        min_tracking_confidence=0.5    # Lowered from 0.7
    )
    MEDIAPIPE_AVAILABLE = True
    print("✓ MediaPipe loaded successfully")
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"⚠ MediaPipe not available: {str(e)}")
    print("  Server will continue without MediaPipe functionality")

# Database Bridge Integration
try:
    import sys
    # Add the parent directory to sys.path to import from the Next.js project
    nextjs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(nextjs_path)
    
    # Import the database bridge
    from database_bridge import DatabaseBridge
    db_bridge = DatabaseBridge()
    DATABASE_AVAILABLE = True
    print("✓ Database Bridge connected successfully")
except ImportError as e:
    DATABASE_AVAILABLE = False
    db_bridge = None
    print(f"⚠️ Database Bridge not available: {e}")

hand_close_start_time = None
FATIGUE_TIME_THRESHOLD = 3  
MOVEMENT_THRESHOLD = 0.01  
prev_hand_pos = None
hand_positions = []

# AI Agent Funktionen
def initialize_ai_agent():
    """AI Agent initialisieren"""
    global learning_analyzer, current_recommendations
    
    if not AI_AGENT_AVAILABLE:
        print("⚠️ AI Agent Module nicht verfügbar")
        return False
    
    try:
        # Gemini Analyzer initialisieren
        gemini_analyzer = GeminiAnalyzer()
        
        # Learning Analyzer initialisieren
        learning_analyzer = LearningAnalyzer(
            analysis_window_minutes=5,
            recommendation_interval_minutes=10
        )
        
        # Gemini mit Learning Analyzer verbinden
        learning_analyzer.set_gemini_analyzer(gemini_analyzer)
        
        # Kontinuierliche Analyse starten
        learning_analyzer.start_continuous_analysis(
            recommendation_callback=handle_new_recommendations
        )
        
        print("🚀 AI Agent erfolgreich initialisiert")
        return True
        
    except Exception as e:
        print(f"❌ Fehler bei AI Agent Initialisierung: {e}")
        return False

def handle_new_recommendations(recommendations: dict, trend_analysis: dict):
    """Neue Empfehlungen verarbeiten"""
    global current_recommendations
    current_recommendations = {
        "recommendations": recommendations,
        "trend_analysis": trend_analysis,
        "timestamp": time.time()
    }
    
    print("📋 Neue AI-Empfehlungen erhalten:")
    print(f"  Status: {recommendations.get('overall_status')}")
    print(f"  Dringlichkeit: {recommendations.get('urgency_level')}")
    print(f"  Nutzerempfehlungen: {len(recommendations.get('user_recommendations', []))}")
    print(f"  App-Aktionen: {len(recommendations.get('app_actions', []))}")

# Versuche AI Agent beim Import zu initialisieren
try:
    if AI_AGENT_AVAILABLE:
        initialize_ai_agent()
except Exception as e:
    print(f"⚠️ AI Agent Initialisierung beim Start fehlgeschlagen: {e}")

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
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = euler_angles.flatten()

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
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = euler_angles.flatten()
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
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


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
    hair_right_pos, hair_left_pos = None, None

    if results_face.multi_face_landmarks:
        # The landmark for the bottom of the chin is at index 152
        chin_landmark = results_face.multi_face_landmarks[0].landmark[152]
        chin_pos = (chin_landmark.x, chin_landmark.y)
        print("✓ Face landmarks detected - chin position found")

        # Hair-Position (first estimate) landmarks 54 and 284
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
    else:
        print("❌ No hands detected - no wrist data available")
        print("💡 Tip: Make sure your hands are visible in the camera frame")

    # Check if pose landmarks were detected
    if results_pose.pose_landmarks:
        print("✓ Pose landmarks detected")
    else:
        print("❌ No pose landmarks detected")

    fatigue_detected = False
    hand_at_head = False
    playing_with_hair = False
    hand_movement = "normal"

    # Modified condition: Allow analysis even without hands if we have face landmarks
    # This will still provide basic fatigue detection based on facial features
    if results_hands.multi_hand_landmarks:
        # Full hand analysis when hands are detected
        print("🔍 Performing full hand + face analysis")
        
        pose_landmarks = None
        nose_pos, mouth_pos = None, None

        if results_pose.pose_landmarks:
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

            # wrist
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_pos = (wrist.x, wrist.y)
            print(f"✓ Processing Hand {i + 1}: Using wrist at ({wrist.x:.3f}, {wrist.y:.3f})")

            # pinky_finger_mcp_point
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
            pinky_mcp_pos = (pinky_mcp.x, pinky_mcp.y)

            # palm
            palm_outside_x = (wrist.x + pinky_mcp.x) / 2
            palm_outside_y = (wrist.y + pinky_mcp.y) / 2
            palm_pos = (palm_outside_x, palm_outside_y)
            print(f"✓ Hand {i + 1}: Palm position calculated at ({palm_outside_x:.3f}, {palm_outside_y:.3f})")

            # Fatigue detection - hand near head or chin
            if chin_pos:
                distance_hand_head = distance_points(palm_pos, chin_pos)
                print(f"✓ Hand {i + 1}: Distance from palm to chin: {distance_hand_head:.3f}")

                if distance_hand_head < 0.2:
                    print(f"⚠️ Hand {i + 1}: Close to head detected (distance: {distance_hand_head:.3f})")
                    hand_at_head = True
                    if hand_close_start_time is None:
                        hand_close_start_time = time.time()
                        print(f"⏱️ Hand {i + 1}: Started timer for fatigue detection")
                    else:
                        elapsed_time = time.time() - hand_close_start_time
                        print(f"⏱️ Hand {i + 1}: Hand close for {elapsed_time:.1f} seconds")
                        if elapsed_time > FATIGUE_TIME_THRESHOLD:
                            fatigue_detected = True
                            print(f"😴 Hand {i + 1}: FATIGUE DETECTED! Hand close for {elapsed_time:.1f} seconds")
                else:
                    if hand_close_start_time is not None:
                        print(f"✓ Hand {i + 1}: Hand moved away from head - resetting timer")
                    hand_close_start_time = None
            elif nose_pos and mouth_pos:
                # Alternative head position detection using pose landmarks
                head_to_hand_distance = min(
                    distance_points(nose_pos, palm_pos),
                    distance_points(mouth_pos, palm_pos)
                )

                if head_to_hand_distance < 0.15:
                    print(
                        f"⚠️ Hand {i + 1}: Close to head detected (pose landmarks, distance: {head_to_hand_distance:.3f})")
                    hand_at_head = True
                    if hand_close_start_time is None:
                        hand_close_start_time = time.time()
                    else:
                        if time.time() - hand_close_start_time > FATIGUE_TIME_THRESHOLD:
                            fatigue_detected = True
                else:
                    hand_close_start_time = None

            # Playing with hair detection
            hair_play_detected = False

            # Method 1: Using hair landmarks if available
            if hair_right_pos and hair_left_pos:
                for tip_idx, tip in enumerate(finger_tips):
                    distance_finger_head = min(
                        distance_points(tip, hair_right_pos),
                        distance_points(tip, hair_left_pos)
                    )
                    if distance_finger_head < 0.15:
                        hair_play_detected = True
                        print(
                            f"💇 Hand {i + 1}: PLAYING WITH HAIR detected! Finger {tip_idx + 1} distance: {distance_finger_head:.3f}")
                        break

            # Method 2: Using nose position if available from pose
            if not hair_play_detected and nose_pos:
                for tip_idx, tip in enumerate(finger_tips):
                    if tip[1] < nose_pos[1] and distance_points(tip, nose_pos) < 0.12:
                        hair_play_detected = True
                        print(f"💇 Hand {i + 1}: PLAYING WITH HAIR detected (using nose)! Finger {tip_idx + 1}")
                        break

            playing_with_hair = playing_with_hair or hair_play_detected

            # Hand movement analysis
            current_hand_pos = wrist_pos
            hand_positions.append(current_hand_pos)

            if prev_hand_pos is not None:
                movement = distance_points(current_hand_pos, prev_hand_pos)
                print(f"✓ Hand {i + 1}: Movement detected: {movement:.4f}")
                if movement < MOVEMENT_THRESHOLD:
                    hand_movement = "ruhig"
                    print(f"😌 Hand {i + 1}: Hand movement is calm")
                else:
                    hand_movement = "normal"
                    print(f"👋 Hand {i + 1}: Hand movement is normal")
            else:
                print(f"✓ Hand {i + 1}: First frame - no previous position to compare")

            prev_hand_pos = current_hand_pos
            
    elif chin_pos:
        # Fallback analysis using only facial landmarks when hands are not detected
        print("🔍 Performing face-only analysis (hands not visible)")
        print("💡 Hand gestures cannot be analyzed without visible hands")
        
        # Reset hand-related timers since we can't see hands
        if hand_close_start_time is not None:
            print("🔄 Resetting hand fatigue timer (hands not visible)")
            hand_close_start_time = None
            
    else:
        print("❌ Insufficient landmarks for any analysis")
        print("  - Missing face landmarks for fallback analysis")
        if not results_hands.multi_hand_landmarks:
            print("  - Missing hand landmarks for full analysis")

    analysis_results = {
        'hand_fatigue_detected': fatigue_detected,
        'hand_at_head': hand_at_head,
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

def mirror_image(img):
    """Spiegelt das Bild horizontal (für natürlichere Kamera-Ansicht)"""
    return cv2.flip(img, 1)

@app.route("/api/start-session", methods=["POST"])
def start_session():
    global current_session_id, session_start_time
    try:
        data = request.get_json() or {}
        session_id = data.get('sessionId') or str(uuid.uuid4())
        
        current_session_id = session_id
        session_start_time = datetime.now()
        
        # Start session in database if available
        if DATABASE_AVAILABLE and db_bridge:
            success = db_bridge.start_session(session_id)
            if not success:
                print(f"⚠️ Failed to start session in database: {session_id}")
        
        return jsonify({
            "status": "success",
            "sessionId": session_id,
            "startTime": session_start_time.isoformat(),
            "message": "Session started successfully",
            "databaseConnected": DATABASE_AVAILABLE
        })
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Error starting session: {str(e)}"
        }), 500

@app.route("/api/end-session", methods=["POST"])
def end_session():
    global current_session_id, session_start_time
    try:
        data = request.get_json() or {}
        session_id = data.get('sessionId') or current_session_id
        
        if not session_id:
            return jsonify({
                "status": "error",
                "message": "No active session to end"
            }), 400
        
        # Calculate duration
        if session_start_time:
            duration = int((datetime.now() - session_start_time).total_seconds())
        else:
            duration = data.get('totalDuration', 0)
        
        # End session in database if available
        if DATABASE_AVAILABLE and db_bridge:
            success = db_bridge.end_session(session_id, duration)
            if not success:
                print(f"⚠️ Failed to end session in database: {session_id}")
        
        current_session_id = None
        session_start_time = None
        
        return jsonify({
            "status": "success",
            "sessionId": session_id,
            "duration": duration,
            "message": "Session ended successfully",
            "databaseConnected": DATABASE_AVAILABLE
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error ending session: {str(e)}"
        }), 500

@app.route("/api/database/export", methods=["GET"])
def export_database():
    """Export database data as JSON"""
    try:
        if DATABASE_AVAILABLE and db_bridge:
            # Get real data from database
            data = db_bridge.get_all_data()
            print(f"🔄 Exporting database data:")
            print(f"  Sessions: {len(data.get('sessions', []))}")
            print(f"  Analyses: {len(data.get('analyses', []))}")
            print(f"  Summaries: {len(data.get('summaries', []))}")
            return jsonify({
                "success": True,
                "data": data,
                "timestamp": datetime.now().isoformat(),
                "source": "database"
            })
        else:
            # Return empty data structure (no mock data)
            print("⚠️ Database not available, returning empty data structure")
            empty_data = {
                "sessions": [],
                "analyses": [],
                "summaries": []
            }
            return jsonify({
                "success": True,
                "data": empty_data,
                "timestamp": datetime.now().isoformat(),
                "source": "empty"
            })
    except Exception as e:
        print(f"❌ Error exporting database: {e}")
        return jsonify({
            "success": False,
            "error": f"Failed to export database: {str(e)}"
        }), 500

@app.route("/api/database/export", methods=["POST"])
def download_database():
    """Download database data as file"""
    try:
        data = request.get_json() or {}
        format_type = data.get('format', 'json')
        
        if format_type == 'json':
            # Get the real data (same as GET endpoint)
            if DATABASE_AVAILABLE and db_bridge:
                export_data = db_bridge.get_all_data()
            else:
                export_data = {
                    "sessions": [],
                    "analyses": [],
                    "summaries": [],
                    "exported_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
            
            from flask import Response
            import json
            
            json_string = json.dumps(export_data, indent=2)
            
            return Response(
                json_string,
                mimetype='application/json',
                headers={
                    'Content-Disposition': f'attachment; filename="study_data_{int(datetime.now().timestamp())}.json"'
                }
            )
        else:
            return jsonify({"error": "Unsupported format"}), 400
            
    except Exception as e:
        return jsonify({
            "error": f"Failed to download database: {str(e)}"
        }), 500

@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify({
        "status": "running",
        "currentSession": current_session_id,
        "sessionStartTime": session_start_time.isoformat() if session_start_time else None,
        "features": {
            "ai_agent": AI_AGENT_AVAILABLE,
            "mediapipe": MEDIAPIPE_AVAILABLE,
            "dlib": predictor is not None
        },
        "database": {
            "available": DATABASE_AVAILABLE,
            "type": "file_based" if DATABASE_AVAILABLE else "unavailable"
        }
    })

@app.route("/api/analyze", methods=["POST"])
def analyze():
    global current_session_id
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        image_data = data['image']
        img = decode_image(image_data)
        
        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        img = resize_image_keep_aspect(img, max_side=640)
        img = mirror_image(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_count = 0
        avgEAR = 999
        gazeLeft = "unknown"
        gazeRight = "unknown"
        attention = "No Data"
        lernfaehigkeitsScore = 0
        methodUsed = "none"
        pitch, yaw, roll = 0, 0, 0
        status = "unknown"
        face_analysis_successful = False  # Track if face analysis was successful

        # Hand analysis
        hand_analysis = analyze_hand_gestures(img)
        hand_fatigue_detected = hand_analysis['hand_fatigue_detected']
        hand_at_head = hand_analysis['hand_at_head']
        playing_with_hair = hand_analysis['playing_with_hair']
        hand_movement_status = hand_analysis['hand_movement']

        # Face detection logic
        if hog_detector is not None:
            faces = hog_detector(gray)
            face_count = len(faces)

            if face_count > 0:
                face = faces[0]
                landmarks = predictor(gray, face)
                landmarks = face_utils.shape_to_np(landmarks)

                leftEye = landmarks[36:42]
                rightEye = landmarks[42:48]

                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                avgEAR = (leftEAR + rightEAR) / 2.0

                gazeLeft = gaze_direction(leftEye, gray)
                gazeRight = gaze_direction(rightEye, gray)

                pitch, yaw, roll = get_head_pose(landmarks, img.shape)
                lernfaehigkeitsScore = lernfaehigkeits_score(avgEAR, gazeLeft, gazeRight, pitch, yaw)
                methodUsed = "dlib"
                face_analysis_successful = True  # Face analysis was successful
            else:
                methodUsed = "dlib_no_face"
        elif MEDIAPIPE_AVAILABLE:
            landmarks = get_landmarks_mediapipe(img)
            if landmarks is not None:
                face_count = 1
                
                if len(landmarks) >= 468:
                    left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                    right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                    
                    if all(i < len(landmarks) for i in left_eye_indices + right_eye_indices):
                        leftEye = landmarks[left_eye_indices]
                        rightEye = landmarks[right_eye_indices]
                        
                        leftEAR = eye_aspect_ratio(leftEye)
                        rightEAR = eye_aspect_ratio(rightEye)
                        avgEAR = (leftEAR + rightEAR) / 2.0
                        
                        gazeLeft = gaze_direction(leftEye, gray)
                        gazeRight = gaze_direction(rightEye, gray)
                        
                        pitch, yaw, roll = get_head_pose_mediapipe(landmarks, img.shape)
                        lernfaehigkeitsScore = lernfaehigkeits_score(avgEAR, gazeLeft, gazeRight, pitch, yaw)
                        methodUsed = "mediapipe"
                        face_analysis_successful = True  # Face analysis was successful
                    else:
                        methodUsed = "mediapipe_insufficient_landmarks"
                else:
                    methodUsed = "mediapipe_few_landmarks"
            else:
                methodUsed = "mediapipe_no_face"

        # Determine attention and status
        if avgEAR != 999:
            if avgEAR < 0.20:
                attention = "müde"
                status = "tired"
            elif gazeLeft == "center" and gazeRight == "center":
                attention = "focused"
                status = "awake"
            else:
                attention = "abgelenkt"
                status = "awake"
        else:
            attention = "No Data"
            status = "unknown"

        # Create response data
        response_data = {
            "status": status,
            "avgEAR": round(avgEAR, 2) if avgEAR != 999 else 999,
            "gazeLeft": gazeLeft,
            "gazeRight": gazeRight,
            "attention": attention,
            "lernfaehigkeitsScore": lernfaehigkeitsScore,
            "methodUsed": methodUsed,
            "facesDetected": face_count,
            "headPose": {
                "pitch": round(pitch, 3),
                "yaw": round(yaw, 3),
                "roll": round(roll, 3)
            },
            "handAnalysis": {
                "hand_fatigue_detected": hand_fatigue_detected,
                "hand_at_head": hand_at_head,
                "playing_with_hair": playing_with_hair,
                "hand_movement": hand_movement_status
            },
            "sessionId": current_session_id,
            "hasActiveSession": current_session_id is not None,
            "timestamp": datetime.now().isoformat(),
            "focusScore": lernfaehigkeitsScore,
            "attentionStatus": attention,
            "fatigueStatus": status,
            "gazeDirection": f"{gazeRight}/{gazeLeft}",
            "earValue": round(avgEAR, 2) if avgEAR != 999 else 0
        }
        
        # Only store analysis in database if face analysis was successful
        if current_session_id and DATABASE_AVAILABLE and db_bridge and face_analysis_successful:
            analysis_data = {
                "sessionId": current_session_id,
                "timestamp": response_data["timestamp"],
                "focusScore": lernfaehigkeitsScore,
                "attentionStatus": attention,
                "fatigueStatus": status,
                "gazeDirection": f"{gazeRight}/{gazeLeft}",
                "earValue": round(avgEAR, 2) if avgEAR != 999 else 0,
                "headPose": {
                    "pitch": round(pitch, 3),
                    "yaw": round(yaw, 3),
                    "roll": round(roll, 3)
                },
                "handAnalysis": {
                    "hand_fatigue_detected": hand_fatigue_detected,
                    "hand_at_head": hand_at_head,
                    "playing_with_hair": playing_with_hair,
                    "hand_movement": hand_movement_status
                },
                "methodUsed": methodUsed,
                "facesDetected": face_count
            }
            
            # Debug: Print what we're about to store
            print(f"💾 Storing analysis with Focus Score: {lernfaehigkeitsScore} (Face detected)")
            
            success = db_bridge.store_analysis(current_session_id, analysis_data)
            if success:
                print(f"📊 Stored analysis for session: {current_session_id}")
            else:
                print(f"❌ Failed to store analysis for session: {current_session_id}")
        elif current_session_id and not face_analysis_successful:
            print(f"⏭️ Skipping database storage - no face analysis (method: {methodUsed})")
                
        response_data["databaseStored"] = DATABASE_AVAILABLE and current_session_id is not None and face_analysis_successful
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error",
            "hasActiveSession": current_session_id is not None
        }), 500

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint"""
    capabilities = [
        "Müdigkeitserkennung (EAR + Hand am Kopf)",
        "Aufmerksamkeits-Tracking",
        "Lernfähigkeits-Score",
        "Hand-Gesten Analyse",
        "Blickrichtungs-Erkennung",
        "Kopfpose Estimation",
    ]
    
    if MEDIAPIPE_AVAILABLE:
        capabilities.append("MediaPipe Hand Tracking")
        capabilities.append("MediaPipe Face Mesh")
    
    return jsonify({
        "status": "fit",
        "capabilities": capabilities,
        "mediapipe_available": MEDIAPIPE_AVAILABLE,
        "dlib_available": hog_detector is not None,
        "ai_agent_available": AI_AGENT_AVAILABLE,
        "ai_agent_active": learning_analyzer is not None
    })

@app.route("/api/recommendations", methods=["GET"])
def get_recommendations():
    """Aktuelle AI-Empfehlungen abrufen"""
    if current_recommendations:
        return jsonify(current_recommendations)
    else:
        return jsonify({
            "status": "no_recommendations",
            "message": "Keine aktuellen Empfehlungen verfügbar"
        })

@app.route("/api/agent-status", methods=["GET"])
def get_agent_status():
    """AI Agent Status abrufen"""
    if learning_analyzer:
        status = learning_analyzer.get_current_status()
        status["available"] = True
        return jsonify(status)
    else:
        return jsonify({
            "available": False,
            "status": "not_initialized",
            "ai_agent_module_available": AI_AGENT_AVAILABLE
        })

@app.route("/api/force-analysis", methods=["POST"])
def force_analysis():
    """Sofortige AI-Analyse erzwingen (für Tests)"""
    if not learning_analyzer:
        return jsonify({
            "error": "AI Agent nicht initialisiert"
        }), 400
    
    try:
        result = learning_analyzer.force_analysis()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "error": f"Analyse fehlgeschlagen: {str(e)}"
        }), 500

@app.route("/api/motivation", methods=["GET"])
def get_motivation():
    """Motivations-Boost abrufen"""
    try:
        # Letzten Score aus dem Buffer holen
        if learning_analyzer.data_buffer:
            last_snapshot = learning_analyzer.data_buffer[-1]
            current_score = last_snapshot.learning_score
            
            # Trend basierend auf letzten paar Werten berechnen
            recent_scores = [d.learning_score for d in list(learning_analyzer.data_buffer)[-5:]]
            if len(recent_scores) >= 2:
                trend = "improving" if recent_scores[-1] > recent_scores[0] else "declining" if recent_scores[-1] < recent_scores[0] else "stable"
            else:
                trend = "stable"
            
            motivation = learning_analyzer.gemini_analyzer.generate_motivation_boost(current_score, trend)
            motivation["current_score"] = current_score
            motivation["trend"] = trend
            
            return jsonify(motivation)
        
    except Exception as e:
        return jsonify({
            "error": f"Motivations-Generierung fehlgeschlagen: {str(e)}"
        }), 500

@app.route("/api/test-gemini", methods=["POST"])
def test_gemini():
    """🧪 TOKEN-SPARSAMER Gemini Test (nur ~50-100 Tokens)"""
    try:
        data = request.get_json()
        focus_score = data.get('focus_score', 75)
        attention = data.get('attention', 'focused')
        
        # Quick Gemini Test über Learning Analyzer
        if learning_analyzer and learning_analyzer.gemini_analyzer:
            result = learning_analyzer.gemini_analyzer.test_gemini_quick(focus_score, attention)
        else:
            result = {"error": "Gemini Analyzer nicht verfügbar"}
        
        return jsonify({
            "status": "success",
            "gemini_test": result,
            "message": "Gemini Test abgeschlossen",
            "cost_info": "Minimale Kosten (~0.001€)"
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Gemini Test fehlgeschlagen: {str(e)}"
        }), 500

if __name__ == "__main__":
    print("🚀 Starting Unified AI Learning Analytics API...")
    print("📊 Features: Müdigkeitserkennung, Hand-Tracking, Lernfähigkeits-Score, Gaze-Detection")
    print(f"📱 MediaPipe: {'✓ Verfügbar' if MEDIAPIPE_AVAILABLE else '✗ Nicht verfügbar'}")
    print(f"🔍 Dlib: {'✓ Verfügbar' if hog_detector else '✗ Nicht verfügbar'}")
    print(f"💾 Database: {'✓ Verfügbar' if DATABASE_AVAILABLE else '✗ Nicht verfügbar'}")
    app.run(host='0.0.0.0', port=5000, debug=True)