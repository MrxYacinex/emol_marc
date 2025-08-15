from flask import Flask, request, jsonify
from flask_cors import CORS

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
    
    # Try to initialize MediaPipe models
    try:
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        print("✓ MediaPipe models initialized successfully")
    except Exception as model_error:
        print(f"⚠ MediaPipe models failed to initialize: {str(model_error)}")
        face_mesh = None
        pose = None
        hands = None
    
    MEDIAPIPE_AVAILABLE = True
    print("✓ MediaPipe loaded successfully")
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"⚠ MediaPipe not available: {str(e)}")
    print("  Server will continue without MediaPipe functionality")

app = Flask(__name__)
CORS(app)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    print("📥 Received analysis request")
    return jsonify({
        "status": "awake",
        "avgEAR": 75.0,
        "gazeLeft": "center",
        "gazeRight": "center",
        "attention": "focused",
        "lernfaehigkeitsScore": 80,
        "methodUsed": "test",
        "facesDetected": 1,
        "headPose": {"pitch": 0, "yaw": 0, "roll": 0},
        "handAnalysis": {
            "hand_fatigue_detected": False,
            "hand_at_head": False,
            "playing_with_hair": False,
            "hand_movement": "normal"
        }
    })

@app.route("/api/test-gemini", methods=["POST"])
def test_gemini():
    return jsonify({
        "status": "success",
        "gemini_test": {"success": True, "test_recommendation": "Great focus!"}
    })

if __name__ == "__main__":
    print("🚀 Starting Test Server...")
    app.run(host='0.0.0.0', port=5000, debug=True)