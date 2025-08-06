from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import dlib
import base64
from combined import analyze_frame as analyze_frame_combined

app = Flask(__name__)
CORS(app)

# Load models
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

@app.route('/api/analyze_frame', methods=['POST'])
def analyze_frame():
    try:
        data = request.json
        
        # Validate input data
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Extract and validate image data
        image_data_parts = data['image'].split(',')
        if len(image_data_parts) != 2:
            return jsonify({"error": "Invalid image data format"}), 400
        
        image_data = image_data_parts[1]
        
        # Decode image
        try:
            decoded_image = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({"error": f"Failed to decode base64 image: {str(e)}"}), 400
        
        # Convert to numpy array
        np_arr = np.frombuffer(decoded_image, np.uint8)
        
        # Check if buffer is not empty
        if np_arr.size == 0:
            return jsonify({"error": "Empty image buffer"}), 400
        
        # Decode image with OpenCV
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Check if image was successfully decoded
        if frame is None:
            return jsonify({"error": "Failed to decode image with OpenCV"}), 400

        # Perform analysis using the correct function from combined.py
        analysis_results = analyze_frame_combined(frame)

        # Handle case where analysis_results might be None or empty
        if analysis_results is None:
            analysis_results = {}

        # Prepare response matching frontend expectations
        # The frontend expects these specific property names
        response_data = {
            # Focus/Learning score
            "lernfaehigkeitsScore": analysis_results.get("lernfaehigkeit", 0),
            "lernfaehigkeit": analysis_results.get("lernfaehigkeit", 0),  # Keep both for compatibility
            
            # Eye aspect ratio
            "ear": analysis_results.get("ear", 0),
            "avgEAR": analysis_results.get("ear", 0),  # Frontend expects avgEAR for final analysis
            
            # Gaze direction
            "left_gaze": analysis_results.get("left_gaze", "unknown"),
            "right_gaze": analysis_results.get("right_gaze", "unknown"),
            "gazeLeft": analysis_results.get("left_gaze", "unknown"),  # Frontend expects gazeLeft
            "gazeRight": analysis_results.get("right_gaze", "unknown"),  # Frontend expects gazeRight
            
            # Head pose
            "pitch": analysis_results.get("pitch", 0),
            "yaw": analysis_results.get("yaw", 0),
            "roll": analysis_results.get("roll", 0),
            "headPose": {
                "pitch": analysis_results.get("pitch", 0),
                "yaw": analysis_results.get("yaw", 0),  
                "roll": analysis_results.get("roll", 0)
            },
            
            # Hand analysis
            "hand_analysis": analysis_results.get("hand_analysis", {
                "hand_fatigue_detected": False,
                "hand_at_head": False,
                "playing_with_hair": False,
                "hand_movement": "normal"
            }),
            "handAnalysis": analysis_results.get("hand_analysis", {  # Frontend expects handAnalysis
                "hand_fatigue_detected": False,
                "hand_at_head": False,
                "playing_with_hair": False,
                "hand_movement": "normal"
            }),
            
            # Status and attention - these might need to be derived from other values
            "status": analysis_results.get("status", "awake"),
            "attention": analysis_results.get("attention", "focused")
        }

        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error in analyze_frame: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)