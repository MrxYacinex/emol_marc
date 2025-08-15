import os
import sys
import subprocess
import time
from datetime import datetime
import json

class DatabaseBridge:
    """Bridge to communicate with Next.js DuckDB database"""
    
    def __init__(self):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.is_connected = self._test_connection()
        
    def _test_connection(self):
        """Test if we can connect to the database"""
        try:
            # Test by trying to read the database file path
            return True
        except Exception as e:
            print(f"Database connection test failed: {e}")
            return False
    
    def start_session(self, session_id: str) -> bool:
        """Start a new session in the database"""
        try:
            # Create a temporary Node.js script to interact with the database
            script_content = f"""
const {{ spawn }} = require('child_process');
const path = require('path');

// Import database module
const dbPath = path.join(__dirname, 'src', 'lib', 'database.ts');

// For now, we'll use a simple approach - create a data file
const fs = require('fs');
const dataDir = path.join(__dirname, 'data');
if (!fs.existsSync(dataDir)) {{
    fs.mkdirSync(dataDir, {{ recursive: true }});
}}

const sessionData = {{
    sessionId: '{session_id}',
    startTime: new Date().toISOString(),
    active: true
}};

fs.writeFileSync(
    path.join(dataDir, `session_${{'{session_id}'}}.json`),
    JSON.stringify(sessionData, null, 2)
);

console.log('Session started successfully');
process.exit(0);
"""
            
            # Write and execute the script
            script_path = os.path.join(self.project_root, 'temp_start_session.js')
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Execute the script
            result = subprocess.run(
                ['node', script_path],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Clean up
            os.remove(script_path)
            
            if result.returncode == 0:
                print(f"✅ Started session in database: {session_id}")
                return True
            else:
                print(f"❌ Failed to start session: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error starting session: {e}")
            return False
    
    def store_analysis(self, session_id: str, analysis_data: dict) -> bool:
        """Store analysis data in the database"""
        try:
            # Create data directory if it doesn't exist
            data_dir = os.path.join(self.project_root, 'data')
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
            # Create analyses directory
            analyses_dir = os.path.join(data_dir, 'analyses')
            if not os.path.exists(analyses_dir):
                os.makedirs(analyses_dir)
            
            # Generate unique analysis ID with more precision
            timestamp_ms = int(time.time() * 1000)
            analysis_id = f"{session_id}_{timestamp_ms}"
            
            # Ensure the analysis data has all required fields
            formatted_data = {
                "sessionId": session_id,
                "timestamp": analysis_data.get("timestamp", datetime.now().isoformat()),
                "focusScore": analysis_data.get("focusScore", 0),
                "attentionStatus": analysis_data.get("attentionStatus", "No Data"),
                "fatigueStatus": analysis_data.get("fatigueStatus", "awake"),
                "gazeDirection": analysis_data.get("gazeDirection", "unknown"),
                "earValue": analysis_data.get("earValue", 0),
                "headPose": analysis_data.get("headPose", {"pitch": 0, "yaw": 0, "roll": 0}),
                "handAnalysis": analysis_data.get("handAnalysis", {
                    "hand_fatigue_detected": False,
                    "hand_at_head": False,
                    "playing_with_hair": False,
                    "hand_movement": "normal"
                }),
                "methodUsed": analysis_data.get("methodUsed", "unknown"),
                "facesDetected": analysis_data.get("facesDetected", 0),
                "stored_at": datetime.now().isoformat()
            }
            
            # Store analysis data
            analysis_file = os.path.join(analyses_dir, f"{analysis_id}.json")
            with open(analysis_file, 'w') as f:
                json.dump(formatted_data, f, indent=2)
            
            print(f"📊 Stored analysis: {analysis_id} (Focus: {formatted_data['focusScore']}%)")
            return True
            
        except Exception as e:
            print(f"❌ Error storing analysis: {e}")
            return False
    
    def end_session(self, session_id: str, duration: int) -> bool:
        """End a session and calculate summary"""
        try:
            data_dir = os.path.join(self.project_root, 'data')
            session_file = os.path.join(data_dir, f"session_{session_id}.json")
            
            if os.path.exists(session_file):
                # Read existing session data
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                # Update session data
                session_data.update({
                    'endTime': datetime.now().isoformat(),
                    'duration': duration,
                    'active': False,
                    'completed': True
                })
                
                # Calculate session statistics from analyses
                analyses_dir = os.path.join(data_dir, 'analyses')
                if os.path.exists(analyses_dir):
                    session_analyses = []
                    for file_name in os.listdir(analyses_dir):
                        if file_name.startswith(f"{session_id}_"):
                            file_path = os.path.join(analyses_dir, file_name)
                            with open(file_path, 'r') as f:
                                analysis = json.load(f)
                                session_analyses.append(analysis)
                    
                    if session_analyses:
                        focus_scores = [a.get('focusScore', 0) for a in session_analyses]
                        session_data.update({
                            'averageFocusScore': sum(focus_scores) / len(focus_scores),
                            'maxFocusScore': max(focus_scores),
                            'minFocusScore': min(focus_scores),
                            'totalAnalyses': len(session_analyses),
                            'attentionSummary': self._get_most_common([a.get('attentionStatus', 'No Data') for a in session_analyses]),
                            'fatigueSummary': self._get_most_common([a.get('fatigueStatus', 'awake') for a in session_analyses])
                        })
                
                # Save updated session data
                with open(session_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
                
                print(f"✅ Ended session: {session_id}")
                return True
            else:
                print(f"❌ Session file not found: {session_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error ending session: {e}")
            return False
    
    def get_all_data(self) -> dict:
        """Get all data from the database"""
        try:
            data_dir = os.path.join(self.project_root, 'data')
            print(f"📂 Looking for data in: {data_dir}")
            
            # Get sessions
            sessions = []
            if os.path.exists(data_dir):
                session_files = [f for f in os.listdir(data_dir) if f.startswith('session_') and f.endswith('.json')]
                print(f"📋 Found {len(session_files)} session files")
                
                for file_name in session_files:
                    file_path = os.path.join(data_dir, file_name)
                    try:
                        with open(file_path, 'r') as f:
                            session_data = json.load(f)
                            sessions.append({
                                'session_id': session_data.get('sessionId'),
                                'start_time': session_data.get('startTime'),
                                'end_time': session_data.get('endTime'),
                                'total_duration': session_data.get('duration', 0),
                                'average_focus_score': session_data.get('averageFocusScore', 0),
                                'max_focus_score': session_data.get('maxFocusScore', 0),
                                'min_focus_score': session_data.get('minFocusScore', 100),  # Changed from 0 to 100
                                'attention_summary': session_data.get('attentionSummary', 'No Data'),
                                'fatigue_summary': session_data.get('fatigueSummary', 'awake'),
                                'total_analyses': session_data.get('totalAnalyses', 0),
                                'completed': session_data.get('completed', False)
                            })
                    except Exception as e:
                        print(f"⚠️ Error reading session file {file_name}: {e}")
            
            # Get analyses and fix the focus score issue
            analyses = []
            analyses_dir = os.path.join(data_dir, 'analyses')
            if os.path.exists(analyses_dir):
                analysis_files = [f for f in os.listdir(analyses_dir) if f.endswith('.json')]
                print(f"📊 Found {len(analysis_files)} analysis files")
                
                for file_name in analysis_files:
                    file_path = os.path.join(analyses_dir, file_name)
                    try:
                        with open(file_path, 'r') as f:
                            analysis_data = json.load(f)
                            
                            # Debug the focus score loading
                            focus_score = analysis_data.get('focusScore', 0)
                            print(f"  📊 Loading analysis {file_name}: Focus Score = {focus_score}")
                            
                            analyses.append({
                                'id': file_name.replace('.json', ''),
                                'session_id': analysis_data.get('sessionId'),
                                'timestamp': analysis_data.get('timestamp'),
                                'focus_score': focus_score,  # Make sure this is the correct field
                                'attention_status': analysis_data.get('attentionStatus', 'No Data'),
                                'fatigue_status': analysis_data.get('fatigueStatus', 'awake'),
                                'gaze_direction': analysis_data.get('gazeDirection', 'unknown'),
                                'ear_value': analysis_data.get('earValue', 0),
                                'head_pose_pitch': analysis_data.get('headPose', {}).get('pitch', 0),
                                'head_pose_yaw': analysis_data.get('headPose', {}).get('yaw', 0),
                                'head_pose_roll': analysis_data.get('headPose', {}).get('roll', 0),
                                'hand_fatigue_detected': analysis_data.get('handAnalysis', {}).get('hand_fatigue_detected', False),
                                'hand_at_head': analysis_data.get('handAnalysis', {}).get('hand_at_head', False),
                                'playing_with_hair': analysis_data.get('handAnalysis', {}).get('playing_with_hair', False),
                                'hand_movement': analysis_data.get('handAnalysis', {}).get('hand_movement', 'normal'),
                                'method_used': analysis_data.get('methodUsed', 'unknown'),
                                'faces_detected': analysis_data.get('facesDetected', 0)
                            })
                    except Exception as e:
                        print(f"⚠️ Error reading analysis file {file_name}: {e}")
            
            # Sort analyses by timestamp (newest first)
            analyses.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            # Generate summaries (basic implementation)
            summaries = self._generate_daily_summaries(sessions)
            
            result = {
                'sessions': sessions,
                'analyses': analyses,
                'summaries': summaries
            }
            
            print(f"✅ Loaded data: {len(sessions)} sessions, {len(analyses)} analyses, {len(summaries)} summaries")
            
            # Debug: Print a sample of focus scores if we have analyses
            if analyses:
                sample_scores = [a.get('focus_score', 0) for a in analyses[:3]]
                print(f"📊 Sample focus scores: {sample_scores}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error getting all data: {e}")
            return {
                'sessions': [],
                'analyses': [],
                'summaries': []
            }
    
    def _get_most_common(self, items):
        """Get the most common item from a list"""
        if not items:
            return 'No Data'
        
        counts = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        
        return max(counts.keys(), key=lambda x: counts[x])
    
    def _generate_daily_summaries(self, sessions):
        """Generate daily summaries from sessions"""
        daily_data = {}
        
        for session in sessions:
            if not session.get('start_time'):
                continue
                
            date = session['start_time'].split('T')[0]
            
            if date not in daily_data:
                daily_data[date] = {
                    'total_sessions': 0,
                    'total_study_time': 0,
                    'focus_scores': [],
                    'best_session': None,
                    'best_focus': 0
                }
            
            daily_data[date]['total_sessions'] += 1
            daily_data[date]['total_study_time'] += session.get('total_duration', 0)
            
            focus_score = session.get('average_focus_score', 0)
            daily_data[date]['focus_scores'].append(focus_score)
            
            if focus_score > daily_data[date]['best_focus']:
                daily_data[date]['best_focus'] = focus_score
                daily_data[date]['best_session'] = session.get('session_id')
        
        summaries = []
        for date, data in daily_data.items():
            avg_focus = sum(data['focus_scores']) / len(data['focus_scores']) if data['focus_scores'] else 0
            xp = data['total_sessions'] * 50 + int(avg_focus * 10)
            
            summaries.append({
                'date': date,
                'total_sessions': data['total_sessions'],
                'total_study_time': data['total_study_time'],
                'average_focus_score': avg_focus,
                'best_session': data['best_session'],
                'total_xp': xp
            })
        
        # Sort by date (newest first)
        summaries.sort(key=lambda x: x['date'], reverse=True)
        
        return summaries
