import time
import threading
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import statistics
import json
import sys
import os

# Füge das übergeordnete Verzeichnis zum Path hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class AnalysisSnapshot:
    timestamp: datetime
    ear: float
    attention_status: str
    gaze_left: str
    gaze_right: str
    head_pose: Dict[str, float]
    hand_analysis: Dict[str, bool]
    learning_score: int

class LearningAnalyzer:
    def __init__(self, analysis_window_minutes: int = 5, 
                 recommendation_interval_minutes: int = 10):
        """
        Kontinuierlicher Lernanalyzer
        
        Args:
            analysis_window_minutes: Zeitfenster für Datensammlung (Standard: 5 min)
            recommendation_interval_minutes: Intervall für Empfehlungen (Standard: 10 min)
        """
        # Konfiguration
        self.analysis_window = timedelta(minutes=analysis_window_minutes)
        self.recommendation_interval = timedelta(minutes=recommendation_interval_minutes)
        
        # Datensammlung
        self.data_buffer = deque(maxlen=1000)  # Letzte 1000 Datenpunkte
        self.last_recommendation_time = datetime.now()
        
        # Threading
        self.is_running = False
        self.analysis_thread = None
        
        # Callbacks
        self.recommendation_callback = None
        
        # Gemini Integration (wird von außen gesetzt)
        self.gemini_analyzer = None
        
        print("📊 Learning Analyzer initialisiert")

    def set_gemini_analyzer(self, gemini_analyzer):
        """Gemini Analyzer setzen"""
        self.gemini_analyzer = gemini_analyzer
        print("🤖 Gemini Analyzer verbunden")

    def add_analysis_data(self, analysis_result: Dict):
        """Neue Analysedaten hinzufügen"""
        try:
            snapshot = AnalysisSnapshot(
                timestamp=datetime.now(),
                ear=analysis_result.get('avgEAR', 0),
                attention_status=analysis_result.get('attention', 'unknown'),
                gaze_left=analysis_result.get('gazeLeft', 'unknown'),
                gaze_right=analysis_result.get('gazeRight', 'unknown'),
                head_pose=analysis_result.get('headPose', {}),
                hand_analysis=analysis_result.get('handAnalysis', {}),
                learning_score=analysis_result.get('lernfaehigkeitsScore', 0)
            )
            
            self.data_buffer.append(snapshot)
            print(f"📊 Daten hinzugefügt: Score={snapshot.learning_score}, Attention={snapshot.attention_status}")
            
        except Exception as e:
            print(f"❌ Fehler beim Hinzufügen der Daten: {e}")

    def analyze_trends(self) -> Dict:
        """Trends und Muster in den gesammelten Daten analysieren"""
        if len(self.data_buffer) < 10:
            return {"status": "insufficient_data", "message": "Nicht genügend Daten für Analyse"}
        
        # Zeitfenster für Analyse
        cutoff_time = datetime.now() - self.analysis_window
        recent_data = [d for d in self.data_buffer if d.timestamp > cutoff_time]
        
        if not recent_data:
            return {"status": "no_recent_data"}
        
        # Statistische Auswertung
        analysis = {
            "timeframe_minutes": self.analysis_window.total_seconds() / 60,
            "data_points": len(recent_data),
            "timestamp": datetime.now().isoformat(),
            
            # EAR (Müdigkeit) Trends
            "fatigue_analysis": {
                "avg_ear": round(statistics.mean([d.ear for d in recent_data]), 2),
                "min_ear": min([d.ear for d in recent_data]),
                "max_ear": max([d.ear for d in recent_data]),
                "fatigue_episodes": len([d for d in recent_data if d.ear < 25.0]),
                "critical_fatigue": len([d for d in recent_data if d.ear < 20.0])
            },
            
            # Aufmerksamkeits-Trends
            "attention_analysis": {
                "attention_ratio": len([d for d in recent_data if d.attention_status == "aufmerksam"]) / len(recent_data),
                "distraction_episodes": len([d for d in recent_data if d.attention_status == "abgelenkt"]),
                "attention_trend": self._calculate_attention_trend(recent_data)
            },
            
            # Lernfähigkeits-Score
            "learning_analysis": {
                "avg_score": round(statistics.mean([d.learning_score for d in recent_data]), 1),
                "min_score": min([d.learning_score for d in recent_data]),
                "max_score": max([d.learning_score for d in recent_data]),
                "score_trend": self._calculate_trend([d.learning_score for d in recent_data[-10:]]),
                "low_score_episodes": len([d for d in recent_data if d.learning_score < 50]),
                "high_performance_ratio": len([d for d in recent_data if d.learning_score >= 80]) / len(recent_data)
            },
            
            # Hand-Analyse
            "hand_analysis": {
                "fatigue_detections": len([d for d in recent_data if d.hand_analysis.get('hand_fatigue_detected', False)]),
                "hand_at_head_frequency": len([d for d in recent_data if d.hand_analysis.get('hand_at_head', False)]),
                "hair_playing_frequency": len([d for d in recent_data if d.hand_analysis.get('playing_with_hair', False)]),
                "restless_behavior": self._analyze_restless_behavior(recent_data)
            },
            
            # Blickrichtung
            "gaze_analysis": {
                "center_gaze_ratio": len([d for d in recent_data if d.gaze_left == "center" and d.gaze_right == "center"]) / len(recent_data),
                "distracted_gaze_ratio": len([d for d in recent_data if d.gaze_left != "center" or d.gaze_right != "center"]) / len(recent_data),
                "gaze_stability": self._analyze_gaze_stability(recent_data)
            },
            
            # Kopfpose-Analyse
            "head_pose_analysis": {
                "avg_yaw": statistics.mean([abs(d.head_pose.get('yaw', 0)) for d in recent_data]),
                "avg_pitch": statistics.mean([abs(d.head_pose.get('pitch', 0)) for d in recent_data]),
                "head_movement_variance": self._calculate_head_movement_variance(recent_data),
                "stable_posture_ratio": len([d for d in recent_data if abs(d.head_pose.get('yaw', 0)) <= 15 and abs(d.head_pose.get('pitch', 0)) <= 15]) / len(recent_data)
            }
        }
        
        # Gesamtbewertung
        analysis["overall_assessment"] = self._calculate_overall_assessment(analysis)
        
        return analysis

    def _calculate_trend(self, values: List[float]) -> str:
        """Einfache Trendberechnung"""
        if len(values) < 3:
            return "insufficient_data"
        
        first_half = statistics.mean(values[:len(values)//2])
        second_half = statistics.mean(values[len(values)//2:])
        
        diff = second_half - first_half
        if diff > 5:
            return "improving"
        elif diff < -5:
            return "declining"
        else:
            return "stable"

    def _calculate_attention_trend(self, recent_data: List[AnalysisSnapshot]) -> str:
        """Aufmerksamkeitstrend berechnen"""
        if len(recent_data) < 6:
            return "insufficient_data"
        
        # Aufmerksamkeit in zwei Hälften teilen
        mid_point = len(recent_data) // 2
        first_half_attention = len([d for d in recent_data[:mid_point] if d.attention_status == "aufmerksam"]) / mid_point
        second_half_attention = len([d for d in recent_data[mid_point:] if d.attention_status == "aufmerksam"]) / (len(recent_data) - mid_point)
        
        diff = second_half_attention - first_half_attention
        if diff > 0.2:
            return "improving"
        elif diff < -0.2:
            return "declining"
        else:
            return "stable"

    def _analyze_restless_behavior(self, recent_data: List[AnalysisSnapshot]) -> Dict:
        """Unruhiges Verhalten analysieren"""
        total_restless_actions = 0
        for d in recent_data:
            if d.hand_analysis.get('hand_at_head', False):
                total_restless_actions += 1
            if d.hand_analysis.get('playing_with_hair', False):
                total_restless_actions += 1
        
        restless_ratio = total_restless_actions / len(recent_data) if recent_data else 0
        
        return {
            "total_actions": total_restless_actions,
            "restless_ratio": restless_ratio,
            "level": "high" if restless_ratio > 0.3 else "medium" if restless_ratio > 0.1 else "low"
        }

    def _analyze_gaze_stability(self, recent_data: List[AnalysisSnapshot]) -> Dict:
        """Blickstabilität analysieren"""
        gaze_changes = 0
        prev_gaze = None
        
        for d in recent_data:
            current_gaze = f"{d.gaze_left}_{d.gaze_right}"
            if prev_gaze and prev_gaze != current_gaze:
                gaze_changes += 1
            prev_gaze = current_gaze
        
        stability_ratio = 1 - (gaze_changes / len(recent_data)) if recent_data else 0
        
        return {
            "changes": gaze_changes,
            "stability_ratio": stability_ratio,
            "level": "high" if stability_ratio > 0.8 else "medium" if stability_ratio > 0.6 else "low"
        }

    def _calculate_head_movement_variance(self, recent_data: List[AnalysisSnapshot]) -> float:
        """Kopfbewegungsvarianz berechnen"""
        if len(recent_data) < 2:
            return 0
        
        yaw_values = [d.head_pose.get('yaw', 0) for d in recent_data]
        pitch_values = [d.head_pose.get('pitch', 0) for d in recent_data]
        
        yaw_variance = statistics.variance(yaw_values) if len(yaw_values) > 1 else 0
        pitch_variance = statistics.variance(pitch_values) if len(pitch_values) > 1 else 0
        
        return (yaw_variance + pitch_variance) / 2

    def _calculate_overall_assessment(self, analysis: Dict) -> Dict:
        """Gesamtbewertung berechnen"""
        score = 0
        max_score = 100
        
        # Müdigkeit (25 Punkte)
        fatigue_score = min(25, analysis['fatigue_analysis']['avg_ear'] / 4)
        score += fatigue_score
        
        # Aufmerksamkeit (25 Punkte)
        attention_score = analysis['attention_analysis']['attention_ratio'] * 25
        score += attention_score
        
        # Lernscore (25 Punkte)
        learning_score = analysis['learning_analysis']['avg_score'] / 4
        score += learning_score
        
        # Stabilität (25 Punkte)
        stability_score = analysis['head_pose_analysis']['stable_posture_ratio'] * 25
        score += stability_score
        
        # Status bestimmen
        if score >= 80:
            status = "excellent"
        elif score >= 60:
            status = "good"
        elif score >= 40:
            status = "concerning"
        else:
            status = "critical"
        
        return {
            "score": round(score, 1),
            "max_score": max_score,
            "percentage": round((score / max_score) * 100, 1),
            "status": status,
            "components": {
                "fatigue": round(fatigue_score, 1),
                "attention": round(attention_score, 1),
                "learning": round(learning_score, 1),
                "stability": round(stability_score, 1)
            }
        }

    def generate_recommendations(self, trend_analysis: Dict) -> Dict:
        """Empfehlungen generieren (über Gemini oder Fallback)"""
        try:
            if self.gemini_analyzer:
                print("🤖 Generiere Empfehlungen mit Gemini...")
                return self.gemini_analyzer.generate_learning_recommendations(trend_analysis)
            else:
                print("⚠️ Gemini nicht verfügbar - verwende Fallback-Empfehlungen")
                return self._fallback_recommendations(trend_analysis)
                
        except Exception as e:
            print(f"❌ Fehler bei Empfehlungsgenerierung: {e}")
            return self._fallback_recommendations(trend_analysis)

    def _fallback_recommendations(self, analysis: Dict) -> Dict:
        """Fallback-Empfehlungen wenn Gemini nicht verfügbar"""
        if not analysis or analysis.get("status") in ["insufficient_data", "no_recent_data"]:
            return {
                "urgency_level": "low",
                "overall_status": "unknown",
                "user_recommendations": [
                    {
                        "type": "immediate",
                        "category": "break",
                        "message": "Machen Sie eine kurze Pause von 5 Minuten",
                        "priority": 3
                    }
                ],
                "app_actions": [],
                "insights": ["Keine ausreichenden Daten für detaillierte Analyse"],
                "next_check_minutes": 5
            }
        
        user_recs = []
        app_actions = []
        insights = []
        urgency = "low"
        overall_status = "good"
        
        # Müdigkeitsanalyse
        fatigue_avg = analysis.get('fatigue_analysis', {}).get('avg_ear', 100)
        if fatigue_avg < 20:
            urgency = "critical"
            overall_status = "critical"
            user_recs.append({
                "type": "immediate",
                "category": "break",
                "message": "SOFORT PAUSE! Sie sind stark übermüdet. 15-20 Minuten Pause empfohlen.",
                "priority": 5
            })
            app_actions.append({
                "action": "session_end",
                "parameters": {"reason": "critical_fatigue"},
                "timing": "immediate",
                "reason": "Kritische Müdigkeit erkannt"
            })
        elif fatigue_avg < 25:
            urgency = "high"
            overall_status = "concerning"
            user_recs.append({
                "type": "immediate",
                "category": "break",
                "message": "Sie wirken müde. Machen Sie eine 10-15 Minuten Pause.",
                "priority": 4
            })
            app_actions.append({
                "action": "break_reminder",
                "parameters": {"duration_minutes": 15},
                "timing": "immediate",
                "reason": "Müdigkeit erkannt"
            })
        
        # Aufmerksamkeitsanalyse
        attention_ratio = analysis.get('attention_analysis', {}).get('attention_ratio', 1)
        if attention_ratio < 0.5:
            if urgency == "low":
                urgency = "medium"
            user_recs.append({
                "type": "immediate",
                "category": "environment",
                "message": "Ihre Aufmerksamkeit lässt nach. Reduzieren Sie Ablenkungen und fokussieren Sie sich neu.",
                "priority": 4
            })
            insights.append("Niedrige Aufmerksamkeitsrate erkannt")
        
        # Hand-Analyse
        restless_level = analysis.get('hand_analysis', {}).get('restless_behavior', {}).get('level', 'low')
        if restless_level == "high":
            user_recs.append({
                "type": "short_term",
                "category": "technique",
                "message": "Sie zeigen Anzeichen von Unruhe. Versuchen Sie bewusste Entspannungsübungen.",
                "priority": 3
            })
            insights.append("Erhöhte Unruhe durch Handbewegungen erkannt")
        
        # Lernscore-Analyse
        avg_score = analysis.get('learning_analysis', {}).get('avg_score', 70)
        if avg_score < 40:
            user_recs.append({
                "type": "long_term",
                "category": "technique",
                "message": "Ihr Lernscore ist niedrig. Überprüfen Sie Ihre Lernstrategie und Umgebung.",
                "priority": 3
            })
        
        # Postur-Analyse
        stable_posture = analysis.get('head_pose_analysis', {}).get('stable_posture_ratio', 1)
        if stable_posture < 0.6:
            user_recs.append({
                "type": "immediate",
                "category": "posture",
                "message": "Achten Sie auf eine aufrechte, stabile Sitzhaltung.",
                "priority": 2
            })
        
        return {
            "urgency_level": urgency,
            "overall_status": overall_status,
            "user_recommendations": user_recs,
            "app_actions": app_actions,
            "insights": insights,
            "next_check_minutes": 5 if urgency == "critical" else 10 if urgency == "high" else 15
        }

    def start_continuous_analysis(self, recommendation_callback: Callable = None):
        """Kontinuierliche Analyse starten"""
        self.recommendation_callback = recommendation_callback
        self.is_running = True
        
        self.analysis_thread = threading.Thread(target=self._analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
        print("🚀 Kontinuierliche Analyse gestartet")

    def stop_continuous_analysis(self):
        """Kontinuierliche Analyse stoppen"""
        self.is_running = False
        if self.analysis_thread:
            self.analysis_thread.join()
        print("⏹️ Analyse gestoppt")

    def _analysis_loop(self):
        """Hauptschleife für kontinuierliche Analyse"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Prüfen ob es Zeit für neue Empfehlungen ist
                if current_time - self.last_recommendation_time >= self.recommendation_interval:
                    print("🔍 Führe Trendanalyse durch...")
                    
                    # Trends analysieren
                    trend_analysis = self.analyze_trends()
                    
                    if trend_analysis.get("status") not in ["insufficient_data", "no_recent_data"]:
                        # Empfehlungen generieren
                        recommendations = self.generate_recommendations(trend_analysis)
                        
                        # Callback ausführen
                        if self.recommendation_callback:
                            self.recommendation_callback(recommendations, trend_analysis)
                        
                        self.last_recommendation_time = current_time
                        
                        # Nächsten Check basierend auf Empfehlung planen
                        next_check = recommendations.get('next_check_minutes', 10)
                        self.recommendation_interval = timedelta(minutes=next_check)
                        
                        print(f"✅ Empfehlungen generiert. Nächster Check in {next_check} Minuten")
                    else:
                        print(f"⏸️ Überspringe Analyse: {trend_analysis.get('message', 'Unbekannter Grund')}")
                
                # Kurz warten bevor nächste Prüfung
                time.sleep(30)  # Prüfe alle 30 Sekunden
                
            except Exception as e:
                print(f"❌ Fehler in Analyseschleife: {e}")
                time.sleep(60)  # Bei Fehler 1 Minute warten

    def get_current_status(self) -> Dict:
        """Aktuellen Status der Analyse abrufen"""
        next_recommendation_seconds = max(0, (self.last_recommendation_time + self.recommendation_interval - datetime.now()).total_seconds())
        
        return {
            "is_running": self.is_running,
            "data_points_collected": len(self.data_buffer),
            "last_recommendation": self.last_recommendation_time.isoformat(),
            "next_recommendation_in_minutes": round(next_recommendation_seconds / 60, 1),
            "analysis_window_minutes": self.analysis_window.total_seconds() / 60,
            "gemini_connected": self.gemini_analyzer is not None
        }

    def force_analysis(self) -> Dict:
        """Sofortige Analyse erzwingen (für Tests/Debug)"""
        print("🔧 Erzwinge sofortige Analyse...")
        trend_analysis = self.analyze_trends()
        
        if trend_analysis.get("status") not in ["insufficient_data", "no_recent_data"]:
            recommendations = self.generate_recommendations(trend_analysis)
            return {
                "trend_analysis": trend_analysis,
                "recommendations": recommendations
            }
        else:
            return {
                "error": "Nicht genügend Daten für Analyse",
                "status": trend_analysis.get("status")
            } 
