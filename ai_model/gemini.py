import os
import json
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("⚠️ google.generativeai nicht installiert. Bitte installieren: pip install google-generativeai")

from typing import Dict, Optional, List
from dotenv import load_dotenv

class GeminiAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        """
        Gemini AI Analyzer für Lernempfehlungen
        
        Args:
            api_key: Gemini API Key (optional, wird aus .env geladen wenn nicht angegeben)
        """
        if not GENAI_AVAILABLE:
            raise ImportError("google.generativeai ist nicht installiert. Bitte installieren: pip install google-generativeai")
        
        load_dotenv()
        
        # API Key laden
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY nicht gefunden. Bitte in .env Datei setzen oder als Parameter übergeben.")
        
        # Gemini konfigurieren
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')  # Günstigeres Modell statt gemini-pro
        
        print("🤖 Gemini Analyzer initialisiert (Flash-Modell für Effizienz)")
    
    def test_gemini_quick(self, focus_score: int = 75, attention: str = "focused") -> Dict:
        """
        🧪 TOKEN-SPARSAMER TEST für Gemini (nur ~50-100 Tokens)
        Kurzer Test mit minimalen Daten
        """
        if not hasattr(self, 'model'):
            return {"error": "Gemini nicht verfügbar"}
            
        try:
            # Minimaler Prompt für Test
            prompt = f"""Kurze Lernempfehlung für:
Focus: {focus_score}%
Attention: {attention}

Antworte in max 30 Wörtern auf Deutsch."""

            response = self.model.generate_content(
                prompt,
                generation_config={
                    'max_output_tokens': 50,  # Sehr begrenzt für Test
                    'temperature': 0.3
                }
            )
            
            return {
                "success": True,
                "test_recommendation": response.text,
                "tokens_used": "~50-100",
                "cost_estimate": "~0.001€"
            }
            
        except Exception as e:
            return {
                "error": f"Gemini Test fehlgeschlagen: {str(e)}",
                "success": False
            }
    
    def generate_learning_recommendations(self, trend_analysis: Dict) -> Dict:
        """
        Hauptmethode zur Generierung von Lernempfehlungen basierend auf Trendanalyse
        
        Args:
            trend_analysis: Analysedaten vom LearningAnalyzer
            
        Returns:
            Dict mit strukturierten Empfehlungen
        """
        try:
            prompt = self._create_learning_analysis_prompt(trend_analysis)
            
            response = self.model.generate_content(prompt)
            recommendations = self._parse_learning_response(response.text)
            
            print("✅ Gemini Empfehlungen erfolgreich generiert")
            return recommendations
            
        except Exception as e:
            print(f"❌ Fehler bei Gemini Analyse: {e}")
            return self._emergency_fallback_recommendations(trend_analysis)
    
    def _create_learning_analysis_prompt(self, analysis: Dict) -> str:
        """
        Detaillierten Prompt für Lernanalyse erstellen
        """
        overall_assessment = analysis.get('overall_assessment', {})
        fatigue_analysis = analysis.get('fatigue_analysis', {})
        attention_analysis = analysis.get('attention_analysis', {})
        learning_analysis = analysis.get('learning_analysis', {})
        hand_analysis = analysis.get('hand_analysis', {})
        gaze_analysis = analysis.get('gaze_analysis', {})
        head_pose_analysis = analysis.get('head_pose_analysis', {})
        
        prompt = f"""
Du bist ein KI-Experte für Lernoptimierung und Aufmerksamkeitsanalyse. Analysiere die folgenden detaillierten Lerndaten eines Studenten und erstelle präzise, wissenschaftlich fundierte Empfehlungen.

## ANALYSEDATEN ({analysis.get('timeframe_minutes', 5)} Minuten, {analysis.get('data_points', 0)} Datenpunkte):

### GESAMTBEWERTUNG:
- Gesamtscore: {overall_assessment.get('score', 0)}/100 ({overall_assessment.get('percentage', 0)}%)
- Status: {overall_assessment.get('status', 'unknown')}
- Komponenten: Müdigkeit {overall_assessment.get('components', {}).get('fatigue', 0)}/25, Aufmerksamkeit {overall_assessment.get('components', {}).get('attention', 0)}/25, Lernen {overall_assessment.get('components', {}).get('learning', 0)}/25, Stabilität {overall_assessment.get('components', {}).get('stability', 0)}/25

### MÜDIGKEITSANALYSE:
- Durchschnittlicher EAR (Augenöffnung): {fatigue_analysis.get('avg_ear', 0)}%
- Müdigkeitsepisoden: {fatigue_analysis.get('fatigue_episodes', 0)}
- Kritische Müdigkeit: {fatigue_analysis.get('critical_fatigue', 0)}
- EAR-Bereich: {fatigue_analysis.get('min_ear', 0)}% - {fatigue_analysis.get('max_ear', 0)}%

### AUFMERKSAMKEITSANALYSE:
- Aufmerksamkeitsrate: {attention_analysis.get('attention_ratio', 0):.1%}
- Ablenkungsepisoden: {attention_analysis.get('distraction_episodes', 0)}
- Aufmerksamkeitstrend: {attention_analysis.get('attention_trend', 'unknown')}

### LERNLEISTUNG:
- Durchschnittlicher Lernscore: {learning_analysis.get('avg_score', 0)}/100
- Score-Bereich: {learning_analysis.get('min_score', 0)} - {learning_analysis.get('max_score', 0)}
- Leistungstrend: {learning_analysis.get('score_trend', 'unknown')}
- Niedrige Leistung: {learning_analysis.get('low_score_episodes', 0)} Episoden
- Hohe Leistung: {learning_analysis.get('high_performance_ratio', 0):.1%}

### VERHALTENSANALYSE:
- Hand-Müdigkeitsdetektionen: {hand_analysis.get('fatigue_detections', 0)}
- Hand am Kopf: {hand_analysis.get('hand_at_head_frequency', 0)}x
- Haare spielen: {hand_analysis.get('hair_playing_frequency', 0)}x
- Unruhe-Level: {hand_analysis.get('restless_behavior', {}).get('level', 'low')}

### BLICKVERHALTEN:
- Zentraler Blick: {gaze_analysis.get('center_gaze_ratio', 0):.1%}
- Abgelenkter Blick: {gaze_analysis.get('distracted_gaze_ratio', 0):.1%}
- Blickstabilität: {gaze_analysis.get('gaze_stability', {}).get('level', 'unknown')}

### KÖRPERHALTUNG:
- Durchschnittliche Kopfneigung: Yaw {head_pose_analysis.get('avg_yaw', 0):.1f}°, Pitch {head_pose_analysis.get('avg_pitch', 0):.1f}°
- Stabile Haltung: {head_pose_analysis.get('stable_posture_ratio', 0):.1%}
- Bewegungsvarianz: {head_pose_analysis.get('head_movement_variance', 0):.2f}

## AUFGABE:
Erstelle detaillierte, personalisierte Empfehlungen in folgendem JSON-Format:

{{
  "urgency_level": "low|medium|high|critical",
  "overall_status": "excellent|good|concerning|critical",
  "confidence_score": 0.0-1.0,
  "user_recommendations": [
    {{
      "type": "immediate|short_term|long_term",
      "category": "break|posture|environment|technique|motivation",
      "title": "Kurzer prägnanter Titel",
      "message": "Klare, motivierende Empfehlung für den Nutzer",
      "scientific_basis": "Kurze wissenschaftliche Begründung",
      "priority": 1-5,
      "estimated_impact": "low|medium|high",
      "implementation_time": "Geschätzte Zeit zur Umsetzung"
    }}
  ],
  "app_actions": [
    {{
      "action": "notification|break_reminder|environment_adjustment|session_pause|focus_mode|study_technique_suggestion",
      "parameters": {{}},
      "timing": "immediate|delayed|scheduled",
      "reason": "Detaillierte Begründung für die App-Aktion",
      "priority": 1-5
    }}
  ],
  "insights": [
    "Wichtige, wissenschaftlich fundierte Erkenntnisse über das aktuelle Lernverhalten",
    "Langfristige Verhaltensmuster und deren Bedeutung",
    "Positive Aspekte, die verstärkt werden sollten"
  ],
  "learning_optimization": {{
    "current_efficiency": "Bewertung der aktuellen Lerneffizienz",
    "improvement_potential": "Verbesserungspotential in %",
    "recommended_study_duration": "Empfohlene verbleibende Lernzeit in Minuten",
    "optimal_break_timing": "Wann sollte die nächste Pause sein"
  }},
  "next_check_minutes": 5-30,
  "session_recommendation": "continue|short_break|long_break|end_session"
}}

## WISSENSCHAFTLICHE GRUNDLAGEN:
- EAR < 20%: Kritische Müdigkeit, sofortige Pause erforderlich
- EAR 20-25%: Moderate Müdigkeit, Pause empfohlen
- Aufmerksamkeit < 60%: Konzentrationsprobleme
- Unruhe-Verhalten: Zeichen von Langeweile oder Überforderung
- Kopfhaltung > 20° Abweichung: Ergonomische Probleme
- Optimale Lernblöcke: 25-45 Minuten mit 5-15 Minuten Pausen

## KOMMUNIKATIONSSTIL:
- Nutzerfreundlich und motivierend
- Wissenschaftlich fundiert aber verständlich
- Konkrete, umsetzbare Handlungsempfehlungen
- Positive Verstärkung einbauen
- Berücksichtigung der aktuellen Lernsituation

Antworte NUR mit dem JSON-Format, keine zusätzlichen Erklärungen.
"""
        return prompt
    
    def _parse_learning_response(self, response_text: str) -> Dict:
        """
        Gemini Response für Lernempfehlungen parsen und validieren
        """
        try:
            # JSON aus Response extrahieren
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("Kein JSON gefunden in Response")
            
            json_str = response_text[start_idx:end_idx]
            recommendations = json.loads(json_str)
            
            # Validierung und Defaults
            recommendations = self._validate_learning_recommendations(recommendations)
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ Fehler beim Parsen der Gemini Learning Response: {e}")
            return self._emergency_fallback_recommendations({})
    
    def _validate_learning_recommendations(self, recommendations: Dict) -> Dict:
        """
        Empfehlungsstruktur validieren und Defaults setzen
        """
        # Basis-Validierung
        recommendations.setdefault('urgency_level', 'medium')
        recommendations.setdefault('overall_status', 'good')
        recommendations.setdefault('confidence_score', 0.7)
        recommendations.setdefault('user_recommendations', [])
        recommendations.setdefault('app_actions', [])
        recommendations.setdefault('insights', [])
        recommendations.setdefault('next_check_minutes', 10)
        recommendations.setdefault('session_recommendation', 'continue')
        
        # Learning Optimization Defaults
        if 'learning_optimization' not in recommendations:
            recommendations['learning_optimization'] = {
                "current_efficiency": "Moderate Effizienz",
                "improvement_potential": "15-25%",
                "recommended_study_duration": "20-30 Minuten",
                "optimal_break_timing": "In 15-20 Minuten"
            }
        
        # User Recommendations validieren
        for rec in recommendations['user_recommendations']:
            rec.setdefault('type', 'short_term')
            rec.setdefault('category', 'technique')
            rec.setdefault('title', 'Lernempfehlung')
            rec.setdefault('message', 'Verbessern Sie Ihre Lernstrategie')
            rec.setdefault('scientific_basis', 'Basierend auf Lernforschung')
            rec.setdefault('priority', 3)
            rec.setdefault('estimated_impact', 'medium')
            rec.setdefault('implementation_time', '5-10 Minuten')
        
        # App Actions validieren
        for action in recommendations['app_actions']:
            action.setdefault('action', 'notification')
            action.setdefault('parameters', {})
            action.setdefault('timing', 'delayed')
            action.setdefault('reason', 'Verbesserung der Lerneffizienz')
            action.setdefault('priority', 3)
        
        return recommendations
    
    def generate_motivation_boost(self, current_score: float, trend: str) -> Dict:
        """
        Motivations-Boost basierend auf aktuellem Score und Trend generieren
        """
        prompt = f"""
Du bist ein Lerncoach. Ein Student hat aktuell einen Lernscore von {current_score}/100 mit Trend "{trend}".

Erstelle eine motivierende, personalisierte Nachricht im JSON-Format:

{{
  "motivation_message": "Ermutigende, spezifische Nachricht",
  "achievement_recognition": "Was der Student gut macht",
  "improvement_suggestion": "Eine konkrete Verbesserung",
  "confidence_boost": "Positive Verstärkung"
}}

Sei spezifisch, ermutigend und konstruktiv.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_simple_json_response(response.text)
        except Exception as e:
            print(f"❌ Fehler bei Motivations-Generierung: {e}")
            return {
                "motivation_message": "Sie machen gute Fortschritte! Bleiben Sie fokussiert.",
                "achievement_recognition": f"Ihr aktueller Score von {current_score} zeigt Ihr Engagement.",
                "improvement_suggestion": "Kleine, regelmäßige Pausen können Ihre Leistung steigern.",
                "confidence_boost": "Jeder Lernschritt bringt Sie Ihrem Ziel näher!"
            }
    
    def analyze_learning_pattern(self, historical_data: List[Dict]) -> Dict:
        """
        Langfristige Lernmuster analysieren
        """
        if len(historical_data) < 5:
            return {"status": "insufficient_data"}
        
        # Vereinfachte Analyse für den Prompt
        avg_scores = [d.get('learning_score', 0) for d in historical_data[-10:]]
        avg_attention = [d.get('attention_ratio', 0) for d in historical_data[-10:]]
        
        prompt = f"""
Analysiere das Lernmuster eines Studenten basierend auf den letzten Daten:

Lernscores: {avg_scores}
Aufmerksamkeitsraten: {avg_attention}

Erstelle eine Musteranalyse im JSON-Format:

{{
  "pattern_type": "consistent|improving|declining|variable",
  "strengths": ["Stärken des Lerners"],
  "weaknesses": ["Verbesserungsbereiche"],
  "optimal_study_times": "Empfohlene Lernzeiten",
  "personalized_strategy": "Individuelle Lernstrategie",
  "long_term_recommendations": ["Langfristige Empfehlungen"]
}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_simple_json_response(response.text)
        except Exception as e:
            print(f"❌ Fehler bei Musteranalyse: {e}")
            return {"status": "analysis_failed"}
    
    def _parse_simple_json_response(self, response_text: str) -> Dict:
        """
        Einfache JSON Response parsen
        """
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
        except:
            return {"status": "parse_failed"}
    
    def _emergency_fallback_recommendations(self, analysis: Dict) -> Dict:
        """
        Notfall-Empfehlungen wenn Gemini komplett fehlschlägt
        """
        return {
            "urgency_level": "medium",
            "overall_status": "unknown",
            "confidence_score": 0.5,
            "user_recommendations": [
                {
                    "type": "immediate",
                    "category": "break",
                    "title": "Kurze Pause",
                    "message": "Machen Sie eine 5-10 Minuten Pause um sich zu erholen.",
                    "scientific_basis": "Regelmäßige Pausen verbessern die Konzentration",
                    "priority": 3,
                    "estimated_impact": "medium",
                    "implementation_time": "5-10 Minuten"
                }
            ],
            "app_actions": [
                {
                    "action": "break_reminder",
                    "parameters": {"duration_minutes": 10},
                    "timing": "immediate",
                    "reason": "Fallback-Empfehlung für bessere Lerneffizienz",
                    "priority": 3
                }
            ],
            "insights": [
                "Automatische Fallback-Analyse aufgrund technischer Probleme",
                "Regelmäßige Pausen sind essentiell für optimales Lernen"
            ],
            "learning_optimization": {
                "current_efficiency": "Unbekannt - technische Probleme",
                "improvement_potential": "Potentiell 10-20%",
                "recommended_study_duration": "15-25 Minuten",
                "optimal_break_timing": "Jetzt"
            },
            "next_check_minutes": 10,
            "session_recommendation": "short_break"
        }
    
    def test_connection(self) -> bool:
        """
        Gemini-Verbindung testen
        """
        try:
            response = self.model.generate_content("Antworte mit 'OK' wenn du mich hörst.")
            return "OK" in response.text.upper()
        except Exception as e:
            print(f"❌ Gemini Verbindungstest fehlgeschlagen: {e}")
            return False

# Test-Funktion für direkte Verwendung
if __name__ == "__main__":
    try:
        analyzer = GeminiAnalyzer()
        
        # Verbindungstest
        if analyzer.test_connection():
            print("✅ Gemini Verbindung erfolgreich!")
        else:
            print("❌ Gemini Verbindung fehlgeschlagen!")
        
        # Test mit Beispieldaten
        test_analysis = {
            "timeframe_minutes": 5,
            "data_points": 15,
            "overall_assessment": {"score": 65, "status": "good"},
            "fatigue_analysis": {"avg_ear": 22, "fatigue_episodes": 2},
            "attention_analysis": {"attention_ratio": 0.7, "attention_trend": "stable"},
            "learning_analysis": {"avg_score": 68, "score_trend": "improving"}
        }
        
        recommendations = analyzer.generate_learning_recommendations(test_analysis)
        print("🎯 Test-Empfehlungen generiert:")
        print(f"Status: {recommendations.get('overall_status')}")
        print(f"Empfehlungen: {len(recommendations.get('user_recommendations', []))}")
        
    except Exception as e:
        print(f"❌ Test fehlgeschlagen: {e}")
        print("Bitte GEMINI_API_KEY in .env Datei setzen!")