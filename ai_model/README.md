# AI-Enhanced Learning Analytics

Dieses System kombiniert Computer Vision mit künstlicher Intelligenz für intelligente Lernanalyse und -optimierung.

## Features

### 🎯 Kern-Funktionen
- **Müdigkeitserkennung**: Eye Aspect Ratio (EAR) Analyse
- **Aufmerksamkeits-Tracking**: Blickrichtung und Kopfpose
- **Hand-Gesture Analyse**: Erkennung von Müdigkeitszeichen
- **Lernfähigkeits-Score**: Intelligente Bewertung der Lerneffizienz

### 🤖 AI Agent Integration
- **Kontinuierliche Analyse**: Sammelt Daten über längere Zeiträume
- **Trend-Erkennung**: Erkennt Muster im Lernverhalten
- **Gemini AI Integration**: Generiert personalisierte Empfehlungen
- **Motivations-Boost**: KI-gestützte Ermutigung

## Installation

### 1. Dependencies installieren
```bash
pip install google-generativeai python-dotenv flask flask-cors opencv-python numpy dlib imutils mediapipe
```

### 2. Gemini API Key einrichten
1. Gehe zu [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Erstelle einen neuen API Key
3. Füge ihn in die `.env` Datei ein:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Dlib Modelle herunterladen
```bash
# shape_predictor_68_face_landmarks.dat muss im ai_model/ Verzeichnis sein
# Download von: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

## API Endpoints

### Core Analysis
- `POST /api/analyze` - Hauptanalyse (Bild → Analysedaten)
- `GET /api/health` - System-Status

### AI Agent
- `GET /api/recommendations` - Aktuelle AI-Empfehlungen
- `GET /api/agent-status` - AI Agent Status
- `POST /api/force-analysis` - Sofortige Analyse erzwingen
- `GET /api/motivation` - Motivations-Boost abrufen

## Verwendung

### Server starten
```bash
cd ai_model
python combined.py
```

### API Beispiel
```javascript
// Bild zur Analyse senden
const response = await fetch('http://localhost:5000/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: base64Image })
});

// AI-Empfehlungen abrufen
const recommendations = await fetch('http://localhost:5000/api/recommendations');
```

## AI Agent Workflow

1. **Datensammlung**: Kontinuierliche Sammlung von Analysedaten
2. **Trend-Analyse**: Auswertung von Mustern über 5-10 Minuten
3. **Gemini Integration**: KI-basierte Empfehlungsgenerierung
4. **Empfehlungs-Delivery**: Strukturierte Ausgabe für Nutzer und App

### Empfehlungsstruktur
```json
{
  "urgency_level": "low|medium|high|critical",
  "overall_status": "excellent|good|concerning|critical",
  "user_recommendations": [
    {
      "type": "immediate|short_term|long_term",
      "category": "break|posture|environment|technique",
      "message": "Konkrete Handlungsempfehlung",
      "priority": 1-5
    }
  ],
  "app_actions": [
    {
      "action": "break_reminder|notification|session_end",
      "timing": "immediate|delayed|scheduled",
      "reason": "Begründung für die Aktion"
    }
  ],
  "learning_optimization": {
    "current_efficiency": "Bewertung der Lerneffizienz",
    "improvement_potential": "Verbesserungspotential",
    "recommended_study_duration": "Empfohlene Lernzeit"
  }
}
```

## Konfiguration

### AI Agent Parameter (in `.env`)
```env
AI_ANALYSIS_WINDOW_MINUTES=5          # Datensammlungs-Zeitfenster
AI_RECOMMENDATION_INTERVAL_MINUTES=10 # Empfehlungs-Intervall
```

### Schwellenwerte
- **EAR < 20%**: Kritische Müdigkeit → Sofortige Pause
- **EAR 20-25%**: Moderate Müdigkeit → Pause empfohlen
- **Aufmerksamkeit < 60%**: Konzentrationsprobleme
- **Lernscore < 40**: Niedrige Effizienz

## Troubleshooting

### Häufige Probleme

1. **"AI Agent nicht verfügbar"**
   - Prüfe ob `agents/analyse.py` und `gemini.py` existieren
   - Prüfe Import-Pfade

2. **"Gemini API key not valid"**
   - Prüfe API Key in `.env` Datei
   - Stelle sicher dass der Key gültig ist

3. **"Dlib models not found"**
   - Lade `shape_predictor_68_face_landmarks.dat` herunter
   - Platziere es im `ai_model/` Verzeichnis

4. **"MediaPipe not available"**
   - Installiere MediaPipe: `pip install mediapipe`
   - System läuft auch ohne MediaPipe (nur mit dlib)

## Entwicklung

### Neue AI-Features hinzufügen
1. Erweitere `LearningAnalyzer.analyze_trends()`
2. Passe Gemini-Prompts in `gemini.py` an
3. Teste mit `/api/force-analysis`

### Custom Empfehlungen
- Bearbeite `_fallback_recommendations()` für Offline-Modus
- Erweitere Gemini-Prompts für spezifische Szenarien

## Architektur

```
ai_model/
├── combined.py           # Haupt-API Server
├── gemini.py            # Gemini AI Integration
├── agents/
│   └── analyse.py       # Learning Analyzer
├── .env                 # Konfiguration
└── README.md           # Diese Datei
```

## Performance

- **Analyse-Zyklen**: Alle 30 Sekunden überprüft
- **Empfehlungen**: Alle 5-15 Minuten (adaptiv)
- **Daten-Buffer**: Letzten 1000 Datenpunkte
- **Gemini Calls**: Nur bei signifikanten Veränderungen

## Lizenz

Dieses Projekt ist für Lern- und Forschungszwecke entwickelt.
