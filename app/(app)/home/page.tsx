"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import {
  Play,
  Pause,
  Square,
  Camera,
  Mic,
  MicOff,
  Settings as SettingsIcon,
  Brain,
  Clock,
  Target,
  TrendingUp,
  Coffee,
  BookOpen,
  Zap,
  RotateCcw,
  BarChart3,
  Trophy,
  Star,
  Flame,
  Users,
  Award,
  Gift,
  Sparkles,
  Rocket,
  Crown,
  Gamepad2,
  Lightbulb,
  Heart,
  ChevronRight,
  Plus
} from "lucide-react"
import { FocusChart } from "@/components/ui/FocusChart"

export default function StudyCompanion() {
  const router = useRouter()
  const [isRecording, setIsRecording] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [isMuted, setIsMuted] = useState(false)
  const [studyTime, setStudyTime] = useState(0)
  const [sessionTime, setSessionTime] = useState(0) // New session time state
  const [sessionProgress, setSessionProgress] = useState(0)
  const [currentSuggestion, setCurrentSuggestion] = useState("")
  
  // AI Analysis Data
  const [focusScore, setFocusScore] = useState(0)
  const [attentionStatus, setAttentionStatus] = useState("No Data")
  const [fatigueStatus, setFatigueStatus] = useState("awake")
  const [gazeDirection, setGazeDirection] = useState("unknown")
  const [headPose, setHeadPose] = useState({ pitch: 0, yaw: 0, roll: 0 })
  const [handAnalysis, setHandAnalysis] = useState({
    hand_fatigue_detected: false,
    hand_at_head: false,
    playing_with_hair: false,
    hand_movement: 'normal'
  })
  const [earValue, setEarValue] = useState(0)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [geminiTestResult, setGeminiTestResult] = useState<string | null>(null)
  const [isGeminiTesting, setIsGeminiTesting] = useState(false)
  const [finalAnalysis, setFinalAnalysis] = useState<{
    lernfaehigkeitsScore?: number
    attention?: string
    status?: string
    avgEAR?: number
    gazeLeft?: string
    gazeRight?: string
    headPose?: { pitch: number, yaw: number, roll: number }
    handAnalysis?: {
      hand_fatigue_detected?: boolean
      hand_at_head?: boolean
      playing_with_hair?: boolean
      hand_movement?: string
    }
  } | null>(null)

  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const analysisIntervalRef = useRef<NodeJS.Timeout | null>(null)

  // Final Analysis Function for frozen frame
  const performFinalAnalysis = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return

    setIsAnalyzing(true)
    
    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    
    if (!ctx) {
      setIsAnalyzing(false)
      return
    }

    // Set canvas size to match video
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Draw final video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    // Convert canvas to base64 image
    const imageDataURL = canvas.toDataURL('image/jpeg', 0.9)

    try {
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageDataURL
        })
      })

      if (response.ok) {
        const data = await response.json()
        
        // Store final analysis
        setFinalAnalysis(data)
        
        // Update current states with final analysis
        setFocusScore(data.lernfaehigkeitsScore || 0)
        setAttentionStatus(data.attention || "No Data")
        setFatigueStatus(data.status || "awake")
        // Fix gaze direction for mirrored camera (swap left and right)
        setGazeDirection(`${data.gazeRight || "unknown"}/${data.gazeLeft || "unknown"}`)
        setHeadPose(data.headPose || { pitch: 0, yaw: 0, roll: 0 })
        setHandAnalysis(data.handAnalysis || {
          hand_fatigue_detected: false,
          hand_at_head: false,
          playing_with_hair: false,
          hand_movement: 'normal'
        })
        setEarValue(data.avgEAR || 0)

        // Generate final AI suggestion
        generateFinalSuggestion(data)
      }
    } catch (error) {
      console.error('Final Analysis Error:', error)
      setCurrentSuggestion("Analysis completed. Session ended successfully.")
    } finally {
      setIsAnalyzing(false)
    }
  }, [])

  // Generate final suggestions based on analysis
  const generateFinalSuggestion = (data: {
    status?: string
    attention?: string
    lernfaehigkeitsScore?: number
    handAnalysis?: {
      hand_fatigue_detected?: boolean
      playing_with_hair?: boolean
    }
  }) => {
    const score = data.lernfaehigkeitsScore || 0
    
    if (score >= 90) {
      setCurrentSuggestion("🎉 Excellent session! You maintained outstanding focus throughout.")
    } else if (score >= 75) {
      setCurrentSuggestion("✅ Great session! Your focus was very good overall.")
    } else if (score >= 60) {
      setCurrentSuggestion("👍 Good session! There's room for improvement in maintaining focus.")
    } else if (data.status === "tired" || data.handAnalysis?.hand_fatigue_detected) {
      setCurrentSuggestion("😴 You seemed tired towards the end. Consider taking longer breaks next time.")
    } else {
      setCurrentSuggestion("📊 Session completed. Review your performance metrics for insights.")
    }
  }

  // AI Analysis Function
  const analyzeFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return

    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    
    if (!ctx) return

    // Set canvas size to match video
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    // Convert canvas to base64 image
    const imageDataURL = canvas.toDataURL('image/jpeg', 0.8)

    try {
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageDataURL
        })
      })

      if (response.ok) {
        const data = await response.json()
        
        // Debug: Log the original gaze data
        console.log("AI Analysis - Original data:", {
          gazeLeft: data.gazeLeft,
          gazeRight: data.gazeRight,
          swapped: `${data.gazeRight}/${data.gazeLeft}`
        })
        
        // Update AI analysis state
        setFocusScore(data.lernfaehigkeitsScore || 0)
        setAttentionStatus(data.attention || "No Data")
        setFatigueStatus(data.status || "awake")
        // Fix gaze direction for mirrored camera (swap left and right)
        setGazeDirection(`${data.gazeRight || "unknown"}/${data.gazeLeft || "unknown"}`)
        setHeadPose(data.headPose || { pitch: 0, yaw: 0, roll: 0 })
        setHandAnalysis(data.handAnalysis || {
          hand_fatigue_detected: false,
          hand_at_head: false,
          playing_with_hair: false,
          hand_movement: 'normal'
        })
        setEarValue(data.avgEAR || 0)

        // Generate AI suggestions based on analysis
        generateAISuggestion(data)
      }
    } catch (error) {
      console.error('AI Analysis Error:', error)
    }
  }, [])

  // Generate suggestions based on AI analysis
  const generateAISuggestion = (data: {
    status?: string
    attention?: string
    lernfaehigkeitsScore?: number
    handAnalysis?: {
      hand_fatigue_detected?: boolean
      playing_with_hair?: boolean
    }
  }) => {
    if (data.status === "tired" || data.handAnalysis?.hand_fatigue_detected) {
      setCurrentSuggestion("You seem tired. Consider taking a break.")
    } else if (data.attention === "abgelenkt") {
      setCurrentSuggestion("Your attention seems to be wandering. Try to refocus.")
    } else if (data.handAnalysis?.playing_with_hair) {
      setCurrentSuggestion("Detected fidgeting. Take a moment to center yourself.")
    } else if ((data.lernfaehigkeitsScore || 0) > 80) {
      setCurrentSuggestion("Great focus! You're in an optimal learning state.")
    } else if ((data.lernfaehigkeitsScore || 0) < 60) {
      setCurrentSuggestion("Try adjusting your posture and focus on the material.")
    } else {
      setCurrentSuggestion("Maintaining good focus. Keep up the good work!")
    }
  }

  // 🧪 TOKEN-SPARSAMER Gemini Test
  const testGemini = useCallback(async () => {
    setIsGeminiTesting(true)
    setGeminiTestResult(null)
    
    try {
      const response = await fetch('http://localhost:5000/api/test-gemini', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          focus_score: focusScore || 75,
          attention: attentionStatus || 'focused'
        })
      })

      if (response.ok) {
        const data = await response.json()
        if (data.gemini_test.success) {
          setGeminiTestResult(data.gemini_test.test_recommendation)
          setCurrentSuggestion(`🤖 Gemini: ${data.gemini_test.test_recommendation}`)
        } else {
          setGeminiTestResult(`Fehler: ${data.gemini_test.error}`)
        }
      } else {
        setGeminiTestResult("Gemini API nicht erreichbar")
      }
    } catch (error) {
      console.error('Gemini Test Error:', error)
      setGeminiTestResult("Netzwerk-Fehler beim Gemini Test")
    } finally {
      setIsGeminiTesting(false)
    }
  }, [focusScore, attentionStatus])

  useEffect(() => {
    if (isRecording && !isPaused) {
      timerRef.current = setInterval(() => {
        setStudyTime((prev) => prev + 1)
        setSessionTime((prev) => prev + 1) // Increment session time
        setSessionProgress((prev) => Math.min(prev + 0.5, 100))
      }, 1000)

      // Start AI analysis every 10 seconds (reduced frequency)
      analysisIntervalRef.current = setInterval(() => {
        analyzeFrame()
      }, 10000)
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
      if (analysisIntervalRef.current) {
        clearInterval(analysisIntervalRef.current)
      }
    }

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
      if (analysisIntervalRef.current) {
        clearInterval(analysisIntervalRef.current)
      }
    }
  }, [isRecording, isPaused, analyzeFrame])

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: !isMuted,
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
      }

      // Start session with Flask backend
      const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      
      try {
        const response = await fetch('http://localhost:5000/api/start-session', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            sessionId: sessionId
          })
        })

        if (response.ok) {
          const data = await response.json()
          console.log('✅ Session started:', data)
          setCurrentSuggestion(`📊 Session started: ${sessionId.substring(0, 12)}... (DB: ${data.databaseConnected ? '✓' : '✗'})`)
        } else {
          console.error('Failed to start session with backend')
          setCurrentSuggestion("Session started locally (backend offline)")
        }
      } catch (error) {
        console.error('Error starting session:', error)
        setCurrentSuggestion("Session started locally (backend offline)")
      }

      setIsRecording(true)
    } catch (error) {
      console.error("Error accessing camera:", error)
    }
  }

  const pauseRecording = () => {
    setIsPaused(!isPaused)
    // Reset session time when pausing
    if (!isPaused) {
      setSessionTime(0)
    }
  }

  const stopRecording = async () => {
    // Stop all intervals first
    if (timerRef.current) {
      clearInterval(timerRef.current)
    }
    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current)
    }

    // Perform final analysis on the frozen frame
    await performFinalAnalysis()

    // End session with Flask backend
    try {
      const response = await fetch('http://localhost:5000/api/end-session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          totalDuration: studyTime
        })
      })

      if (response.ok) {
        const data = await response.json()
        console.log('✅ Session ended:', data)
        setCurrentSuggestion(`✅ Session completed! Duration: ${formatTime(studyTime)} (${data.databaseConnected ? 'Saved to DB' : 'Local only'})`)
      } else {
        console.error('Failed to end session with backend')
        setCurrentSuggestion(`Session completed! Duration: ${formatTime(studyTime)}`)
      }
    } catch (error) {
      console.error('Error ending session:', error)
      setCurrentSuggestion(`Session completed! Duration: ${formatTime(studyTime)}`)
    }

    // Stop the video stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
    }

    setIsRecording(false)
    setIsPaused(false)
  }

  const resetSession = () => {
    // Stop all intervals
    if (timerRef.current) {
      clearInterval(timerRef.current)
    }
    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current)
    }

    // Stop video stream if running
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
    }

    // Reset all state
    setIsRecording(false)
    setIsPaused(false)
    setStudyTime(0)
    setSessionTime(0) // Reset session time
    setSessionProgress(0)
    setCurrentSuggestion("")
    setFocusScore(0)
    setAttentionStatus("No Data")
    setFatigueStatus("awake")
    setGazeDirection("unknown")
    setHeadPose({ pitch: 0, yaw: 0, roll: 0 })
    setHandAnalysis({
      hand_fatigue_detected: false,
      hand_at_head: false,
      playing_with_hair: false,
      hand_movement: 'normal'
    })
    setEarValue(0)
    setIsAnalyzing(false)
    setFinalAnalysis(null)
  }

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60

    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`
    }
    return `${minutes}:${secs.toString().padStart(2, "0")}`
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-orange-50/30 dark:from-[#0a0a0a] dark:via-[#121212] dark:to-[#1a1a1a] p-4 md:p-6 relative overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-20 w-72 h-72 bg-gradient-to-r from-[#f4895c]/20 to-[#f4a072]/20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-gradient-to-r from-[#f4a072]/15 to-[#f4c572]/15 rounded-full blur-3xl animate-pulse delay-1000" />
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-gradient-to-r from-[#f4895c]/5 to-[#f4a072]/5 rounded-full blur-3xl animate-spin" style={{ animationDuration: '60s' }} />
      </div>

      <div className="mx-auto max-w-7xl space-y-8 relative z-10">
        {/* Enhanced Header with Stats */}
        <div className="relative">
          <div className="absolute inset-0 bg-gradient-to-r from-[#f4895c]/10 to-[#f4a072]/10 rounded-3xl blur-3xl" />
          <div className="relative bg-white/80 dark:bg-[#1a1a1a]/80 backdrop-blur-xl border border-gray-200/50 dark:border-gray-800/50 rounded-3xl p-8 animate-in fade-in-0 slide-in-from-top-8 duration-1000 shadow-2xl">
            <div className="flex items-center justify-between mb-6">
              <div className="space-y-2">
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <div className="w-12 h-12 bg-gradient-to-br from-[#f4895c] to-[#f4a072] rounded-2xl flex items-center justify-center shadow-lg">
                      <Brain className="h-6 w-6 text-white" />
                    </div>
                  </div>
                  <div>
                    <h1 className="text-4xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 dark:from-white dark:to-gray-300 bg-clip-text text-transparent animate-in slide-in-from-left-6 duration-1000 delay-300">
                      EMOL
                    </h1>                  
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-3">  
                {/* Action Buttons */}
                <Button 
                  variant="ghost" 
                  size="icon" 
                  className="text-gray-600 dark:text-gray-400 hover:bg-[#f4895c]/20 hover:text-[#f4895c] dark:hover:bg-[#f4895c]/20 dark:hover:text-[#f4a072] transition-all duration-300 hover:scale-110 rounded-xl" 
                  onClick={() => router.push('/analytics')}
                >
                  <BarChart3 className="h-5 w-5" />
                </Button>
                <Button 
                  variant="ghost" 
                  size="icon" 
                  className="text-gray-600 dark:text-gray-400 hover:bg-[#f4895c]/20 hover:text-[#f4895c] dark:hover:bg-[#f4895c]/20 dark:hover:text-[#f4a072] transition-all duration-300 hover:scale-110 rounded-xl" 
                  onClick={() => router.push('/settings')}
                >
                  <SettingsIcon className="h-5 w-5" />
                </Button>
              </div>
            </div>
            
            {/* Progress Ring */}
            <div className="flex items-center justify-center mb-6">
              <div className="flex items-center gap-8">
                {/* Session Time Ring */}
                <div className="relative">
                  <svg className="w-28 h-28 transform -rotate-90" viewBox="0 0 120 120">
                    <circle cx="60" cy="60" r="54" fill="none" stroke="currentColor" strokeWidth="6" className="text-gray-200 dark:text-gray-700" />
                    <circle 
                      cx="60" 
                      cy="60" 
                      r="54" 
                      fill="none" 
                      stroke="url(#sessionGradient)" 
                      strokeWidth="6" 
                      strokeLinecap="round"
                      strokeDasharray={`${Math.min(sessionTime * 0.5, 100) * 3.39} 339`}
                      className="transition-all duration-1000 ease-out"
                    />
                    <defs>
                      <linearGradient id="sessionGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#3b82f6" />
                        <stop offset="100%" stopColor="#1d4ed8" />
                      </linearGradient>
                    </defs>
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className="text-xl font-bold text-white dark:text-white">{formatTime(sessionTime)}</div>
                      <div className="text-xs text-white dark:text-white">Session</div>
                    </div>
                  </div>
                </div>

                {/* Study Time Ring */}
                <div className="relative">
                  <svg className="w-32 h-32 transform -rotate-90" viewBox="0 0 120 120">
                    <circle cx="60" cy="60" r="54" fill="none" stroke="currentColor" strokeWidth="8" className="text-gray-200 dark:text-gray-700" />
                    <circle 
                      cx="60" 
                      cy="60" 
                      r="54" 
                      fill="none" 
                      stroke="url(#gradient)" 
                      strokeWidth="8" 
                      strokeLinecap="round"
                      strokeDasharray={`${sessionProgress * 3.39} 339`}
                      className="transition-all duration-1000 ease-out"
                    />
                    <defs>
                      <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#f4895c" />
                        <stop offset="100%" stopColor="#f4a072" />
                      </linearGradient>
                    </defs>
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-white dark:text-white">{formatTime(studyTime)}</div>
                      <div className="text-xs text-white dark:text-white">Total Time</div> 
                    </div>
                  </div>
                </div>
              </div>
            </div>

          </div>
        </div>

        {/* Gamification Bar */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 animate-in fade-in-0 slide-in-from-bottom-8 duration-1000 delay-300">
          <Card className="bg-gradient-to-r from-purple-500 to-purple-600 text-white border-0 shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 group cursor-pointer">
            <CardContent className="p-4 text-center">
              <div className="flex items-center justify-center mb-2">
                <Crown className="h-6 w-6 group-hover:rotate-12 transition-transform duration-300" />
              </div>
              <div className="text-xl font-bold">Level 12</div>
              <div className="text-purple-100 text-xs">Study Master</div>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-r from-orange-500 to-red-500 text-white border-0 shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 group cursor-pointer">
            <CardContent className="p-4 text-center">
              <div className="flex items-center justify-center mb-2">
                <Flame className="h-6 w-6 group-hover:animate-bounce transition-transform duration-300" />
              </div>
              <div className="text-xl font-bold">7 Days</div>
              <div className="text-orange-100 text-xs">Study Streak</div>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-r from-emerald-500 to-green-600 text-white border-0 shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 group cursor-pointer">
            <CardContent className="p-4 text-center">
              <div className="flex items-center justify-center mb-2">
                <Trophy className="h-6 w-6 group-hover:scale-110 transition-transform duration-300" />
              </div>
              <div className="text-xl font-bold">2,450</div>
              <div className="text-emerald-100 text-xs">XP Points</div>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white border-0 shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 group cursor-pointer">
            <CardContent className="p-4 text-center">
              <div className="flex items-center justify-center mb-2">
                <Users className="h-6 w-6 group-hover:scale-110 transition-transform duration-300" />
              </div>
              <div className="text-xl font-bold">#42</div>
              <div className="text-blue-100 text-xs">Global Rank</div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Grid */}
        <div className="grid gap-8 lg:grid-cols-3 animate-in fade-in-0 slide-in-from-bottom-8 duration-1000 delay-500">
          {/* Enhanced Video Feed */}
          <div className="lg:col-span-2">
            <Card className="bg-gradient-to-br from-white to-gray-50/50 dark:from-[#1a1a1a] dark:to-[#252525]/50 border-gray-200/50 dark:border-gray-800/50 shadow-2xl hover:shadow-3xl transition-all duration-500 hover:scale-[1.01] group relative overflow-hidden">
              {/* Animated border gradient */}
              <div className="absolute inset-0 bg-gradient-to-r from-[#f4895c]/20 via-[#f4a072]/20 to-[#f4c572]/20 rounded-xl blur-sm opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              
              <CardHeader className="pb-4 relative z-10">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-3">
                    <div className="relative">
                      <div className="p-3 rounded-xl bg-gradient-to-br from-[#f4895c] to-[#f4a072] text-white shadow-lg group-hover:shadow-xl transition-all duration-300 group-hover:scale-110 group-hover:rotate-3">
                        <Camera className="h-5 w-5" />
                      </div>
                      <div className="absolute -top-1 -right-1 w-4 h-4 bg-gradient-to-r from-green-400 to-emerald-500 rounded-full animate-pulse shadow-lg" />
                    </div>
                    <div>
                      <div>AI Study Session</div>
                      <div className="text-sm font-normal text-gray-600 dark:text-gray-400">Focus & Performance Tracking</div>
                    </div>
                  </CardTitle>
                  <div className="flex items-center gap-3">
                    {/* Live Metrics */}
                    {isRecording && (
                      <div className="flex items-center gap-3 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-3 border border-green-200 dark:border-green-800">
                        <div className="text-center">
                          <div className="text-lg font-bold text-green-600">{focusScore}%</div>
                          <div className="text-xs text-green-600/80">Focus</div>
                        </div>
                        <div className="w-px h-8 bg-green-300 dark:bg-green-600" />
                        <div className="text-center">
                          <div className="text-lg font-bold text-blue-600">{Math.floor(sessionTime / 60)}m</div>
                          <div className="text-xs text-blue-600/80">Session</div>
                        </div>
                        <div className="w-px h-8 bg-green-300 dark:bg-green-600" />
                        <div className="text-center">
                          <div className="text-lg font-bold text-purple-600">{Math.floor(studyTime / 60)}m</div>
                          <div className="text-xs text-purple-600/80">Total</div>
                        </div>
                      </div>
                    )}
                    
                    {/* Action Buttons */}
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm border border-white/20 dark:border-gray-700/20 text-gray-600 dark:text-gray-400 hover:bg-[#f4895c]/20 hover:text-[#f4895c] dark:hover:bg-[#f4895c]/20 dark:hover:text-[#f4a072] transition-all duration-300 hover:scale-105 active:scale-95 rounded-xl px-3 py-2 shadow-lg" 
                      onClick={() => router.push('/analytics')}
                    >
                      <BarChart3 className="h-4 w-4 mr-1" />
                      <span className="text-xs font-medium">Analytics</span>
                    </Button>
                    
                    {/* Developer Test Button */}
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      className="bg-purple-500/20 dark:bg-purple-600/20 backdrop-blur-sm border border-purple-300/20 dark:border-purple-700/20 text-purple-600 dark:text-purple-400 hover:bg-purple-500/30 hover:text-purple-700 dark:hover:bg-purple-600/30 dark:hover:text-purple-300 transition-all duration-300 hover:scale-105 active:scale-95 rounded-xl px-3 py-2 shadow-lg" 
                      onClick={analyzeFrame}
                      disabled={!isRecording}
                    >
                      <Brain className="h-4 w-4 mr-1" />
                      <span className="text-xs font-medium">Test AI</span>
                    </Button>
                    
                    {isRecording && (
                      <div className="flex items-center gap-2 bg-gradient-to-r from-red-500 to-pink-500 text-white rounded-xl px-4 py-2 shadow-lg animate-pulse">
                        <div className="w-2 h-2 bg-white rounded-full animate-ping" />
                        <span className="text-xs font-medium">LIVE</span>
                      </div>
                    )}
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="space-y-6 relative z-10">
                <div className="relative aspect-video overflow-hidden rounded-2xl bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-800 dark:to-gray-900 shadow-2xl border border-gray-200/50 dark:border-gray-700/50">
                  <video 
                    ref={videoRef} 
                    autoPlay 
                    muted 
                    className="h-full w-full object-cover transition-all duration-500" 
                    style={{ 
                      transform: 'scaleX(-1)',
                      WebkitTransform: 'scaleX(-1)',
                      MozTransform: 'scaleX(-1)',
                      msTransform: 'scaleX(-1)'
                    }}
                  />
                  
                  {/* Overlay Elements */}
                  {isRecording && (
                    <div className="absolute top-4 left-4 flex items-center gap-3">
                      <div className="bg-black/50 backdrop-blur-md rounded-lg px-3 py-2 text-white text-sm font-medium">
                        🎯 {attentionStatus}
                      </div>
                      <div className="bg-black/50 backdrop-blur-md rounded-lg px-3 py-2 text-white text-sm font-medium">
                        ⚡ {earValue.toFixed(1)}% EAR
                      </div>
                    </div>
                  )}
                  
                  {!isRecording && (
                    <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-black/10 to-black/20 backdrop-blur-sm">
                      <div className="text-center text-gray-600 dark:text-gray-300 animate-in fade-in-0 zoom-in-50 duration-1000">
                        <div className="relative mb-6">
                          <div className="relative">
                            <Rocket className="mx-auto h-16 w-16 text-[#f4895c] animate-bounce" />
                            <div className="absolute inset-0 mx-auto h-16 w-16 rounded-full bg-[#f4895c]/20 animate-ping" />
                            <Sparkles className="absolute -top-2 -right-2 h-6 w-6 text-yellow-400 animate-pulse" />
                          </div>
                        </div>
                        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">Ready to Level Up? 🚀</h3>
                        <p className="text-gray-600 dark:text-gray-400 mb-4">Start your AI-powered study session and boost your focus!</p>
                        <div className="flex items-center justify-center gap-4 text-sm">
                          <div className="flex items-center gap-1">
                            <Brain className="h-4 w-4 text-purple-500" />
                            <span>AI Analysis</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <Target className="h-4 w-4 text-green-500" />
                            <span>Focus Tracking</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <Trophy className="h-4 w-4 text-yellow-500" />
                            <span>XP Rewards</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Controls */}
                <div className="flex items-center justify-center gap-3">
                  {!isRecording ? (
                    <>
                      <Button 
                        onClick={startRecording} 
                        className="group relative bg-gradient-to-r from-[#f4895c] via-[#f4a072] to-[#f4b462] hover:from-[#f4a072] hover:via-[#f4b462] hover:to-[#f4c572] text-white shadow-2xl hover:shadow-3xl transition-all duration-500 ease-out flex items-center justify-center px-8 py-4 rounded-2xl transform hover:scale-110 active:scale-95 font-bold text-lg border-2 border-white/20"
                      >
                        <div className="absolute inset-0 bg-gradient-to-r from-white/20 to-transparent rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                        <div className="relative flex items-center gap-3">
                          <div className="relative">
                            <Play className="h-6 w-6 transition-all duration-500 ease-out group-hover:scale-125" />
                            <div className="absolute inset-0 h-6 w-6 bg-white/30 rounded-full animate-ping group-hover:animate-pulse" />
                          </div>
                          <span className="font-bold tracking-wide">START MISSION</span>
                          <Rocket className="h-5 w-5 transition-all duration-500 group-hover:translate-x-1 group-hover:scale-110" />
                          <Sparkles className="h-4 w-4 text-yellow-300 animate-pulse" />
                        </div>
                      </Button>
                      
                      {(studyTime > 0 || finalAnalysis) && (
                        <Button 
                          variant="outline" 
                          onClick={resetSession}
                          className="border-gray-300 dark:border-gray-600 bg-white dark:bg-[#1a1a1a] text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-all duration-300 hover:scale-105 active:scale-95 hover:rotate-12 rounded-full p-3"
                        >
                          <RotateCcw className="h-4 w-4 transition-transform duration-300" />
                        </Button>
                      )}
                    </>
                  ) : (
                    <>
                      <Button 
                        variant="outline" 
                        onClick={pauseRecording} 
                        className="border-gray-300 dark:border-gray-600 bg-white dark:bg-[#1a1a1a] text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-all duration-300 hover:scale-105 active:scale-95 rounded-full p-3"
                      >
                        <div className="relative">
                          {isPaused ? <Play className="h-4 w-4 animate-pulse" /> : <Pause className="h-4 w-4" />}
                        </div>
                      </Button>
                      <Button
                        variant="outline"
                        onClick={stopRecording}
                        disabled={isAnalyzing}
                        className="border-red-300 dark:border-red-600 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 bg-white dark:bg-[#1a1a1a] transition-all duration-300 hover:scale-105 active:scale-95 disabled:hover:scale-100 rounded-full p-3"
                      >
                        {isAnalyzing ? (
                          <div className="flex items-center">
                            <div className="animate-spin h-4 w-4 border-2 border-red-500 border-t-transparent rounded-full" />
                          </div>
                        ) : (
                          <Square className="h-4 w-4" />
                        )}
                      </Button>
                    </>
                  )}

                  <Button 
                    variant="outline" 
                    onClick={() => setIsMuted(!isMuted)} 
                    className={`transition-all duration-300 hover:scale-105 active:scale-95 rounded-full p-3 ${
                      isMuted 
                        ? 'border-red-300 dark:border-red-600 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 bg-white dark:bg-[#1a1a1a] animate-pulse' 
                        : 'border-gray-300 dark:border-gray-600 bg-white dark:bg-[#1a1a1a] text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                    }`}
                  >
                    <div className="relative">
                      {isMuted ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                      {!isMuted && (
                        <div className="absolute -top-1 -right-1 h-2 w-2 bg-green-500 rounded-full animate-pulse" />
                      )}
                    </div>
                  </Button>
                </div>

                {/* Live Analytics Panel - Shows during recording */}
                {isRecording && (
                  <div className="mt-4 p-4 rounded-xl bg-gradient-to-r from-[#f4895c]/10 to-[#f4a072]/10 border border-[#f4895c]/20 animate-in fade-in-0 slide-in-from-bottom-4 duration-500">
                    <div className="flex items-center gap-2 mb-3">
                      <div className="p-1.5 rounded-lg bg-gradient-to-br from-[#f4895c] to-[#f4a072] text-white shadow-sm">
                        <BarChart3 className="h-3 w-3" />
                      </div>
                      <h4 className="text-sm font-semibold text-gray-900 dark:text-white">Live Analytics</h4>
                      <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse ml-auto" />
                    </div>
                    <div className="grid grid-cols-3 gap-4 text-center">
                      <div className="p-3 rounded-lg bg-white/50 dark:bg-[#1a1a1a]/50 backdrop-blur-sm border border-white/20 dark:border-gray-800/20">
                        <div className="text-lg font-bold text-gray-900 dark:text-white">{Math.floor(sessionTime / 60)}:{(sessionTime % 60).toString().padStart(2, '0')}</div>
                        <div className="text-xs text-gray-600 dark:text-gray-400">Session Time</div>
                      </div>
                      <div className="p-3 rounded-lg bg-white/50 dark:bg-[#1a1a1a]/50 backdrop-blur-sm border border-white/20 dark:border-gray-800/20">
                        <div className="text-lg font-bold text-gray-900 dark:text-white">{Math.floor(studyTime / 60)}:{(studyTime % 60).toString().padStart(2, '0')}</div>
                        <div className="text-xs text-gray-600 dark:text-gray-400">Total Time</div>
                      </div>
                      <div className="p-3 rounded-lg bg-white/50 dark:bg-[#1a1a1a]/50 backdrop-blur-sm border border-white/20 dark:border-gray-800/20">
                        <div className="text-lg font-bold text-gray-900 dark:text-white">{focusScore}%</div>
                        <div className="text-xs text-gray-600 dark:text-gray-400">Current Focus</div>
                      </div>
                    </div>
                    <div className="mt-3 flex items-center justify-between">
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        className="text-xs text-gray-600 dark:text-gray-400 hover:text-[#f4895c] dark:hover:text-[#f4a072] transition-colors duration-300" 
                        onClick={() => router.push('/analytics')}
                      >
                        View Full Analytics →
                      </Button>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        Updates every second
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Enhanced Sidebar */}
          <div className="space-y-6 animate-in fade-in-0 slide-in-from-right-8 duration-1000 delay-1000">

            {/* Focus Chart Component */}
            <FocusChart 
              currentFocusScore={focusScore} 
              isRecording={isRecording}
              className="hover:shadow-lg transition-all duration-500 hover:scale-[1.02]"
            />

            {/* AI Analysis Details */}
            <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm hover:shadow-lg transition-all duration-500 hover:scale-[1.02] group relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-[#f4895c]/5 to-[#f4a072]/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              <CardHeader className="pb-3 relative">
                <CardTitle className="text-base text-gray-900 dark:text-white flex items-center gap-2">
                  <div className="p-1.5 rounded-lg bg-gradient-to-br from-[#f4895c] to-[#f4a072] text-white shadow-sm group-hover:shadow-md transition-all duration-300 group-hover:scale-110 group-hover:rotate-12">
                    <Brain className="h-3.5 w-3.5" />
                  </div>
                  AI Analysis 
                  {finalAnalysis && !isRecording && (
                    <span className="text-xs bg-gradient-to-r from-[#f4c572] to-[#f4bf62] text-white px-3 py-1 rounded-full ml-2 shadow-sm animate-bounce">
                      Final Results
                    </span>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 relative">
                {isAnalyzing && (
                  <div className="flex items-center justify-center p-4 animate-in fade-in-0 zoom-in-50 duration-500">
                    <div className="relative">
                      <div className="animate-spin h-8 w-8 border-3 border-[#f4895c] border-t-transparent rounded-full" />
                      <div className="absolute inset-0 animate-ping h-8 w-8 border-2 border-[#f4a072] border-t-transparent rounded-full opacity-30" />
                    </div>
                    <span className="text-sm text-gray-600 dark:text-gray-400 ml-3 animate-pulse">Analyzing final frame...</span>
                  </div>
                )}
                
                {/* Enhanced AI Results Display */}
                <div className="bg-gradient-to-br from-gray-50 to-white dark:from-gray-900 dark:to-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700 shadow-inner">
                  {/* Focus Score - Main Metric */}
                  <div className="text-center mb-4 p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-600 shadow-sm">
                    <div className="text-3xl font-bold text-[#f4895c] mb-1">{focusScore}%</div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 uppercase tracking-wide">Focus Score</div>
                    <div className={`inline-block px-3 py-1 rounded-full text-xs font-medium mt-2 ${
                      focusScore >= 80 ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400' :
                      focusScore >= 60 ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400' :
                      'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
                    }`}>
                      {focusScore >= 80 ? '🎯 Excellent' : focusScore >= 60 ? '⚡ Good' : '📈 Needs Focus'}
                    </div>
                  </div>
                  
                  {/* Detailed Metrics Grid */}
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="p-3 rounded-lg bg-white dark:bg-[#0f0f0f] border border-gray-200 dark:border-gray-800 transition-all duration-300 hover:scale-105">
                      <div className="flex items-center gap-2 mb-1">
                        <div className={`w-2 h-2 rounded-full ${fatigueStatus === 'tired' ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`} />
                        <span className="text-gray-600 dark:text-gray-400 text-[10px] uppercase tracking-wide">Energy Level:</span>
                      </div>
                      <div className={`font-bold text-sm ${fatigueStatus === 'tired' ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'} capitalize`}>
                        {fatigueStatus === 'tired' ? '😴 Tired' : '⚡ Awake'}
                      </div>
                    </div>
                    
                    <div className="p-3 rounded-lg bg-white dark:bg-[#0f0f0f] border border-gray-200 dark:border-gray-800 transition-all duration-300 hover:scale-105">
                      <div className="flex items-center gap-2 mb-1">
                        <div className="w-2 h-2 rounded-full bg-blue-500" />
                        <span className="text-gray-600 dark:text-gray-400 text-[10px] uppercase tracking-wide">Attention:</span>
                      </div>
                      <div className={`font-bold text-sm capitalize ${
                        attentionStatus === 'focused' ? 'text-green-600 dark:text-green-400' :
                        attentionStatus === 'abgelenkt' ? 'text-red-600 dark:text-red-400' :
                        'text-gray-600 dark:text-gray-400'
                      }`}>
                        {attentionStatus === 'focused' ? '🎯 Focused' : 
                         attentionStatus === 'abgelenkt' ? '👀 Distracted' : 
                         '• ' + attentionStatus}
                      </div>
                    </div>
                    
                    <div className="p-3 rounded-lg bg-white dark:bg-[#0f0f0f] border border-gray-200 dark:border-gray-800 transition-all duration-300 hover:scale-105">
                      <div className="flex items-center gap-2 mb-1">
                        <div className="w-2 h-2 rounded-full bg-purple-500" />
                        <span className="text-gray-600 dark:text-gray-400 text-[10px] uppercase tracking-wide">Gaze:</span>
                      </div>
                      <div className="font-bold text-sm text-purple-600 dark:text-purple-400">
                        {gazeDirection.replace('left', '← Left').replace('right', 'Right →').replace('center', '• Center').replace('unknown', '• Unknown')}
                      </div>
                    </div>
                    
                    <div className="p-3 rounded-lg bg-white dark:bg-[#0f0f0f] border border-gray-200 dark:border-gray-800 transition-all duration-300 hover:scale-105">
                      <div className="flex items-center gap-2 mb-1">
                        <div className="w-2 h-2 rounded-full bg-orange-500" />
                        <span className="text-gray-600 dark:text-gray-400 text-[10px] uppercase tracking-wide">Eye Activity:</span>
                      </div>
                      <div className="font-bold text-sm text-orange-600 dark:text-orange-400">{earValue.toFixed(1)}% EAR</div>
                    </div>
                  </div>
                  
                  {/* Analysis Frequency Info */}
                  <div className="mt-3 text-center">
                    <div className="text-xs text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-800 rounded-lg px-3 py-2 inline-block">
                      📊 AI Analysis every 10 seconds • Last update: {new Date().toLocaleTimeString()}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3 text-xs" style={{ display: 'none' }}>
                  <div className="p-2 rounded-lg bg-white dark:bg-[#0f0f0f] border border-gray-200 dark:border-gray-800 transition-all duration-300 hover:scale-105 hover:bg-gray-50 dark:hover:bg-gray-700">
                    <span className="text-gray-600 dark:text-gray-400 text-[10px] uppercase tracking-wide">Fatigue:</span>
                    <div className={`font-bold text-sm ${fatigueStatus === 'tired' ? 'text-red-600 dark:text-red-400 animate-pulse' : 'text-green-400 dark:text-green-600'} capitalize`}>
                      {fatigueStatus}
                    </div>
                  </div>
                  <div className="p-2 rounded-lg bg-white dark:bg-[#0f0f0f] border border-gray-200 dark:border-gray-800 transition-all duration-300 hover:scale-105 hover:bg-gray-50 dark:hover:bg-gray-700">
                    <span className="text-gray-600 dark:text-gray-400 text-[10px] uppercase tracking-wide">Gaze Direction:</span>
                    <div className="font-bold text-sm text-gray-900 dark:text-white">
                      {gazeDirection.replace('left', '←').replace('right', '→').replace('unknown', '•')}
                    </div>
                  </div>
                  <div className="p-2 rounded-lg bg-white dark:bg-[#0f0f0f] border border-gray-200 dark:border-gray-800 transition-all duration-300 hover:scale-105 hover:bg-gray-50 dark:hover:bg-gray-700">
                    <span className="text-gray-600 dark:text-gray-400 text-[10px] uppercase tracking-wide">EAR:</span>
                    <div className="font-bold text-sm text-[#f4895c]">{earValue.toFixed(1)}%</div>
                  </div>
                  <div className="p-2 rounded-lg bg-white dark:bg-[#0f0f0f] border border-gray-200 dark:border-gray-800 transition-all duration-300 hover:scale-105 hover:bg-gray-50 dark:hover:bg-gray-700">
                    <span className="text-gray-600 dark:text-gray-400 text-[10px] uppercase tracking-wide">Head Pose:</span>
                    <div className="font-bold text-sm text-gray-900 dark:text-white">
                      Y:{headPose.yaw.toFixed(0)}° P:{headPose.pitch.toFixed(0)}°
                    </div>
                  </div>
                </div>
                
                {handAnalysis.hand_fatigue_detected && (
                  <div className="rounded-xl bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 p-3 text-xs text-yellow-800 dark:text-yellow-300 border border-yellow-200 dark:border-yellow-800 shadow-sm animate-in slide-in-from-left-4 duration-500">
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-2 bg-yellow-500 rounded-full animate-pulse" />
                      <span className="font-medium">⚠️ Hand fatigue detected</span>
                    </div>
                  </div>
                )}
                
                {handAnalysis.playing_with_hair && (
                  <div className="rounded-xl bg-gradient-to-r from-[#f4c572]/20 to-[#f4bf62]/20 dark:from-[#f4bf62]/10 dark:to-[#f4c572]/10 p-3 text-xs text-[#7A4A2A] dark:text-[#f4bf62] border border-[#f4bf62]/40 dark:border-[#f4bf62]/25 shadow-sm animate-in slide-in-from-right-4 duration-500">
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-2 bg-[#f4bf62] rounded-full animate-bounce" />
                      <span className="font-medium">💫 Fidgeting detected</span>
                    </div>
                  </div>
                )}
                
                {finalAnalysis && !isRecording && (
                  <div className="border-t border-gray-200 dark:border-gray-700 pt-3 mt-3 animate-in fade-in-0 slide-in-from-bottom-4 duration-1000">
                    <div className="text-xs text-gray-600 dark:text-gray-400 mb-2 font-semibold uppercase tracking-wide">Session Summary:</div>
                    <div className="bg-white dark:bg-[#0f0f0f] rounded-xl p-4 text-xs border border-gray-200 dark:border-gray-800 shadow-inner">
                      <div className="grid grid-cols-2 gap-3">
                        <div className="flex flex-col space-y-1">
                          <span className="text-gray-500 dark:text-gray-400 text-[10px] uppercase tracking-wide">Final Focus:</span>
                          <span className="font-bold text-lg text-[#f4895c]">{finalAnalysis.lernfaehigkeitsScore}</span>
                        </div>
                        <div className="flex flex-col space-y-1">
                          <span className="text-gray-500 dark:text-gray-400 text-[10px] uppercase tracking-wide">Final Attention:</span>
                          <span className="font-semibold text-gray-900 dark:text-white capitalize">{finalAnalysis.attention}</span>
                        </div>
                        <div className="flex flex-col space-y-1">
                          <span className="text-gray-500 dark:text-gray-400 text-[10px] uppercase tracking-wide">Final Status:</span>
                          <span className="font-semibold text-gray-900 dark:text-white capitalize">{finalAnalysis.status}</span>
                        </div>
                        <div className="flex flex-col space-y-1">
                          <span className="text-gray-500 dark:text-gray-400 text-[10px] uppercase tracking-wide">Final EAR:</span>
                          <span className="font-semibold text-gray-900 dark:text-white">{finalAnalysis.avgEAR?.toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Hidden canvas for AI analysis */}
            <canvas ref={canvasRef} style={{ display: 'none' }} />

            {/* AI Insights */}
            <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm hover:shadow-lg transition-all duration-500 hover:scale-[1.02] group relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-[#f4895c]/5 to-[#f4a072]/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              <CardHeader className="pb-3 relative">
                <CardTitle className="text-base text-gray-900 dark:text-white flex items-center gap-2">
                  <div className="p-1.5 rounded-lg bg-gradient-to-br from-[#f4895c] to-[#f4a072] text-white shadow-sm group-hover:shadow-md transition-all duration-300 group-hover:scale-110 group-hover:rotate-12">
                    <Brain className="h-3.5 w-3.5" />
                  </div>
                  AI Insights
                </CardTitle>
              </CardHeader>
              <CardContent className="relative">
                {currentSuggestion ? (
                  <div className="space-y-3 animate-in fade-in-0 slide-in-from-bottom-4 duration-500">
                    <div className="rounded-xl bg-[#f4c572]/30 dark:bg-[#f4bf62]/15 p-4 text-sm text-[#7A4A2A] dark:text-[#f4bf62] border border-[#f4bf62]/40 dark:border-[#f4bf62]/25 shadow-sm relative overflow-hidden">
                      <div className="relative">{currentSuggestion}</div>
                    </div>

                    <div className="flex gap-2">
                      <Button size="sm" variant="ghost" className="h-8 text-xs hover:bg-gray-100 dark:hover:bg-gray-700 transition-all duration-300 hover:scale-105 group/btn">
                        <Coffee className="mr-1 h-3 w-3 transition-transform duration-300 group-hover/btn:rotate-12" />
                        Break
                      </Button>
                      <Button size="sm" variant="ghost" className="h-8 text-xs hover:bg-gray-100 dark:hover:bg-gray-700 transition-all duration-300 hover:scale-105 group/btn">
                        <BookOpen className="mr-1 h-3 w-3 transition-transform duration-300 group-hover/btn:scale-110" />
                        Switch Topic
                      </Button>
                      <Button size="sm" variant="ghost" className="h-8 text-xs hover:bg-gray-100 dark:hover:bg-gray-700 transition-all duration-300 hover:scale-105 group/btn">
                        <Zap className="mr-1 h-3 w-3 transition-transform duration-300 group-hover/btn:rotate-12" />
                        Deep Dive
                      </Button>
                    </div>
                    
                    {/* Gemini Test Section */}
                    <div className="mt-4 pt-3 border-t border-gray-200 dark:border-gray-700">
                      <Button 
                        size="sm" 
                        variant="outline" 
                        onClick={testGemini}
                        disabled={isGeminiTesting}
                        className="w-full h-9 text-xs bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 border-purple-200 dark:border-purple-800 text-purple-600 dark:text-purple-400 hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-all duration-300 hover:scale-105 disabled:hover:scale-100"
                      >
                        {isGeminiTesting ? (
                          <div className="flex items-center gap-2">
                            <div className="animate-spin h-3 w-3 border-2 border-purple-500 border-t-transparent rounded-full" />
                            Testing Gemini...
                          </div>
                        ) : (
                          <div className="flex items-center gap-2">
                            <Sparkles className="h-3 w-3" />
                            🧪 Test Gemini AI (~0.001€)
                          </div>
                        )}
                      </Button>
                      
                      {geminiTestResult && (
                        <div className="mt-2 p-2 rounded-lg bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 text-xs text-purple-700 dark:text-purple-300">
                          <div className="font-medium mb-1">🤖 Gemini Response:</div>
                          <div className="italic">{geminiTestResult}</div>
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-6 animate-in fade-in-0 duration-500">
                    <div className="relative mb-3">
                      <Brain className="mx-auto h-8 w-8 text-gray-400 dark:text-gray-600 animate-pulse" />
                      <div className="absolute inset-0 mx-auto h-8 w-8 rounded-full bg-[#f4895c]/20 animate-ping" />
                    </div>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Start a session to receive AI-powered insights</p>
                  </div>
                )}
              </CardContent>
            </Card> 
          </div>
        </div>

        {/* Bottom Achievement Section */}
        <div className="mt-12 animate-in fade-in-0 slide-in-from-bottom-8 duration-1000 delay-1200">
          <Card className="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 text-white border-0 shadow-2xl hover:shadow-3xl transition-all duration-500 hover:scale-[1.01] group relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-r from-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <CardContent className="p-8 relative z-10">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6 items-center">
                {/* Achievement Text */}
                <div className="md:col-span-2 space-y-4">
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center">
                      <Trophy className="h-6 w-6 text-yellow-300" />
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold">Achievement Unlocked!</h3>
                      <p className="text-indigo-100">You&apos;re building incredible study habits!</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      <Flame className="h-5 w-5 text-orange-300" />
                      <span className="font-medium">7-day streak</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Star className="h-5 w-5 text-yellow-300" />
                      <span className="font-medium">2,450 XP earned</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Crown className="h-5 w-5 text-purple-300" />
                      <span className="font-medium">Level 12 reached</span>
                    </div>
                  </div>
                </div>
                
                {/* Quick Stats */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-4 bg-white/10 rounded-xl backdrop-blur-sm">
                    <div className="text-2xl font-bold">47</div>
                    <div className="text-indigo-100 text-sm">Sessions</div>
                  </div>
                  <div className="text-center p-4 bg-white/10 rounded-xl backdrop-blur-sm">
                    <div className="text-2xl font-bold">39h</div>
                    <div className="text-indigo-100 text-sm">Study Time</div>
                  </div>
                </div>
                
                {/* CTA Button */}
                <div className="text-center">
                  <Button 
                    onClick={() => router.push('/analytics')}
                    className="bg-white text-purple-600 hover:bg-gray-50 font-bold px-6 py-3 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-110"
                  >
                    <div className="flex items-center gap-2">
                      <Gift className="h-5 w-5" />
                      View Rewards
                    </div>
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
