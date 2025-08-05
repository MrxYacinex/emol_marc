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
} from "lucide-react"

export default function StudyCompanion() {
  const router = useRouter()
  const [isRecording, setIsRecording] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [isMuted, setIsMuted] = useState(false)
  const [studyTime, setStudyTime] = useState(0)
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
      const response = await fetch('http://localhost:5000/api/analyze_frame', {
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
        setGazeDirection(`${data.gazeLeft || "unknown"}/${data.gazeRight || "unknown"}`)
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
      const response = await fetch('http://localhost:5000/api/analyze_frame', {
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
        
        // Update AI analysis state
        setFocusScore(data.lernfaehigkeitsScore || 0)
        setAttentionStatus(data.attention || "No Data")
        setFatigueStatus(data.status || "awake")
        setGazeDirection(`${data.gazeLeft || "unknown"}/${data.gazeRight || "unknown"}`)
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
    if (data.status === "tired") {
      setCurrentSuggestion("You seem tired. Consider taking a break chat.")
    } else if (data.handAnalysis?.hand_fatigue_detected){
      setCurrentSuggestion("Detected hand fatigue. Try to focus on the material or take a break.")
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

  useEffect(() => {
    if (isRecording && !isPaused) {
      timerRef.current = setInterval(() => {
        setStudyTime((prev) => prev + 1)
        setSessionProgress((prev) => Math.min(prev + 0.5, 100))
      }, 1000)

      // Start AI analysis every 3 seconds
      analysisIntervalRef.current = setInterval(() => {
        analyzeFrame()
      }, 3000)
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

      setIsRecording(true)
    } catch (error) {
      console.error("Error accessing camera:", error)
    }
  }

  const pauseRecording = () => {
    setIsPaused(!isPaused)
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
    <div className="min-h-screen bg-white dark:bg-[#121212] p-4 md:p-6">
      <div className="mx-auto max-w-6xl space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-gray-900 dark:text-white">Study Companion</h1>
            <p className="text-sm text-gray-600 dark:text-gray-400">AI-powered study session analysis</p>
          </div>
          <Button variant="ghost" size="icon" className="text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800" onClick={() => router.push('/settings')}>
            <SettingsIcon className="h-5 w-5" />
          </Button>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          {/* Video Feed */}
          <div className="lg:col-span-2">
            <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm">
              <CardHeader className="pb-4">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg text-gray-900 dark:text-white">Study Session</CardTitle>
                  <div className="flex items-center gap-2">
                    {isRecording && (
                      <Badge variant="secondary" className="bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-400 border-green-200 dark:border-green-800">
                        <div className="mr-1 h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                        Recording
                      </Badge>
                    )}
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="relative aspect-video overflow-hidden rounded-xl bg-gray-200 dark:bg-gray-800">
                  <video ref={videoRef} autoPlay muted className="h-full w-full object-cover" />
                  {!isRecording && (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center text-gray-600 dark:text-gray-300">
                        <Camera className="mx-auto h-12 w-12 mb-4 opacity-70" />
                        <p className="text-sm opacity-90">Click start to begin your study session</p>
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
                        className="group bg-green-600 hover:bg-green-700 text-white shadow-sm transition-all duration-[400ms] ease-out flex items-center justify-center px-3 py-2"
                      >
                        <Play className="h-4 w-4 transition-all duration-[400ms] ease-out group-hover:mr-2" />
                        <span className="max-w-0 overflow-hidden opacity-0 transition-all duration-[400ms] ease-out group-hover:max-w-[120px] group-hover:opacity-100 whitespace-nowrap">
                          Start Session
                        </span>
                      </Button>
                      
                      {(studyTime > 0 || finalAnalysis) && (
                        <Button 
                          variant="outline" 
                          onClick={resetSession}
                          className="border-gray-300 dark:border-gray-600 bg-white dark:bg-[#1a1a1a] text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
                        >
                          <RotateCcw className="h-4 w-4" />
                        </Button>
                      )}
                    </>
                  ) : (
                    <>
                      <Button variant="outline" onClick={pauseRecording} className="border-gray-300 dark:border-gray-600 bg-white dark:bg-[#1a1a1a] text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700">
                        {isPaused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
                      </Button>
                      <Button
                        variant="outline"
                        onClick={stopRecording}
                        disabled={isAnalyzing}
                        className="border-red-300 dark:border-red-600 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 bg-white dark:bg-[#1a1a1a]"
                      >
                        {isAnalyzing ? (
                          <>
                            <div className="animate-spin h-4 w-4 border-2 border-red-500 border-t-transparent rounded-full mr-2" />
                            Analyzing...
                          </>
                        ) : (
                          <Square className="h-4 w-4" />
                        )}
                      </Button>
                    </>
                  )}

                  <Button 
                    variant="outline" 
                    onClick={() => setIsMuted(!isMuted)} 
                    className={`border-gray-300 dark:border-gray-600 bg-white dark:bg-[#1a1a1a] hover:bg-gray-50 dark:hover:bg-gray-700 ${
                      isMuted 
                        ? 'text-red-600 dark:text-red-400 border-red-300 dark:border-red-600 hover:bg-red-50 dark:hover:bg-red-900/20' 
                        : 'text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    {isMuted ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Session Stats */}
            <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm">
              <CardHeader className="pb-3">
                <CardTitle className="text-base text-gray-900 dark:text-white flex items-center gap-2">
                  <Clock className="h-4 w-4" />
                  Session Stats
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-center">
                  <div className="text-2xl font-mono font-semibold text-gray-900 dark:text-white">{formatTime(studyTime)}</div>
                  <p className="text-xs text-gray-600 dark:text-gray-400">Study Time</p>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Progress</span>
                    <span className="text-gray-900 dark:text-white">{Math.round(sessionProgress)}%</span>
                  </div>
                  <Progress value={sessionProgress} className="h-2" />
                </div>

                <Separator className="dark:bg-gray-700" />

                <div className="grid grid-cols-2 gap-4 text-center">
                  <div>
                    <div className="text-lg font-semibold text-gray-900 dark:text-white">{focusScore}</div>
                    <p className="text-xs text-gray-600 dark:text-gray-400">Focus Score</p>
                  </div>
                  <div>
                    <div className="text-lg font-semibold text-gray-900 dark:text-white">{attentionStatus}</div>
                    <p className="text-xs text-gray-600 dark:text-gray-400">Attention</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* AI Analysis Details */}
            <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm">
              <CardHeader className="pb-3">
                <CardTitle className="text-base text-gray-900 dark:text-white flex items-center gap-2">
                  <Brain className="h-4 w-4" />
                  AI Analysis {finalAnalysis && !isRecording && (
                    <span className="text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 px-2 py-1 rounded-full ml-2 border border-blue-200 dark:border-blue-800">
                      Final Results
                    </span>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {isAnalyzing && (
                  <div className="flex items-center justify-center p-4">
                    <div className="animate-spin h-6 w-6 border-2 border-blue-500 border-t-transparent rounded-full mr-3" />
                    <span className="text-sm text-gray-600 dark:text-gray-400">Analyzing final frame...</span>
                  </div>
                )}
                
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">Fatigue:</span>
                    <div className={`font-semibold ${fatigueStatus === 'tired' ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'}`}>
                      {fatigueStatus}
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">Gaze:</span>
                    <div className="font-semibold text-gray-900 dark:text-white">{gazeDirection}</div>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">EAR:</span>
                    <div className="font-semibold text-gray-900 dark:text-white">{earValue.toFixed(1)}%</div>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">Head Pose: </span>
                    <div className="font-semibold text-gray-900 dark:text-white">
                      Y:{headPose.yaw.toFixed(0)}° P:{headPose.pitch.toFixed(0)}°
                    </div>
                  </div>
                </div>
                
                {handAnalysis.hand_fatigue_detected && (
                  <div className="rounded-lg bg-yellow-50 dark:bg-yellow-900/20 p-2 text-xs text-yellow-800 dark:text-yellow-300 border border-yellow-200 dark:border-yellow-800">
                    ⚠️ Hand fatigue detected
                  </div>
                )}
                
                {handAnalysis.playing_with_hair && (
                  <div className="rounded-lg bg-blue-50 dark:bg-blue-900/20 p-2 text-xs text-blue-800 dark:text-blue-300 border border-blue-200 dark:border-blue-800">
                    💫 Fidgeting detected
                  </div>
                )}
                
                {finalAnalysis && !isRecording && (
                  <div className="border-t border-gray-200 dark:border-gray-700 pt-3 mt-3">
                    <div className="text-xs text-gray-600 dark:text-gray-400 mb-2">Session Summary:</div>
                    <div className="bg-gray-50 dark:bg-[#0f0f0f] rounded-lg p-3 text-xs border border-gray-200 dark:border-gray-800">
                      <div className="grid grid-cols-2 gap-2">
                        <div>Final Focus Score: <span className="font-semibold text-gray-900 dark:text-white">{finalAnalysis.lernfaehigkeitsScore}</span></div>
                        <div>Final Attention: <span className="font-semibold text-gray-900 dark:text-white">{finalAnalysis.attention}</span></div>
                        <div>Final Status: <span className="font-semibold text-gray-900 dark:text-white">{finalAnalysis.status}</span></div>
                        <div>Final EAR: <span className="font-semibold text-gray-900 dark:text-white">{finalAnalysis.avgEAR?.toFixed(1)}%</span></div>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Hidden canvas for AI analysis */}
            <canvas ref={canvasRef} style={{ display: 'none' }} />

            {/* AI Insights */}
            <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm">
              <CardHeader className="pb-3">
                <CardTitle className="text-base text-gray-900 dark:text-white flex items-center gap-2">
                  <Brain className="h-4 w-4" />
                  AI Insights
                </CardTitle>
              </CardHeader>
              <CardContent>
                {currentSuggestion ? (
                  <div className="space-y-3">
                    <div className="rounded-lg bg-blue-50 dark:bg-blue-900/20 p-3 text-sm text-blue-900 dark:text-blue-300 border border-blue-200 dark:border-blue-800">{currentSuggestion}</div>

                    <div className="flex gap-2">
                      <Button size="sm" variant="ghost" className="h-8 text-xs hover:bg-gray-100 dark:hover:bg-gray-700">
                        <Coffee className="mr-1 h-3 w-3" />
                        Break
                      </Button>
                      <Button size="sm" variant="ghost" className="h-8 text-xs hover:bg-gray-100 dark:hover:bg-gray-700">
                        <BookOpen className="mr-1 h-3 w-3" />
                        Switch Topic
                      </Button>
                      <Button size="sm" variant="ghost" className="h-8 text-xs hover:bg-gray-100 dark:hover:bg-gray-700">
                        <Zap className="mr-1 h-3 w-3" />
                        Deep Dive
                      </Button>
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-gray-500 dark:text-gray-400">Start a session to receive AI-powered insights</p>
                )}
              </CardContent>
            </Card>

            {/* Quick Actions */}
            <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm">
              <CardHeader className="pb-3">
                <CardTitle className="text-base text-gray-900 dark:text-white flex items-center gap-2">
                  <Target className="h-4 w-4" />
                  Quick Actions
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button variant="ghost" className="w-full justify-start h-9 text-sm hover:bg-gray-100 dark:hover:bg-gray-700">
                  <TrendingUp className="mr-2 h-4 w-4" />
                  View Analytics
                </Button>
                <Button variant="ghost" className="w-full justify-start h-9 text-sm hover:bg-gray-100 dark:hover:bg-gray-700">
                  <BookOpen className="mr-2 h-4 w-4" />
                  Study Goals
                </Button>
                <Button variant="ghost" className="w-full justify-start h-9 text-sm hover:bg-gray-100 dark:hover:bg-gray-700">
                  <Coffee className="mr-2 h-4 w-4" />
                  Break Timer
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
