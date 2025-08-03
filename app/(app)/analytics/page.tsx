"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  ArrowLeft,
  TrendingUp,
  Brain,
  Clock,
  Target,
  Calendar,
  Activity,
  BarChart3,
  LineChart,
  AlertTriangle,
  CheckCircle,
  Flame
} from "lucide-react"

// Mock data for analytics - replace with real data from your backend
const mockData = {
  totalSessions: 47,
  totalStudyTime: 2340, // minutes
  averageSession: 49.8, // minutes
  averageFocus: 78.5,
  weeklyGoal: 20, // hours
  currentStreak: 7,
  bestStreak: 14,
  todayStats: {
    sessions: 3,
    studyTime: 165, // minutes
    focusScore: 82,
    avgFocusTime: 45 // minutes per session
  },
  focusDistribution: {
    excellent: 25,
    good: 45,
    fair: 20,
    poor: 10
  },
  weeklyData: [
    { day: 'Mon', sessions: 3, focusScore: 82, studyTime: 150, productivity: 88 },
    { day: 'Tue', sessions: 2, focusScore: 75, studyTime: 120, productivity: 72 },
    { day: 'Wed', sessions: 4, focusScore: 88, studyTime: 200, productivity: 91 },
    { day: 'Thu', sessions: 1, focusScore: 65, studyTime: 60, productivity: 67 },
    { day: 'Fri', sessions: 3, focusScore: 79, studyTime: 180, productivity: 85 },
    { day: 'Sat', sessions: 2, focusScore: 84, studyTime: 100, productivity: 80 },
    { day: 'Sun', sessions: 3, focusScore: 91, studyTime: 165, productivity: 94 }
  ],
  monthlyProgress: [
    { week: 'Week 1', hours: 18.5, sessions: 12, avgFocus: 75 },
    { week: 'Week 2', hours: 22.3, sessions: 15, avgFocus: 81 },
    { week: 'Week 3', hours: 19.8, sessions: 11, avgFocus: 78 },
    { week: 'Week 4', hours: 25.1, sessions: 18, avgFocus: 83 }
  ],
  studyPatterns: {
    bestTimeOfDay: '9-11 AM',
    optimalSessionLength: '45-60 minutes',
    mostProductiveDay: 'Wednesday',
    averageBreakTime: 15 // minutes
  },
  recentSessions: [
    { date: '2025-08-03', time: '14:30', duration: 45, focusScore: 85, status: 'excellent', notes: 'Great concentration' },
    { date: '2025-08-02', time: '16:15', duration: 60, focusScore: 72, status: 'good', notes: 'Slight distraction' },
    { date: '2025-08-02', time: '10:00', duration: 30, focusScore: 91, status: 'excellent', notes: 'Perfect morning session' },
    { date: '2025-08-01', time: '20:45', duration: 55, focusScore: 68, status: 'fair', notes: 'Evening fatigue' },
    { date: '2025-08-01', time: '11:30', duration: 40, focusScore: 89, status: 'excellent', notes: 'High energy' }
  ],
  insights: [
    { type: 'positive', message: 'Your focus scores have improved by 15% this week!', action: 'Keep up the momentum' },
    { type: 'warning', message: 'Consider taking more breaks during long sessions', action: 'Try the Pomodoro technique' },
    { type: 'tip', message: 'Your best focus time is between 9-11 AM', action: 'Schedule important tasks then' }
  ],
  achievements: [
    { title: '7-Day Streak', description: 'Studied for 7 consecutive days', earned: true, date: '2025-08-03' },
    { title: 'Focus Master', description: 'Achieved 90%+ focus score', earned: true, date: '2025-08-02' },
    { title: 'Marathon Learner', description: 'Study for 3+ hours in a day', earned: false, progress: 75 },
    { title: 'Early Bird', description: 'Complete morning sessions for a week', earned: false, progress: 42 }
  ]
}

export default function Analytics() {
  const router = useRouter()
  const [selectedPeriod, setSelectedPeriod] = useState('week')

  const formatTime = (minutes: number) => {
    const hours = Math.floor(minutes / 60)
    const mins = minutes % 60
    if (hours > 0) {
      return `${hours}h ${mins}m`
    }
    return `${mins}m`
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent': return 'text-green-600 dark:text-green-400'
      case 'good': return 'text-blue-600 dark:text-blue-400'
      case 'fair': return 'text-yellow-600 dark:text-yellow-400'
      case 'poor': return 'text-red-600 dark:text-red-400'
      default: return 'text-gray-600 dark:text-gray-400'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'excellent': return <CheckCircle className="h-4 w-4" />
      case 'good': return <Target className="h-4 w-4" />
      case 'fair': return <Clock className="h-4 w-4" />
      case 'poor': return <AlertTriangle className="h-4 w-4" />
      default: return <Activity className="h-4 w-4" />
    }
  }

  return (
    <div className="min-h-screen bg-white dark:bg-[#121212] p-4 md:p-6">
      <div className="mx-auto max-w-7xl space-y-6">
        {/* Header */}
        <div className="relative">
          <div className="absolute inset-0 bg-gradient-to-r from-[#f4895c]/10 to-[#f4a072]/10 rounded-3xl blur-3xl" />
          <button 
            className="relative flex items-center gap-6 p-6 rounded-2xl bg-white/50 dark:bg-[#1a1a1a]/50 backdrop-blur-xl border border-gray-200/50 dark:border-gray-800/50 animate-in fade-in-0 slide-in-from-top-8 duration-1000 hover:bg-white/70 dark:hover:bg-[#1a1a1a]/70 transition-all duration-300 hover:scale-[1.02] group w-full text-left"
            onClick={() => router.push('/home')}
          >
            <ArrowLeft className="h-6 w-6 text-gray-600 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white transition-all duration-300 group-hover:scale-110 group-hover:-translate-x-1" />
            <div className="flex-1">
              <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 dark:from-white dark:to-gray-300 bg-clip-text text-transparent animate-in slide-in-from-left-6 duration-1000 delay-300">
                Analytics Dashboard
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1 animate-in slide-in-from-left-4 duration-1000 delay-500">
                Track your study progress and insights
              </p>
            </div>
            <div className="hidden md:block p-3 rounded-xl bg-gradient-to-br from-[#f4895c] to-[#f4a072] text-white shadow-lg animate-in fade-in-0 zoom-in-50 duration-1000 delay-700 group-hover:shadow-xl group-hover:scale-110 transition-all duration-300">
              <BarChart3 className="h-6 w-6" />
            </div>
          </button>
        </div>

        {/* Period Selector */}
        <div className="flex gap-2 animate-in fade-in-0 slide-in-from-bottom-4 duration-1000 delay-300">
          {['day', 'week', 'month', 'all'].map((period) => (
            <Button
              key={period}
              variant={selectedPeriod === period ? "default" : "outline"}
              size="sm"
              onClick={() => setSelectedPeriod(period)}
              className={selectedPeriod === period 
                ? "bg-gradient-to-r from-[#f4895c] to-[#f4a072] text-white shadow-md hover:shadow-lg transition-all duration-300 hover:scale-105" 
                : "border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700 transition-all duration-300 hover:scale-105"
              }
            >
              {period.charAt(0).toUpperCase() + period.slice(1)}
            </Button>
          ))}
        </div>

        {/* Key Metrics - Simple 4-column layout */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 animate-in fade-in-0 slide-in-from-bottom-8 duration-1000 delay-500">
          <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm hover:shadow-md transition-all duration-300 hover:scale-[1.02] group">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm text-gray-600 dark:text-gray-400">Total Sessions</CardTitle>
                <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900/30">
                  <Calendar className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">{mockData.totalSessions}</div>
              <div className="flex items-center mt-1">
                <TrendingUp className="h-3 w-3 text-green-600 mr-1" />
                <span className="text-xs text-green-600">+12% this week</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm hover:shadow-md transition-all duration-300 hover:scale-[1.02] group">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm text-gray-600 dark:text-gray-400">Study Time</CardTitle>
                <div className="p-2 rounded-lg bg-orange-100 dark:bg-orange-900/30">
                  <Clock className="h-4 w-4 text-orange-600 dark:text-orange-400" />
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">{formatTime(mockData.totalStudyTime)}</div>
              <div className="flex items-center mt-1">
                <TrendingUp className="h-3 w-3 text-green-600 mr-1" />
                <span className="text-xs text-green-600">+8% this week</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm hover:shadow-md transition-all duration-300 hover:scale-[1.02] group">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm text-gray-600 dark:text-gray-400">Focus Score</CardTitle>
                <div className="p-2 rounded-lg bg-green-100 dark:bg-green-900/30">
                  <Brain className="h-4 w-4 text-green-600 dark:text-green-400" />
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">{mockData.averageFocus}%</div>
              <div className="flex items-center mt-1">
                <TrendingUp className="h-3 w-3 text-green-600 mr-1" />
                <span className="text-xs text-green-600">+15% this week</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm hover:shadow-md transition-all duration-300 hover:scale-[1.02] group">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm text-gray-600 dark:text-gray-400">Current Streak</CardTitle>
                <div className="p-2 rounded-lg bg-purple-100 dark:bg-purple-900/30">
                  <Flame className="h-4 w-4 text-purple-600 dark:text-purple-400" />
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">{mockData.currentStreak} days</div>
              <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                Best: {mockData.bestStreak} days
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Charts Section - Balanced Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-in fade-in-0 slide-in-from-bottom-8 duration-1000 delay-700">
          {/* Weekly Progress Chart */}
          <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm hover:shadow-md transition-all duration-300 hover:scale-[1.01]">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg text-gray-900 dark:text-white flex items-center gap-2">
                <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900/30">
                  <LineChart className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                </div>
                Weekly Focus Trend
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {mockData.weeklyData.map((day) => (
                  <div key={day.day} className="flex items-center gap-4">
                    <div className="w-10 text-sm text-gray-600 dark:text-gray-400 font-medium">{day.day}</div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm text-gray-700 dark:text-gray-300">{day.focusScore}%</span>
                        <span className="text-xs text-gray-500 dark:text-gray-400">{formatTime(day.studyTime)}</span>
                      </div>
                      <div className="relative h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                        <div 
                          className="absolute top-0 left-0 h-full bg-blue-500 rounded-full transition-all duration-1000"
                          style={{ width: `${day.focusScore}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Today's Summary & Weekly Goal */}
          <div className="space-y-6">
            {/* Today's Summary */}
            <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm hover:shadow-md transition-all duration-300 hover:scale-[1.01]">
              <CardHeader className="pb-4">
                <CardTitle className="text-lg text-gray-900 dark:text-white flex items-center gap-2">
                  <div className="p-2 rounded-lg bg-green-100 dark:bg-green-900/30">
                    <Calendar className="h-4 w-4 text-green-600 dark:text-green-400" />
                  </div>
                  Today&apos;s Summary
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50">
                    <div className="text-xl font-bold text-gray-900 dark:text-white">{mockData.todayStats.sessions}</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">Sessions</div>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50">
                    <div className="text-xl font-bold text-gray-900 dark:text-white">{formatTime(mockData.todayStats.studyTime)}</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">Study Time</div>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50">
                    <div className="text-xl font-bold text-gray-900 dark:text-white">{mockData.todayStats.focusScore}%</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">Focus Score</div>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50">
                    <div className="text-xl font-bold text-gray-900 dark:text-white">{formatTime(mockData.todayStats.avgFocusTime)}</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">Avg Session</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Weekly Goal Progress */}
            <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm hover:shadow-md transition-all duration-300 hover:scale-[1.01]">
              <CardHeader className="pb-4">
                <CardTitle className="text-lg text-gray-900 dark:text-white flex items-center gap-2">
                  <div className="p-2 rounded-lg bg-orange-100 dark:bg-orange-900/30">
                    <Target className="h-4 w-4 text-orange-600 dark:text-orange-400" />
                  </div>
                  Weekly Goal
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Progress</span>
                    <span className="text-sm font-semibold text-gray-900 dark:text-white">
                      {(mockData.totalStudyTime / 60).toFixed(1)} / {mockData.weeklyGoal}h
                    </span>
                  </div>
                  <div className="relative h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className="absolute top-0 left-0 h-full bg-orange-500 rounded-full transition-all duration-1000"
                      style={{ width: `${Math.min((mockData.totalStudyTime / 60 / mockData.weeklyGoal) * 100, 100)}%` }}
                    />
                  </div>
                  <div className="text-center">
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {Math.round((mockData.totalStudyTime / 60 / mockData.weeklyGoal) * 100)}% Complete
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Bottom Section - Clean Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-in fade-in-0 slide-in-from-bottom-8 duration-1000 delay-900">
          {/* Recent Sessions */}
          <Card className="lg:col-span-2 bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm hover:shadow-md transition-all duration-300 hover:scale-[1.01]">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg text-gray-900 dark:text-white flex items-center gap-2">
                <div className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800">
                  <Activity className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                </div>
                Recent Sessions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {mockData.recentSessions.slice(0, 5).map((session, index) => (
                  <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition-all duration-200">
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg ${getStatusColor(session.status)} bg-current bg-opacity-10`}>
                        {getStatusIcon(session.status)}
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium text-gray-900 dark:text-white">
                            {formatTime(session.duration)}
                          </span>
                          <span className="text-xs text-gray-500 dark:text-gray-400">
                            {session.time}
                          </span>
                        </div>
                        <div className="text-xs text-gray-600 dark:text-gray-400">
                          {session.date} • {session.notes}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-semibold text-gray-900 dark:text-white">
                        {session.focusScore}%
                      </div>
                      <div className={`text-xs capitalize ${getStatusColor(session.status)}`}>
                        {session.status}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Study Insights */}
          <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm hover:shadow-md transition-all duration-300 hover:scale-[1.01]">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg text-gray-900 dark:text-white flex items-center gap-2">
                <div className="p-2 rounded-lg bg-purple-100 dark:bg-purple-900/30">
                  <Brain className="h-4 w-4 text-purple-600 dark:text-purple-400" />
                </div>
                Study Insights
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Study Patterns */}
                <div className="space-y-3">
                  <div className="p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
                    <div className="text-xs text-blue-600 dark:text-blue-400 font-medium">Best Time</div>
                    <div className="text-sm font-bold text-blue-800 dark:text-blue-300">{mockData.studyPatterns.bestTimeOfDay}</div>
                  </div>
                  <div className="p-3 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800">
                    <div className="text-xs text-green-600 dark:text-green-400 font-medium">Optimal Length</div>
                    <div className="text-sm font-bold text-green-800 dark:text-green-300">{mockData.studyPatterns.optimalSessionLength}</div>
                  </div>
                  <div className="p-3 rounded-lg bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800">
                    <div className="text-xs text-purple-600 dark:text-purple-400 font-medium">Most Productive</div>
                    <div className="text-sm font-bold text-purple-800 dark:text-purple-300">{mockData.studyPatterns.mostProductiveDay}</div>
                  </div>
                </div>

                {/* AI Tips */}
                <div className="space-y-2">
                  <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">AI Tips</h4>
                  {mockData.insights.slice(0, 2).map((insight, index) => (
                    <div key={index} className={`p-2 rounded text-xs ${
                      insight.type === 'positive' ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300' :
                      insight.type === 'warning' ? 'bg-yellow-50 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-300' :
                      'bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300'
                    }`}>
                      {insight.message}
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Achievements - Simple Grid */}
        <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm hover:shadow-md transition-all duration-300 hover:scale-[1.01] animate-in fade-in-0 slide-in-from-bottom-8 duration-1000 delay-1100">
          <CardHeader className="pb-4">
            <CardTitle className="text-lg text-gray-900 dark:text-white flex items-center gap-2">
              <div className="p-2 rounded-lg bg-yellow-100 dark:bg-yellow-900/30">
                <Target className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
              </div>
              Achievements
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {mockData.achievements.map((achievement, index) => (
                <div key={index} className={`p-4 rounded-lg border transition-all duration-200 hover:scale-105 ${
                  achievement.earned 
                    ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800' 
                    : 'bg-gray-50 dark:bg-gray-800/50 border-gray-200 dark:border-gray-700'
                }`}>
                  <div className="text-center">
                    <div className={`mx-auto mb-2 w-10 h-10 rounded-full flex items-center justify-center ${
                      achievement.earned 
                        ? 'bg-green-500 text-white' 
                        : 'bg-gray-300 dark:bg-gray-600 text-gray-600 dark:text-gray-300'
                    }`}>
                      {achievement.earned ? <CheckCircle className="h-5 w-5" /> : <Target className="h-5 w-5" />}
                    </div>
                    <div className={`font-medium text-sm mb-1 ${
                      achievement.earned 
                        ? 'text-green-800 dark:text-green-300' 
                        : 'text-gray-700 dark:text-gray-300'
                    }`}>
                      {achievement.title}
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400 mb-2">{achievement.description}</div>
                    {achievement.earned ? (
                      <div className="text-xs text-green-600 dark:text-green-400 font-medium">
                        ✓ Completed
                      </div>
                    ) : (
                      <div>
                        <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                          {achievement.progress}%
                        </div>
                        <div className="relative h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                          <div 
                            className="absolute top-0 left-0 h-full bg-blue-500 rounded-full transition-all duration-1000"
                            style={{ width: `${achievement.progress}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
