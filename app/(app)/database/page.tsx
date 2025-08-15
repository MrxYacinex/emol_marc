"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Database, Download, Eye, AlertCircle, ArrowLeft, RefreshCw } from "lucide-react"

export default function DatabaseViewer() {
  const router = useRouter()
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('analyses')

  const loadData = async () => {
    setLoading(true)
    setError(null)
    try {
      console.log('🔄 Loading data from Flask server...')
      const response = await fetch('http://localhost:5000/api/database/export')
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const contentType = response.headers.get('content-type')
      if (!contentType || !contentType.includes('application/json')) {
        const text = await response.text()
        console.error('Non-JSON response:', text.substring(0, 200) + '...')
        throw new Error('Server returned non-JSON response. Make sure the Flask server is running on port 5000.')
      }
      
      const result = await response.json()
      console.log('📊 Received data:', result)
      
      if (result.success) {
        setData(result.data)
        console.log('✅ Data loaded successfully:', {
          sessions: result.data.sessions?.length || 0,
          analyses: result.data.analyses?.length || 0,
          summaries: result.data.summaries?.length || 0,
          source: result.source
        })
      } else {
        throw new Error(result.error || 'Failed to load data')
      }
    } catch (error) {
      console.error('Error loading data:', error)
      if (error instanceof Error) {
        if (error.message.includes('fetch')) {
          setError('Cannot connect to Flask server. Make sure it\'s running on http://localhost:5000')
        } else {
          setError(error.message)
        }
      } else {
        setError('Unknown error occurred')
      }
    } finally {
      setLoading(false)
    }
  }

  const downloadData = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/database/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format: 'json' })
      })
      
      if (!response.ok) {
        throw new Error(`Download failed: ${response.status}`)
      }
      
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `study_data_${Date.now()}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Error downloading data:', error)
      setError(error instanceof Error ? error.message : 'Download failed')
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString()
  }

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    if (hours > 0) {
      return `${hours}h ${minutes}m`
    }
    return `${minutes}m`
  }

  const getStatusBadge = (status: string, type: 'attention' | 'fatigue') => {
    const colors = {
      attention: {
        'focused': 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400',
        'abgelenkt': 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400',
        'müde': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400'
      },
      fatigue: {
        'awake': 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400',
        'tired': 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
      }
    }
    
    const colorClass = colors[type][status] || 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400'
    
    return (
      <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${colorClass}`}>
        {status}
      </span>
    )
  }

  const getFocusScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600 dark:text-green-400'
    if (score >= 60) return 'text-yellow-600 dark:text-yellow-400'
    return 'text-red-600 dark:text-red-400'
  }

  const renderAnalysesTable = () => {
    if (!data?.analyses || data.analyses.length === 0) {
      return (
        <div className="text-center py-8">
          <Database className="h-12 w-12 text-gray-400 mx-auto mb-3" />
          <p className="text-gray-100 dark:text-gray-100">No analyses data found</p>
          <p className="text-xs text-gray-100 dark:text-gray-100 mt-2">
            Start a study session to generate analysis data
          </p>
        </div>
      )
    }

    return (
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <th className="text-left p-3 font-medium text-white dark:text-white">Timestamp</th>
              <th className="text-left p-3 font-medium text-white dark:text-white">Session</th>
              <th className="text-center p-3 font-medium text-white dark:text-white">Focus Score</th>
              <th className="text-center p-3 font-medium text-white dark:text-white">Attention</th>
              <th className="text-center p-3 font-medium text-white dark:text-white">Fatigue</th>
              <th className="text-center p-3 font-medium text-white dark:text-white">Gaze</th>
              <th className="text-center p-3 font-medium text-white dark:text-white">EAR</th>
              <th className="text-center p-3 font-medium text-white dark:text-white">Hand Gestures</th>
              <th className="text-center p-3 font-medium text-white dark:text-white">Method</th>
            </tr>
          </thead>
          <tbody>
            {data.analyses.map((analysis: any, index: number) => (
              <tr 
                key={analysis.id || index} 
                className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors duration-200"
              >
                <td className="p-3 text-sm text-white dark:text-white">
                  {formatTimestamp(analysis.timestamp)}
                </td>
                <td className="p-3 text-sm text-gray-600 dark:text-gray-400 font-mono">
                  {analysis.session_id?.substring(0, 8)}...
                </td>
                <td className="p-3 text-center">
                  <span className={`text-lg font-bold ${getFocusScoreColor(analysis.focus_score || 0)}`}>
                    {analysis.focus_score || 0}%
                  </span>
                </td>
                <td className="p-3 text-center">
                  {getStatusBadge(analysis.attention_status || 'No Data', 'attention')}
                </td>
                <td className="p-3 text-center">
                  {getStatusBadge(analysis.fatigue_status || 'awake', 'fatigue')}
                </td>
                <td className="p-3 text-center text-sm text-gray-600 dark:text-gray-400">
                  {analysis.gaze_direction?.replace('/', ' / ') || 'unknown'}
                </td>
                <td className="p-3 text-center text-sm text-gray-900 dark:text-white font-mono">
                  {analysis.ear_value?.toFixed(2) || '0.00'}
                </td>
                <td className="p-3 text-center">
                  <div className="flex flex-col gap-1">
                    {analysis.hand_fatigue_detected && (
                      <span className="inline-block px-2 py-1 bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400 rounded text-xs">
                        Fatigue
                      </span>
                    )}
                    {analysis.hand_at_head && (
                      <span className="inline-block px-2 py-1 bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400 rounded text-xs">
                        Hand@Head
                      </span>
                    )}
                    {analysis.playing_with_hair && (
                      <span className="inline-block px-2 py-1 bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-400 rounded text-xs">
                        Hair Play
                      </span>
                    )}
                    {!analysis.hand_fatigue_detected && !analysis.hand_at_head && !analysis.playing_with_hair && (
                      <span className="inline-block px-2 py-1 bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400 rounded text-xs">
                        Normal
                      </span>
                    )}
                  </div>
                </td>
                <td className="p-3 text-center">
                  <span className={`inline-block px-2 py-1 rounded text-xs font-medium ${
                    analysis.method_used === 'dlib' 
                      ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400'
                      : 'bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-400'
                  }`}>
                    {analysis.method_used || 'unknown'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }

  const renderSessionsTable = () => {
    if (!data?.sessions || data.sessions.length === 0) {
      return (
        <div className="text-center py-8">
          <Database className="h-12 w-12 text-gray-400 mx-auto mb-3" />
          <p className="text-gray-500 dark:text-gray-400">No sessions data found</p>
        </div>
      )
    }

    return (
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <th className="text-left p-3 font-medium text-white dark:text-white">Session ID</th>
              <th className="text-left p-3 font-medium text-white dark:text-white">Start Time</th>
              <th className="text-left p-3 font-medium text-white dark:text-white">Duration</th>
              <th className="text-center p-3 font-medium text-white dark:text-white">Avg Focus</th>
              <th className="text-center p-3 font-medium text-white dark:text-white">Attention</th>
              <th className="text-center p-3 font-medium text-white dark:text-white">Analyses</th>
              <th className="text-center p-3 font-medium text-white dark:text-white">Status</th>
            </tr>
          </thead>
          <tbody>
            {data.sessions.map((session: any, index: number) => (
              <tr 
                key={session.session_id || index} 
                className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors duration-200"
              >
                <td className="p-3 text-sm text-gray-900 dark:text-white font-mono">
                  {session.session_id}
                </td>
                <td className="p-3 text-sm text-gray-600 dark:text-gray-400">
                  {formatTimestamp(session.start_time)}
                </td>
                <td className="p-3 text-sm text-gray-900 dark:text-white">
                  {formatDuration(session.total_duration || 0)}
                </td>
                <td className="p-3 text-center">
                  <span className={`text-lg font-bold ${getFocusScoreColor(session.average_focus_score || 0)}`}>
                    {session.average_focus_score?.toFixed(1) || 0}%
                  </span>
                </td>
                <td className="p-3 text-center">
                  {getStatusBadge(session.attention_summary || 'No Data', 'attention')}
                </td>
                <td className="p-3 text-center text-sm text-gray-900 dark:text-white">
                  {session.total_analyses || 0}
                </td>
                <td className="p-3 text-center">
                  <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${
                    session.completed 
                      ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
                      : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400'
                  }`}>
                    {session.completed ? 'Completed' : 'In Progress'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }

  const renderSummariesTable = () => {
    if (!data?.summaries || data.summaries.length === 0) {
      return (
        <div className="text-center py-8">
          <Database className="h-12 w-12 text-gray-400 mx-auto mb-3" />
          <p className="text-gray-500 dark:text-gray-400">No summaries data found</p>
        </div>
      )
    }

    return (
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <th className="text-left p-3 font-medium text-gray-900 dark:text-white">Date</th>
              <th className="text-center p-3 font-medium text-gray-900 dark:text-white">Sessions</th>
              <th className="text-center p-3 font-medium text-gray-900 dark:text-white">Study Time</th>
              <th className="text-center p-3 font-medium text-gray-900 dark:text-white">Avg Focus</th>
              <th className="text-center p-3 font-medium text-gray-900 dark:text-white">Best Session</th>
              <th className="text-center p-3 font-medium text-gray-900 dark:text-white">XP Earned</th>
            </tr>
          </thead>
          <tbody>
            {data.summaries.map((summary: any, index: number) => (
              <tr 
                key={summary.date || index} 
                className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors duration-200"
              >
                <td className="p-3 text-sm text-gray-900 dark:text-white font-medium">
                  {new Date(summary.date).toLocaleDateString()}
                </td>
                <td className="p-3 text-center text-sm text-gray-900 dark:text-white">
                  {summary.total_sessions || 0}
                </td>
                <td className="p-3 text-center text-sm text-gray-900 dark:text-white">
                  {formatDuration(summary.total_study_time || 0)}
                </td>
                <td className="p-3 text-center">
                  <span className={`text-lg font-bold ${getFocusScoreColor(summary.average_focus_score || 0)}`}>
                    {summary.average_focus_score?.toFixed(1) || 0}%
                  </span>
                </td>
                <td className="p-3 text-center text-sm text-gray-600 dark:text-gray-400 font-mono">
                  {summary.best_session?.substring(0, 8)}...
                </td>
                <td className="p-3 text-center">
                  <span className="text-lg font-bold text-purple-600 dark:text-purple-400">
                    {summary.total_xp || 0}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-orange-50/30 dark:from-[#0a0a0a] dark:via-[#121212] dark:to-[#1a1a1a] p-4 md:p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => router.push('/home')}
              className="text-gray-600 dark:text-gray-400 hover:bg-[#f4895c]/20 hover:text-[#f4895c] transition-all duration-300 hover:scale-110 rounded-xl"
            >
              <ArrowLeft className="h-5 w-5" />
            </Button>
            <div className="flex items-center gap-3">
              <Database className="h-8 w-8 text-blue-600" />
              <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 dark:from-white dark:to-gray-300 bg-clip-text text-transparent">
                Database Viewer
              </h1>
            </div>
          </div>
          <div className="flex gap-3">
            <Button onClick={loadData} disabled={loading} variant="outline">
              {loading ? (
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Eye className="h-4 w-4 mr-2" />
              )}
              {loading ? 'Loading...' : 'Refresh'}
            </Button>
            <Button onClick={downloadData} disabled={!data || loading}>
              <Download className="h-4 w-4 mr-2" />
              Download JSON
            </Button>
          </div>
        </div>

        {/* Connection Status */}
        <Card className={`border-2 ${error ? 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20' : 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20'}`}>
          <CardContent className="p-4">
            <div className={`flex items-center gap-2 ${error ? 'text-red-800 dark:text-red-300' : 'text-green-800 dark:text-green-300'}`}>
              <div className={`h-3 w-3 rounded-full ${error ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`} />
              <span className="font-medium">
                {error ? 'Connection Error:' : 'Connected to Flask Server'}
              </span>
              {error && <span>{error}</span>}
            </div>
            {error && (
              <div className="mt-2 text-sm text-red-600 dark:text-red-400">
                Make sure the Flask server is running: <code>python ai_model/combined.py</code>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Tabs */}
        <div className="flex gap-2 border-b border-gray-200 dark:border-gray-700">
          {['analyses', 'sessions', 'summaries'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 font-medium rounded-t-lg transition-all duration-300 ${
                activeTab === tab
                  ? 'bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 border-b-2 border-blue-600'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-800/50'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)} 
              {data && `(${data[tab]?.length || 0})`}
            </button>
          ))}
        </div>

        {/* Content */}
        <Card className="bg-white/80 dark:bg-[#1a1a1a]/80 backdrop-blur-xl border border-gray-200/50 dark:border-gray-800/50 shadow-xl">
          <CardHeader>
            <CardTitle className="text-xl text-gray-900 dark:text-white">
              {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Data
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="text-center py-8">
                <RefreshCw className="h-12 w-12 text-gray-400 mx-auto mb-3 animate-spin" />
                <p className="text-gray-500 dark:text-gray-400">Loading database...</p>
              </div>
            ) : data ? (
              <div className="space-y-4">
                {activeTab === 'analyses' && renderAnalysesTable()}
                {activeTab === 'sessions' && renderSessionsTable()}
                {activeTab === 'summaries' && renderSummariesTable()}
              </div>
            ) : (
              <div className="text-center py-8">
                <Database className="h-12 w-12 text-gray-400 mx-auto mb-3" />
                <p className="text-gray-500 dark:text-gray-400">
                  Click refresh to load data
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Summary Stats */}
        {data && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="bg-white/80 dark:bg-[#1a1a1a]/80 backdrop-blur-xl border border-gray-200/50 dark:border-gray-800/50 shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105">
              <CardContent className="p-6 text-center">
                <div className="text-3xl font-bold text-blue-600 mb-2">
                  {data.sessions?.length || 0}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Total Sessions</div>
              </CardContent>
            </Card>
            <Card className="bg-white/80 dark:bg-[#1a1a1a]/80 backdrop-blur-xl border border-gray-200/50 dark:border-gray-800/50 shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105">
              <CardContent className="p-6 text-center">
                <div className="text-3xl font-bold text-green-600 mb-2">
                  {data.analyses?.length || 0}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">AI Analyses</div>
              </CardContent>
            </Card>
            <Card className="bg-white/80 dark:bg-[#1a1a1a]/80 backdrop-blur-xl border border-gray-200/50 dark:border-gray-800/50 shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105">
              <CardContent className="p-6 text-center">
                <div className="text-3xl font-bold text-purple-600 mb-2">
                  {data.summaries?.length || 0}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Daily Summaries</div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Instructions */}
        <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-200 dark:border-blue-800">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-blue-800 dark:text-blue-300">
              <Database className="h-5 w-5" />
              <span className="font-medium">Database Status:</span>
              <span>Connected to Flask server. Currently showing mock data structure.</span>
            </div>
            <div className="mt-2 text-sm text-blue-600 dark:text-blue-400">
              • Make sure Flask server is running: <code>python ai_model/combined.py</code><br/>
              • Database integration will store real session data once implemented
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
