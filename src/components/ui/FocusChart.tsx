"use client"

import { useState, useEffect } from "react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface FocusDataPoint {
  timestamp: string
  time: string
  focusScore: number
}

interface FocusChartProps {
  currentFocusScore: number
  isRecording: boolean
  className?: string
}

export function FocusChart({ currentFocusScore, isRecording, className = "" }: FocusChartProps) {
  const [focusData, setFocusData] = useState<FocusDataPoint[]>([])

  // Load data from localStorage on mount
  useEffect(() => {
    const savedData = localStorage.getItem('focusChartData')
    if (savedData) {
      try {
        const parsedData = JSON.parse(savedData)
        if (Array.isArray(parsedData)) {
          setFocusData(parsedData)
        }
      } catch (error) {
        console.error('Error loading focus chart data:', error)
        localStorage.removeItem('focusChartData')
      }
    }
  }, [])

  // Add new focus score data point
  useEffect(() => {
    if (isRecording && currentFocusScore > 0) {
      const now = new Date()
      const newDataPoint: FocusDataPoint = {
        timestamp: now.toISOString(),
        time: now.toLocaleTimeString('en-US', { 
          hour12: false, 
          hour: '2-digit', 
          minute: '2-digit',
          second: '2-digit'
        }),
        focusScore: currentFocusScore
      }

      setFocusData(prevData => {
        const updatedData = [...prevData, newDataPoint].slice(-50) // Keep last 50 points
        localStorage.setItem('focusChartData', JSON.stringify(updatedData))
        return updatedData
      })
    }
  }, [currentFocusScore, isRecording])

  // Clear data function
  const clearData = () => {
    setFocusData([])
    localStorage.removeItem('focusChartData')
  }

  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white dark:bg-gray-800 p-3 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg">
          <p className="text-sm font-medium text-gray-900 dark:text-white">{`Time: ${label}`}</p>
          <p className="text-sm text-orange-600 dark:text-orange-400">
            {`Focus: ${payload[0].value}%`}
          </p>
        </div>
      )
    }
    return null
  }

  return (
    <div className={`bg-white dark:bg-[#1a1a1a] border border-gray-200 dark:border-gray-800 rounded-xl p-4 shadow-sm ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-gray-900 dark:text-white">Focus Score Timeline</h3>
        {focusData.length > 0 && (
          <button
            onClick={clearData}
            className="text-xs text-gray-500 hover:text-red-500 transition-colors duration-200"
          >
            Clear Data
          </button>
        )}
      </div>
      
      {focusData.length === 0 ? (
        <div className="h-48 flex items-center justify-center text-gray-400">
          <div className="text-center">
            <div className="text-sm mb-2">📊</div>
            <p className="text-xs">Start recording to see your focus timeline</p>
          </div>
        </div>
      ) : (
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={focusData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis 
                dataKey="time" 
                tick={{ fontSize: 10 }}
                stroke="#6b7280"
              />
              <YAxis 
                domain={[0, 100]} 
                tick={{ fontSize: 10 }}
                stroke="#6b7280"
              />
              <Tooltip content={<CustomTooltip />} />
              <Line 
                type="monotone" 
                dataKey="focusScore" 
                stroke="#f4895c" 
                strokeWidth={2}
                dot={{ fill: '#f4895c', strokeWidth: 2, r: 3 }}
                activeDot={{ r: 5, stroke: '#f4895c', strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      
      {focusData.length > 0 && (
        <div className="mt-3 text-xs text-gray-500 dark:text-gray-400 text-center">
          Showing last {focusData.length} data points • Updates during recording
        </div>
      )}
    </div>
  )
}
