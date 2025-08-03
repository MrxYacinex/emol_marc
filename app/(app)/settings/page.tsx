"use client"

import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { useTheme } from "../../../src/contexts/ThemeContext"
import {
  ArrowLeft,
  Monitor,
  Brain,
  Camera,
  Bell,
  Shield,
  Info,
  Sun,
  Moon,
  Palette,
} from "lucide-react"

export default function SettingsPage() {
  const router = useRouter()
  const { theme, setTheme } = useTheme()

  return (
    <div className="min-h-screen bg-white dark:bg-[#121212] p-4 md:p-6">
      <div className="mx-auto max-w-4xl space-y-6">
        {/* Header */}
        <div className="flex items-center gap-4 mb-8">
          <Button 
            variant="ghost" 
            size="icon" 
            className="text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
            onClick={() => router.push('/home')}
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div>
            <h1 className="text-2xl font-semibold text-gray-900 dark:text-white">Settings</h1>
            <p className="text-sm text-gray-600 dark:text-gray-400">Customize your study companion experience</p>
          </div>
        </div>

        <div className="space-y-6">
          {/* Theme Settings */}
          <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg text-gray-900 dark:text-white flex items-center gap-2">
                <Palette className="h-5 w-5" />
                Appearance
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">Choose your preferred theme</p>
                <div className="grid grid-cols-3 gap-3">
                  <Button
                    variant={theme === 'light' ? 'default' : 'outline'}
                    onClick={() => setTheme('light')}
                    className={`h-12 flex flex-col items-center gap-1 ${
                      theme === 'light' 
                        ? 'bg-blue-600 hover:bg-blue-700 text-white' 
                        : 'border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 bg-white dark:bg-[#121212] hover:bg-gray-50 dark:hover:bg-gray-700'
                    }`}
                  >
                    <Sun className="h-4 w-4" />
                    <span className="text-xs">Light</span>
                  </Button>
                  <Button
                    variant={theme === 'dark' ? 'default' : 'outline'}
                    onClick={() => setTheme('dark')}
                    className={`h-12 flex flex-col items-center gap-1 ${
                      theme === 'dark' 
                        ? 'bg-blue-600 hover:bg-blue-700 text-white' 
                        : 'border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 bg-white dark:bg-[#121212] hover:bg-gray-50 dark:hover:bg-gray-700'
                    }`}
                  >
                    <Moon className="h-4 w-4" />
                    <span className="text-xs">Dark</span>
                  </Button>
                  <Button
                    variant={theme === 'auto' ? 'default' : 'outline'}
                    onClick={() => setTheme('auto')}
                    className={`h-12 flex flex-col items-center gap-1 ${
                      theme === 'auto' 
                        ? 'bg-blue-600 hover:bg-blue-700 text-white' 
                        : 'border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 bg-white dark:bg-[#121212] hover:bg-gray-50 dark:hover:bg-gray-700'
                    }`}
                  >
                    <Monitor className="h-4 w-4" />
                    <span className="text-xs">Auto</span>
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* AI Analysis Settings */}
          <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg text-gray-900 dark:text-white flex items-center gap-2">
                <Brain className="h-5 w-5" />
                AI Analysis
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <div className="text-sm font-medium text-gray-900 dark:text-white">Real-time Analysis</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">Enable AI-powered focus tracking</div>
                  </div>
                  <div className="w-10 h-6 bg-blue-600 rounded-full relative cursor-pointer">
                    <div className="w-4 h-4 bg-white rounded-full absolute top-1 right-1 transition-all"></div>
                  </div>
                </div>

                <Separator className="dark:bg-gray-700" />

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <div className="text-sm font-medium text-gray-900 dark:text-white">Fatigue Detection</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">Alert when signs of tiredness are detected</div>
                  </div>
                  <div className="w-10 h-6 bg-blue-600 rounded-full relative cursor-pointer">
                    <div className="w-4 h-4 bg-white rounded-full absolute top-1 right-1 transition-all"></div>
                  </div>
                </div>

                <Separator className="dark:bg-gray-700" />

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <div className="text-sm font-medium text-gray-900 dark:text-white">Hand Tracking</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">Monitor hand movements and gestures</div>
                  </div>
                  <div className="w-10 h-6 bg-blue-600 rounded-full relative cursor-pointer">
                    <div className="w-4 h-4 bg-white rounded-full absolute top-1 right-1 transition-all"></div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Camera Settings */}
          <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg text-gray-900 dark:text-white flex items-center gap-2">
                <Camera className="h-5 w-5" />
                Camera & Video
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <div className="text-sm font-medium text-gray-900 dark:text-white">Auto-start Audio</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">Automatically enable microphone</div>
                  </div>
                  <div className="w-10 h-6 bg-gray-300 dark:bg-gray-600 rounded-full relative cursor-pointer">
                    <div className="w-4 h-4 bg-white rounded-full absolute top-1 left-1 transition-all"></div>
                  </div>
                </div>

                <Separator className="dark:bg-gray-700" />

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <div className="text-sm font-medium text-gray-900 dark:text-white">High Quality Video</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">Use higher resolution for better analysis</div>
                  </div>
                  <div className="w-10 h-6 bg-blue-600 rounded-full relative cursor-pointer">
                    <div className="w-4 h-4 bg-white rounded-full absolute top-1 right-1 transition-all"></div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Notifications */}
          <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg text-gray-900 dark:text-white flex items-center gap-2">
                <Bell className="h-5 w-5" />
                Notifications
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <div className="text-sm font-medium text-gray-900 dark:text-white">Focus Reminders</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">Get alerts when attention drifts</div>
                  </div>
                  <div className="w-10 h-6 bg-blue-600 rounded-full relative cursor-pointer">
                    <div className="w-4 h-4 bg-white rounded-full absolute top-1 right-1 transition-all"></div>
                  </div>
                </div>

                <Separator className="dark:bg-gray-700" />

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <div className="text-sm font-medium text-gray-900 dark:text-white">Break Suggestions</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">Recommended rest periods</div>
                  </div>
                  <div className="w-10 h-6 bg-blue-600 rounded-full relative cursor-pointer">
                    <div className="w-4 h-4 bg-white rounded-full absolute top-1 right-1 transition-all"></div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Privacy */}
          <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg text-gray-900 dark:text-white flex items-center gap-2">
                <Shield className="h-5 w-5" />
                Privacy & Data
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <div className="text-sm font-medium text-gray-900 dark:text-white">Local Processing</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">All analysis happens on your device</div>
                  </div>
                  <div className="w-10 h-6 bg-green-600 rounded-full relative cursor-pointer">
                    <div className="w-4 h-4 bg-white rounded-full absolute top-1 right-1 transition-all"></div>
                  </div>
                </div>

                <Separator className="dark:bg-gray-700" />

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <div className="text-sm font-medium text-gray-900 dark:text-white">Save Session Data</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">Store analytics for progress tracking</div>
                  </div>
                  <div className="w-10 h-6 bg-blue-600 rounded-full relative cursor-pointer">
                    <div className="w-4 h-4 bg-white rounded-full absolute top-1 right-1 transition-all"></div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* About */}
          <Card className="bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 shadow-sm">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg text-gray-900 dark:text-white flex items-center gap-2">
                <Info className="h-5 w-5" />
                About
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between items-center py-2">
                  <span className="text-gray-600 dark:text-gray-400">Version</span>
                  <span className="text-gray-900 dark:text-white font-medium">1.0.0</span>
                </div>
                <Separator className="dark:bg-gray-700" />
                <div className="flex justify-between items-center py-2">
                  <span className="text-gray-600 dark:text-gray-400">AI Model</span>
                  <span className="text-gray-900 dark:text-white font-medium">Emotion Analysis v2.1</span>
                </div>
                <Separator className="dark:bg-gray-700" />
                <div className="flex justify-between items-center py-2">
                  <span className="text-gray-600 dark:text-gray-400">Framework</span>
                  <span className="text-gray-900 dark:text-white font-medium">Next.js 15</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
