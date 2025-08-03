"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { useTheme } from "../../../src/contexts/ThemeContext"
import { LucideIcon } from "lucide-react"
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
  Volume2,
  VolumeX,
  Eye,
  Zap,
  Clock,
  Heart,
  Settings,
  Database,
  Smartphone,
  CheckCircle,
  Lightbulb,
  Target,
  TrendingUp,
  Download,
  RotateCcw,
  HelpCircle,
  ExternalLink,
} from "lucide-react"

export default function SettingsPage() {
  const router = useRouter()
  const { theme, setTheme } = useTheme()
  
  // State for toggle switches
  const [realtimeAnalysis, setRealtimeAnalysis] = useState(true)
  const [fatigueDetection, setFatigueDetection] = useState(true)
  const [handTracking, setHandTracking] = useState(true)
  const [autoAudio, setAutoAudio] = useState(false)
  const [highQualityVideo, setHighQualityVideo] = useState(true)
  const [focusReminders, setFocusReminders] = useState(true)
  const [breakSuggestions, setBreakSuggestions] = useState(true)
  const [localProcessing, setLocalProcessing] = useState(true)
  const [saveSessionData, setSaveSessionData] = useState(true)
  const [notifications, setNotifications] = useState(true)

  // Toggle component
  const ToggleSwitch = ({ 
    isOn, 
    onToggle, 
    color = "bg-gradient-to-r from-[#f4895c] to-[#f4a072]",
    disabled = false 
  }: { 
    isOn: boolean
    onToggle: () => void
    color?: string
    disabled?: boolean
  }) => (
    <div 
      className={`relative w-12 h-6 rounded-full cursor-pointer transition-all duration-300 transform hover:scale-105 ${
        disabled ? 'opacity-50 cursor-not-allowed' : ''
      } ${
        isOn 
          ? color
          : 'bg-gray-300 dark:bg-gray-600'
      }`}
      onClick={disabled ? undefined : onToggle}
    >
      <div 
        className={`absolute w-5 h-5 bg-white rounded-full top-0.5 transition-all duration-300 transform ${
          isOn ? 'translate-x-6 shadow-lg' : 'translate-x-0.5 shadow-md'
        }`}
      />
      {isOn && (
        <div className="absolute inset-0 rounded-full bg-orange-100/40 animate-ping" />
      )}
    </div>
  )

  const SettingCard = ({ 
    icon: Icon, 
    title, 
    children, 
    gradient = false,
    className = "" 
  }: {
    icon: LucideIcon
    title: string
    children: React.ReactNode
    gradient?: boolean
    className?: string
  }) => (
    <Card className={`group relative overflow-hidden transition-all duration-500 hover:scale-[1.02] hover:shadow-xl bg-white dark:bg-[#1a1a1a] border-gray-200 dark:border-gray-800 ${className}`}>
      {gradient && (
        <div className="absolute inset-0 bg-gradient-to-br from-[#f4895c]/5 to-[#f4a072]/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
      )}
      <CardHeader className="pb-4 relative">
        <CardTitle className="text-lg text-gray-900 dark:text-white flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-gradient-to-br from-[#f4895c] to-[#f4a072] text-white shadow-md group-hover:shadow-lg transition-all duration-300 group-hover:scale-110 group-hover:rotate-12">
            <Icon className="h-5 w-5" />
          </div>
          <span className="group-hover:translate-x-1 transition-transform duration-300">{title}</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="relative">
        {children}
      </CardContent>
    </Card>
  )

  const SettingItem = ({ 
    title, 
    description, 
    isOn, 
    onToggle, 
    icon: Icon,
    color,
    disabled = false,
    badge = null
  }: {
    title: string
    description: string
    isOn: boolean
    onToggle: () => void
    icon?: LucideIcon
    color?: string
    disabled?: boolean
    badge?: string | null
  }) => (
    <div className="group flex items-center justify-between p-4 rounded-xl hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-all duration-300">
      <div className="flex items-center gap-3 flex-1">
        {Icon && (
          <div className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 group-hover:bg-gray-200 dark:group-hover:bg-gray-700 transition-colors duration-300">
            <Icon className="h-4 w-4 text-gray-600 dark:text-gray-400" />
          </div>
        )}
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <div className="text-sm font-medium text-gray-900 dark:text-white group-hover:translate-x-1 transition-transform duration-300">
              {title}
            </div>
            {badge && (
              <span className="text-xs px-2 py-1 rounded-full bg-gradient-to-r from-[#f4c572] to-[#f4bf62] text-white font-medium shadow-sm">
                {badge}
              </span>
            )}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400 group-hover:text-gray-700 dark:group-hover:text-gray-300 transition-colors duration-300">
            {description}
          </div>
        </div>
      </div>
      <ToggleSwitch isOn={isOn} onToggle={onToggle} color={color} disabled={disabled} />
    </div>
  )

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white dark:from-[#121212] dark:to-[#1a1a1a] p-4 md:p-6">
      <div className="mx-auto max-w-5xl space-y-8">
        {/* Modern Header */}
        <div className="relative">
          <div className="absolute inset-0 bg-gradient-to-r from-[#f4895c]/10 to-[#f4a072]/10 rounded-3xl blur-3xl" />
          <button 
            className="relative flex items-center gap-6 p-6 rounded-2xl bg-white/50 dark:bg-[#1a1a1a]/50 backdrop-blur-xl border border-gray-200/50 dark:border-gray-800/50 animate-in fade-in-0 slide-in-from-top-8 duration-1000 hover:bg-white/70 dark:hover:bg-[#1a1a1a]/70 transition-all duration-300 hover:scale-[1.02] group w-full text-left"
            onClick={() => router.push('/home')}
          >
            <ArrowLeft className="h-6 w-6 text-gray-600 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white transition-all duration-300 group-hover:scale-110 group-hover:-translate-x-1" />
            <div className="flex-1">
              <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 dark:from-white dark:to-gray-300 bg-clip-text text-transparent animate-in slide-in-from-left-6 duration-1000 delay-300">
                Settings
              </h1>
            </div>
            <div className="hidden md:block p-3 rounded-xl bg-gradient-to-br from-[#f4895c] to-[#f4a072] text-white shadow-lg animate-in fade-in-0 zoom-in-50 duration-1000 delay-700 group-hover:shadow-xl group-hover:scale-110 transition-all duration-300">
              <Settings className="h-6 w-6" />
            </div>
          </button>
        </div>

        <div className="grid gap-6 animate-in fade-in-0 slide-in-from-bottom-8 duration-1000 delay-300">
          {/* Theme Settings */}
          <SettingCard icon={Palette} title="Appearance & Theme" gradient>
            <div className="space-y-6">
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Personalize the visual experience to match your preferences
              </p>
              <div className="grid grid-cols-3 gap-4">
                {[
                  { key: 'light', icon: Sun, label: 'Light', desc: 'Clean & bright' },
                  { key: 'dark', icon: Moon, label: 'Dark', desc: 'Easy on eyes' },
                  { key: 'auto', icon: Monitor, label: 'Auto', desc: 'System sync' }
                ].map(({ key, icon: Icon, label, desc }) => (
                  <Button
                    key={key}
                    variant="outline"
                    onClick={() => setTheme(key as 'light' | 'dark' | 'auto')}
                    className={`group h-auto p-4 flex flex-col items-center gap-3 transition-all duration-500 hover:scale-105 ${
                      theme === key 
                        ? 'border-[#f4895c] bg-gradient-to-br from-[#f4895c]/10 to-[#f4a072]/10 text-[#f4895c] shadow-lg' 
                        : 'border-gray-200 dark:border-gray-700 hover:border-[#f4895c]/50 hover:bg-gray-50 dark:hover:bg-gray-800/50'
                    }`}
                  >
                    <div className={`p-3 rounded-xl transition-all duration-300 ${
                      theme === key 
                        ? 'bg-gradient-to-br from-[#f4895c] to-[#f4a072] text-white shadow-md' 
                        : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 group-hover:bg-[#f4895c]/10'
                    }`}>
                      <Icon className="h-5 w-5" />
                    </div>
                    <div className="text-center">
                      <div className="font-medium text-sm">{label}</div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">{desc}</div>
                    </div>
                  </Button>
                ))}
              </div>
            </div>
          </SettingCard>

          {/* AI Analysis Settings */}
          <SettingCard icon={Brain} title="AI Analysis & Intelligence" gradient>
            <div className="space-y-2">
              <SettingItem
                title="Real-time Analysis"
                description="Enable continuous AI monitoring and focus tracking"
                isOn={realtimeAnalysis}
                onToggle={() => setRealtimeAnalysis(!realtimeAnalysis)}
                icon={Zap}
                badge="Core"
              />
              <Separator className="my-2 bg-gray-200 dark:bg-gray-700" />
              <SettingItem
                title="Fatigue Detection"
                description="Smart alerts when tiredness patterns are detected"
                isOn={fatigueDetection}
                onToggle={() => setFatigueDetection(!fatigueDetection)}
                icon={Eye}
                badge="Pro"
              />
              <Separator className="my-2 bg-gray-200 dark:bg-gray-700" />
              <SettingItem
                title="Hand Movement Tracking"
                description="Monitor gestures and hand-based distraction signals"
                isOn={handTracking}
                onToggle={() => setHandTracking(!handTracking)}
                icon={Target}
              />
            </div>
          </SettingCard>

          {/* Media & Performance */}
          <SettingCard icon={Camera} title="Camera & Performance" gradient>
            <div className="space-y-2">
              <SettingItem
                title="Auto-start Audio"
                description="Automatically enable microphone when starting sessions"
                isOn={autoAudio}
                onToggle={() => setAutoAudio(!autoAudio)}
                icon={autoAudio ? Volume2 : VolumeX}
              />
              <Separator className="my-2 bg-gray-200 dark:bg-gray-700" />
              <SettingItem
                title="High Quality Video"
                description="Enhanced resolution for superior analysis accuracy"
                isOn={highQualityVideo}
                onToggle={() => setHighQualityVideo(!highQualityVideo)}
                icon={TrendingUp}
                badge="HD"
              />
            </div>
          </SettingCard>

          {/* Smart Notifications */}
          <SettingCard icon={Bell} title="Smart Notifications" gradient>
            <div className="space-y-2">
              <SettingItem
                title="Focus Reminders"
                description="Gentle nudges when attention starts to wander"
                isOn={focusReminders}
                onToggle={() => setFocusReminders(!focusReminders)}
                icon={Lightbulb}
              />
              <Separator className="my-2 bg-gray-200 dark:bg-gray-700" />
              <SettingItem
                title="Break Suggestions"
                description="AI-recommended rest periods based on your patterns"
                isOn={breakSuggestions}
                onToggle={() => setBreakSuggestions(!breakSuggestions)}
                icon={Clock}
                badge="Smart"
              />
              <Separator className="my-2 bg-gray-200 dark:bg-gray-700" />
              <SettingItem
                title="Push Notifications"
                description="System notifications for important updates"
                isOn={notifications}
                onToggle={() => setNotifications(!notifications)}
                icon={Smartphone}
              />
            </div>
          </SettingCard>

          {/* Privacy & Security */}
          <SettingCard icon={Shield} title="Privacy & Security" gradient>
            <div className="space-y-2">
              <SettingItem
                title="Local Processing"
                description="All AI analysis happens securely on your device"
                isOn={localProcessing}
                onToggle={() => setLocalProcessing(!localProcessing)}
                icon={CheckCircle}
                color="bg-gradient-to-r from-green-500 to-green-600"
                disabled
                badge="Secure"
              />
              <Separator className="my-2 bg-gray-200 dark:bg-gray-700" />
              <SettingItem
                title="Save Session Analytics"
                description="Store anonymized data for progress insights"
                isOn={saveSessionData}
                onToggle={() => setSaveSessionData(!saveSessionData)}
                icon={Database}
              />
            </div>
          </SettingCard>

          {/* Data Management */}
          <SettingCard icon={Database} title="Data Management" gradient>
            <div className="grid md:grid-cols-2 gap-4">
              <Button variant="outline" className="h-auto p-4 flex flex-col items-center gap-3 group hover:scale-105 transition-all duration-300 hover:border-[#f4895c]/50">
                <div className="p-3 rounded-xl bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 group-hover:bg-blue-200 dark:group-hover:bg-blue-800/50 transition-colors duration-300">
                  <Download className="h-5 w-5" />
                </div>
                <div className="text-center">
                  <div className="font-medium text-sm">Export Data</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Download your session history</div>
                </div>
              </Button>
              <Button variant="outline" className="h-auto p-4 flex flex-col items-center gap-3 group hover:scale-105 transition-all duration-300 hover:border-[#f4895c]/50">
                <div className="p-3 rounded-xl bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 group-hover:bg-red-200 dark:group-hover:bg-red-800/50 transition-colors duration-300">
                  <RotateCcw className="h-5 w-5" />
                </div>
                <div className="text-center">
                  <div className="font-medium text-sm">Reset Settings</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Restore default configuration</div>
                </div>
              </Button>
            </div>
          </SettingCard>

          {/* System Information */}
          <SettingCard icon={Info} title="System Information" gradient>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                {[
                  { label: 'App Version', value: '2.1.0', icon: Smartphone },
                  { label: 'AI Model', value: 'EmotionNet v3.2', icon: Brain },
                  { label: 'Framework', value: 'Next.js 15', icon: Zap }
                ].map(({ label, value, icon: Icon }) => (
                  <div key={label} className="flex items-center justify-between p-3 rounded-xl bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors duration-300 group">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-white dark:bg-gray-700 shadow-sm group-hover:shadow-md transition-shadow duration-300">
                        <Icon className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                      </div>
                      <span className="text-sm text-gray-600 dark:text-gray-400">{label}</span>
                    </div>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">{value}</span>
                  </div>
                ))}
              </div>
              <div className="space-y-4">
                <Button variant="outline" className="w-full justify-start gap-3 h-auto p-4 group hover:scale-105 transition-all duration-300 hover:border-[#f4895c]/50">
                  <div className="p-2 rounded-lg bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400">
                    <HelpCircle className="h-4 w-4" />
                  </div>
                  <div className="text-left">
                    <div className="text-sm font-medium">Help & Support</div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">Get assistance and tutorials</div>
                  </div>
                  <ExternalLink className="h-4 w-4 ml-auto opacity-50 group-hover:opacity-100 transition-opacity duration-300" />
                </Button>
                <Button variant="outline" className="w-full justify-start gap-3 h-auto p-4 group hover:scale-105 transition-all duration-300 hover:border-[#f4895c]/50">
                  <div className="p-2 rounded-lg bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400">
                    <Heart className="h-4 w-4" />
                  </div>
                  <div className="text-left">
                    <div className="text-sm font-medium">Feedback</div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">Share your experience with us</div>
                  </div>
                  <ExternalLink className="h-4 w-4 ml-auto opacity-50 group-hover:opacity-100 transition-opacity duration-300" />
                </Button>
              </div>
            </div>
          </SettingCard>
        </div>

        {/* Footer */}
        <div className="text-center py-8">
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Study Companion • Powered by AI • Built with ❤️
          </p>
        </div>
      </div>
    </div>
  )
}
