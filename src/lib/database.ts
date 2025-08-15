import Database from 'duckdb';

interface AIAnalysisData {
  sessionId: string;
  timestamp: string;
  focusScore: number;
  attentionStatus: string;
  fatigueStatus: string;
  gazeDirection: string;
  earValue: number;
  headPose: {
    pitch: number;
    yaw: number;
    roll: number;
  };
  handAnalysis: {
    hand_fatigue_detected: boolean;
    hand_at_head: boolean;
    playing_with_hair: boolean;
    hand_movement: string;
  };
  methodUsed: string;
  facesDetected: number;
}

interface StudySession {
  sessionId: string;
  startTime: string;
  endTime?: string;
  totalDuration: number;
  averageFocusScore: number;
  maxFocusScore: number;
  minFocusScore: number;
  attentionSummary: string;
  fatigueSummary: string;
  totalAnalyses: number;
  completed: boolean;
}

interface SessionSummary {
  date: string;
  totalSessions: number;
  totalStudyTime: number;
  averageFocusScore: number;
  bestSession: string;
  totalXP: number;
}

class StudyDatabase {
  private db: Database.Database;
  private isInitialized: boolean = false;

  constructor() {
    this.db = new Database.Database(':memory:');
    this.initializeDatabase();
  }

  private async initializeDatabase(): Promise<void> {
    return new Promise((resolve, reject) => {
      // Create study_sessions table
      this.db.run(`
        CREATE TABLE IF NOT EXISTS study_sessions (
          session_id VARCHAR PRIMARY KEY,
          start_time TIMESTAMP,
          end_time TIMESTAMP,
          total_duration INTEGER DEFAULT 0,
          average_focus_score DECIMAL(5,2) DEFAULT 0.0,
          max_focus_score DECIMAL(5,2) DEFAULT 0.0,
          min_focus_score DECIMAL(5,2) DEFAULT 100.0,
          attention_summary VARCHAR DEFAULT 'No Data',
          fatigue_summary VARCHAR DEFAULT 'awake',
          total_analyses INTEGER DEFAULT 0,
          completed BOOLEAN DEFAULT FALSE,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
      `, (err) => {
        if (err) {
          console.error('Error creating study_sessions table:', err);
          reject(err);
          return;
        }

        // Create ai_analyses table
        this.db.run(`
          CREATE TABLE IF NOT EXISTS ai_analyses (
            id INTEGER PRIMARY KEY,
            session_id VARCHAR,
            timestamp TIMESTAMP,
            focus_score DECIMAL(5,2),
            attention_status VARCHAR,
            fatigue_status VARCHAR,
            gaze_direction VARCHAR,
            ear_value DECIMAL(5,2),
            head_pose_pitch DECIMAL(8,3),
            head_pose_yaw DECIMAL(8,3),
            head_pose_roll DECIMAL(8,3),
            hand_fatigue_detected BOOLEAN,
            hand_at_head BOOLEAN,
            playing_with_hair BOOLEAN,
            hand_movement VARCHAR,
            method_used VARCHAR,
            faces_detected INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES study_sessions(session_id)
          )
        `, (err) => {
          if (err) {
            console.error('Error creating ai_analyses table:', err);
            reject(err);
            return;
          }

          // Create daily_summaries table
          this.db.run(`
            CREATE TABLE IF NOT EXISTS daily_summaries (
              date DATE PRIMARY KEY,
              total_sessions INTEGER DEFAULT 0,
              total_study_time INTEGER DEFAULT 0,
              average_focus_score DECIMAL(5,2) DEFAULT 0.0,
              best_session VARCHAR,
              total_xp INTEGER DEFAULT 0,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
          `, (err) => {
            if (err) {
              console.error('Error creating daily_summaries table:', err);
              reject(err);
              return;
            }

            // Create indexes for better performance
            this.db.run(`
              CREATE INDEX IF NOT EXISTS idx_analyses_session_timestamp 
              ON ai_analyses(session_id, timestamp)
            `, (err) => {
              if (err) {
                console.error('Error creating index:', err);
                reject(err);
                return;
              }

              console.log('✅ DuckDB database initialized successfully');
              this.isInitialized = true;
              resolve();
            });
          });
        });
      });
    });
  }

  // Start a new study session
  async startSession(sessionId: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const startTime = new Date().toISOString();
      
      this.db.run(
        `INSERT INTO study_sessions (session_id, start_time) VALUES (?, ?)`,
        [sessionId, startTime],
        (err) => {
          if (err) {
            console.error('Error starting session:', err);
            reject(err);
          } else {
            console.log(`📝 Started session: ${sessionId}`);
            resolve();
          }
        }
      );
    });
  }

  // Store AI analysis data
  async storeAnalysis(sessionId: string, data: AIAnalysisData): Promise<void> {
    return new Promise((resolve, reject) => {
      this.db.run(
        `INSERT INTO ai_analyses (
          session_id, timestamp, focus_score, attention_status, fatigue_status,
          gaze_direction, ear_value, head_pose_pitch, head_pose_yaw, head_pose_roll,
          hand_fatigue_detected, hand_at_head, playing_with_hair, hand_movement,
          method_used, faces_detected
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          sessionId,
          data.timestamp,
          data.focusScore,
          data.attentionStatus,
          data.fatigueStatus,
          data.gazeDirection,
          data.earValue,
          data.headPose.pitch,
          data.headPose.yaw,
          data.headPose.roll,
          data.handAnalysis.hand_fatigue_detected,
          data.handAnalysis.hand_at_head,
          data.handAnalysis.playing_with_hair,
          data.handAnalysis.hand_movement,
          data.methodUsed,
          data.facesDetected
        ],
        (err) => {
          if (err) {
            console.error('Error storing analysis:', err);
            reject(err);
          } else {
            console.log(`📊 Stored analysis for session: ${sessionId}`);
            resolve();
          }
        }
      );
    });
  }

  // End a study session and calculate summary
  async endSession(sessionId: string, totalDuration: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const endTime = new Date().toISOString();

      // First, calculate session statistics
      this.db.all(
        `SELECT 
          AVG(focus_score) as avg_focus,
          MAX(focus_score) as max_focus,
          MIN(focus_score) as min_focus,
          COUNT(*) as total_analyses,
          MODE(attention_status) as common_attention,
          MODE(fatigue_status) as common_fatigue
        FROM ai_analyses 
        WHERE session_id = ?`,
        [sessionId],
        (err, rows: any[]) => {
          if (err) {
            console.error('Error calculating session stats:', err);
            reject(err);
            return;
          }

          const stats = rows[0] || {
            avg_focus: 0,
            max_focus: 0,
            min_focus: 0,
            total_analyses: 0,
            common_attention: 'No Data',
            common_fatigue: 'awake'
          };

          // Update the session with calculated statistics
          this.db.run(
            `UPDATE study_sessions SET 
              end_time = ?,
              total_duration = ?,
              average_focus_score = ?,
              max_focus_score = ?,
              min_focus_score = ?,
              attention_summary = ?,
              fatigue_summary = ?,
              total_analyses = ?,
              completed = TRUE
            WHERE session_id = ?`,
            [
              endTime,
              totalDuration,
              stats.avg_focus || 0,
              stats.max_focus || 0,
              stats.min_focus || 0,
              stats.common_attention || 'No Data',
              stats.common_fatigue || 'awake',
              stats.total_analyses || 0,
              sessionId
            ],
            (err) => {
              if (err) {
                console.error('Error ending session:', err);
                reject(err);
              } else {
                console.log(`✅ Ended session: ${sessionId}`);
                this.updateDailySummary();
                resolve();
              }
            }
          );
        }
      );
    });
  }

  // Update daily summary
  private async updateDailySummary(): Promise<void> {
    return new Promise((resolve, reject) => {
      const today = new Date().toISOString().split('T')[0];

      this.db.all(
        `SELECT 
          COUNT(*) as total_sessions,
          SUM(total_duration) as total_time,
          AVG(average_focus_score) as avg_focus,
          MAX(average_focus_score) as best_focus,
          session_id as best_session
        FROM study_sessions 
        WHERE DATE(start_time) = ? AND completed = TRUE`,
        [today],
        (err, rows: any[]) => {
          if (err) {
            console.error('Error calculating daily summary:', err);
            reject(err);
            return;
          }

          const summary = rows[0];
          if (!summary || summary.total_sessions === 0) {
            resolve();
            return;
          }

          // Calculate XP based on sessions and focus
          const xp = (summary.total_sessions * 50) + Math.floor((summary.avg_focus || 0) * 10);

          this.db.run(
            `INSERT OR REPLACE INTO daily_summaries 
            (date, total_sessions, total_study_time, average_focus_score, best_session, total_xp, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)`,
            [
              today,
              summary.total_sessions,
              summary.total_time || 0,
              summary.avg_focus || 0,
              summary.best_session,
              xp,
              new Date().toISOString()
            ],
            (err) => {
              if (err) {
                console.error('Error updating daily summary:', err);
                reject(err);
              } else {
                console.log(`📈 Updated daily summary for ${today}`);
                resolve();
              }
            }
          );
        }
      );
    });
  }

  // Get session analytics
  async getSessionAnalytics(sessionId: string): Promise<StudySession | null> {
    return new Promise((resolve, reject) => {
      this.db.get(
        `SELECT * FROM study_sessions WHERE session_id = ?`,
        [sessionId],
        (err, row: any) => {
          if (err) {
            console.error('Error getting session analytics:', err);
            reject(err);
          } else {
            resolve(row as StudySession || null);
          }
        }
      );
    });
  }

  // Get recent sessions
  async getRecentSessions(limit: number = 10): Promise<StudySession[]> {
    return new Promise((resolve, reject) => {
      this.db.all(
        `SELECT * FROM study_sessions 
         WHERE completed = TRUE 
         ORDER BY start_time DESC 
         LIMIT ?`,
        [limit],
        (err, rows: any[]) => {
          if (err) {
            console.error('Error getting recent sessions:', err);
            reject(err);
          } else {
            resolve(rows as StudySession[]);
          }
        }
      );
    });
  }

  // Get daily summaries for the last N days
  async getDailySummaries(days: number = 7): Promise<SessionSummary[]> {
    return new Promise((resolve, reject) => {
      this.db.all(
        `SELECT * FROM daily_summaries 
         ORDER BY date DESC 
         LIMIT ?`,
        [days],
        (err, rows: any[]) => {
          if (err) {
            console.error('Error getting daily summaries:', err);
            reject(err);
          } else {
            resolve(rows as SessionSummary[]);
          }
        }
      );
    });
  }

  // Get focus score trends for a session
  async getFocusTrends(sessionId: string): Promise<Array<{timestamp: string, focusScore: number}>> {
    return new Promise((resolve, reject) => {
      this.db.all(
        `SELECT timestamp, focus_score as focusScore 
         FROM ai_analyses 
         WHERE session_id = ? 
         ORDER BY timestamp ASC`,
        [sessionId],
        (err, rows: any[]) => {
          if (err) {
            console.error('Error getting focus trends:', err);
            reject(err);
          } else {
            resolve(rows);
          }
        }
      );
    });
  }

  // Get overall statistics
  async getOverallStats(): Promise<{
    totalSessions: number;
    totalStudyTime: number;
    averageFocusScore: number;
    totalXP: number;
    currentStreak: number;
  }> {
    return new Promise((resolve, reject) => {
      this.db.get(
        `SELECT 
          COUNT(*) as totalSessions,
          SUM(total_duration) as totalStudyTime,
          AVG(average_focus_score) as averageFocusScore
        FROM study_sessions 
        WHERE completed = TRUE`,
        [],
        (err, sessionStats: any) => {
          if (err) {
            console.error('Error getting overall stats:', err);
            reject(err);
            return;
          }

          // Get total XP
          this.db.get(
            `SELECT SUM(total_xp) as totalXP FROM daily_summaries`,
            [],
            (err, xpStats: any) => {
              if (err) {
                console.error('Error getting XP stats:', err);
                reject(err);
                return;
              }

              // Calculate current streak (simplified)
              this.db.get(
                `SELECT COUNT(*) as currentStreak 
                 FROM daily_summaries 
                 WHERE date >= date('now', '-7 days') 
                 AND total_sessions > 0`,
                [],
                (err, streakStats: any) => {
                  if (err) {
                    console.error('Error getting streak stats:', err);
                    reject(err);
                    return;
                  }

                  resolve({
                    totalSessions: sessionStats?.totalSessions || 0,
                    totalStudyTime: sessionStats?.totalStudyTime || 0,
                    averageFocusScore: sessionStats?.averageFocusScore || 0,
                    totalXP: xpStats?.totalXP || 0,
                    currentStreak: streakStats?.currentStreak || 0
                  });
                }
              );
            }
          );
        }
      );
    });
  }

  // Close database connection
  async close(): Promise<void> {
    return new Promise((resolve) => {
      this.db.close((err) => {
        if (err) {
          console.error('Error closing database:', err);
        } else {
          console.log('📫 Database connection closed');
        }
        resolve();
      });
    });
  }
}

// Export singleton instance
export const studyDB = new StudyDatabase();
export type { AIAnalysisData, StudySession, SessionSummary };