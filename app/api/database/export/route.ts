import { NextRequest, NextResponse } from 'next/server'
import { studyDB } from '@/lib/database'

export async function GET() {
  try {
    const data = await studyDB.getAllDataAsJSON()
    
    return NextResponse.json({
      success: true,
      data,
      timestamp: new Date().toISOString()
    })
  } catch (error) {
    console.error('Error exporting database:', error)
    return NextResponse.json(
      { error: 'Failed to export database' },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const { format } = await request.json()
    
    if (format === 'json') {
      const data = await studyDB.getAllDataAsJSON()
      
      // Return as downloadable file
      const jsonString = JSON.stringify(data, null, 2)
      
      return new NextResponse(jsonString, {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Content-Disposition': `attachment; filename="study_data_${Date.now()}.json"`
        }
      })
    }
    
    return NextResponse.json({ error: 'Unsupported format' }, { status: 400 })
  } catch (error) {
    console.error('Error exporting database:', error)
    return NextResponse.json(
      { error: 'Failed to export database' },
      { status: 500 }
    )
  }
}
