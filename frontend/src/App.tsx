/**
 * KneeAI - Main Application Component.
 * AI-powered knee OA analysis and implant planning platform.
 */
import { useState, useCallback } from 'react'
import KneeViewer3D from './components/KneeViewer3D'
import PatientProfile from './components/PatientProfile'
import MeasurementsPanel from './components/MeasurementsPanel'
import ImplantMatching from './components/ImplantMatching'
import OAAssessment from './components/OAAssessment'
import CaseSelector from './components/CaseSelector'
import { analyzeCase, getMeshUrl, CaseAnalysis } from './api'

type Tab = 'viewer' | 'measurements' | 'implants' | 'research'

export default function App() {
  const [selectedCaseId, setSelectedCaseId] = useState<number | null>(null)
  const [analysis, setAnalysis] = useState<CaseAnalysis | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<Tab>('viewer')
  const [meshUrl, setMeshUrl] = useState<string | null>(null)

  const handleSelectCase = useCallback(async (caseId: number) => {
    setSelectedCaseId(caseId)
    setIsLoading(true)
    setError(null)
    setMeshUrl(null)

    try {
      const result = await analyzeCase(caseId)
      setAnalysis(result)
      // Set mesh URL for 3D viewer
      setMeshUrl(getMeshUrl(caseId))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
      setAnalysis(null)
    } finally {
      setIsLoading(false)
    }
  }, [])

  const tabs: { id: Tab; label: string }[] = [
    { id: 'viewer', label: '3D Viewer' },
    { id: 'measurements', label: 'Measurements' },
    { id: 'implants', label: 'Implant Match' },
    { id: 'research', label: 'OA Research' },
  ]

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-950 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center">
              <span className="text-lg font-bold text-white">K</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">KneeAI</h1>
              <p className="text-xs text-gray-500">Knee OA Analysis & Implant Planning</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {isLoading && (
              <div className="flex items-center gap-2 text-blue-400 text-sm">
                <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                Analyzing...
              </div>
            )}
            {error && (
              <div className="text-red-400 text-sm">{error}</div>
            )}
          </div>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left sidebar - Case selection */}
        <aside className="w-80 border-r border-gray-800 p-4 overflow-y-auto">
          <CaseSelector
            selectedCaseId={selectedCaseId}
            onSelectCase={handleSelectCase}
            isLoading={isLoading}
          />
          <div className="mt-4">
            <PatientProfile metadata={analysis?.metadata || null} />
          </div>
        </aside>

        {/* Center - Main content area */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Tab bar */}
          <div className="border-b border-gray-800 px-6">
            <nav className="flex gap-1">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-400'
                      : 'border-transparent text-gray-400 hover:text-gray-200'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-auto p-6">
            {activeTab === 'viewer' && (
              <div className="h-full">
                <KneeViewer3D meshUrl={meshUrl} isLoading={isLoading} />
              </div>
            )}

            {activeTab === 'measurements' && (
              <div className="max-w-4xl">
                <MeasurementsPanel measurements={analysis?.measurements || null} />
              </div>
            )}

            {activeTab === 'implants' && (
              <div className="max-w-4xl">
                <ImplantMatching matches={analysis?.implant_matches || null} />
              </div>
            )}

            {activeTab === 'research' && (
              <div className="max-w-4xl">
                <OAAssessment
                  assessment={analysis?.oa_assessment || null}
                  referenceComparison={analysis?.reference_comparison || null}
                />
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
