/**
 * Case selector panel for browsing and selecting patient cases.
 */
import { useState, useEffect } from 'react'
import { fetchCases, CaseListItem } from '../api'

interface CaseSelectorProps {
  selectedCaseId: number | null;
  onSelectCase: (caseId: number) => void;
  isLoading: boolean;
}

export default function CaseSelector({ selectedCaseId, onSelectCase, isLoading }: CaseSelectorProps) {
  const [cases, setCases] = useState<CaseListItem[]>([])
  const [total, setTotal] = useState(0)
  const [filter, setFilter] = useState({ split: '', kl_grade: '' })
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    loadCases()
  }, [filter])

  async function loadCases() {
    setLoading(true)
    try {
      const params: Record<string, unknown> = { limit: 20 }
      if (filter.split) params.split = filter.split
      if (filter.kl_grade !== '') params.kl_grade = parseInt(filter.kl_grade)

      const data = await fetchCases(params as Parameters<typeof fetchCases>[0])
      setCases(data.cases)
      setTotal(data.total)
    } catch {
      // API not available yet, show placeholder
      setCases([])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <div className="card-header">Case Browser</div>

      {/* Filters */}
      <div className="flex gap-2 mb-4">
        <select
          className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-300 focus:outline-none focus:border-blue-500"
          value={filter.split}
          onChange={(e) => setFilter(f => ({ ...f, split: e.target.value }))}
        >
          <option value="">All Splits</option>
          <option value="train">Training</option>
          <option value="test">Testing</option>
        </select>

        <select
          className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-300 focus:outline-none focus:border-blue-500"
          value={filter.kl_grade}
          onChange={(e) => setFilter(f => ({ ...f, kl_grade: e.target.value }))}
        >
          <option value="">All KL Grades</option>
          <option value="0">KL 0 - No OA</option>
          <option value="1">KL 1 - Doubtful</option>
          <option value="2">KL 2 - Mild</option>
          <option value="3">KL 3 - Moderate</option>
          <option value="4">KL 4 - Severe</option>
        </select>
      </div>

      <div className="text-xs text-gray-500 mb-3">{total} cases found</div>

      {/* Case list */}
      <div className="space-y-1 max-h-[300px] overflow-y-auto">
        {loading ? (
          <div className="text-center py-4 text-gray-500 text-sm">Loading cases...</div>
        ) : cases.length === 0 ? (
          <div className="text-center py-4 text-gray-500 text-sm">
            No cases available. Start the backend server.
          </div>
        ) : (
          cases.map((c) => (
            <button
              key={c['CMT-ID']}
              onClick={() => onSelectCase(c['CMT-ID'])}
              disabled={isLoading}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                selectedCaseId === c['CMT-ID']
                  ? 'bg-blue-900/50 border border-blue-700 text-blue-200'
                  : 'hover:bg-gray-800 text-gray-300'
              } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <div className="flex items-center justify-between">
                <span className="font-mono">Case {c['CMT-ID']}</span>
                <span className="text-xs text-gray-500">
                  KL{c.KLGrade} • {c.Gender_Label} • {c.Age}y
                </span>
              </div>
            </button>
          ))
        )}
      </div>
    </div>
  )
}
