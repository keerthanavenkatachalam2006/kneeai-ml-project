/**
 * Implant matching results panel with rankings and explanations.
 */
import { ImplantMatch } from '../api'

interface ImplantMatchingProps {
  matches: Record<string, ImplantMatch[]> | null;
}

function FitBadge({ category }: { category: string }) {
  const styles: Record<string, string> = {
    optimal: 'badge-optimal',
    acceptable: 'badge-acceptable',
    overhang_risk: 'badge-overhang',
    underhang_risk: 'badge-underhang',
  }

  const labels: Record<string, string> = {
    optimal: 'Optimal Fit',
    acceptable: 'Acceptable',
    overhang_risk: 'Overhang Risk',
    underhang_risk: 'Underhang Risk',
  }

  return (
    <span className={`badge ${styles[category] || 'badge-acceptable'}`}>
      {labels[category] || category}
    </span>
  )
}

function ScoreBar({ score }: { score: number }) {
  const color = score >= 80 ? 'bg-green-500' : score >= 60 ? 'bg-yellow-500' : score >= 40 ? 'bg-orange-500' : 'bg-red-500'

  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${score}%` }} />
      </div>
      <span className="text-xs font-mono text-gray-400 w-10 text-right">{score}%</span>
    </div>
  )
}

function ImplantCard({ match, rank }: { match: ImplantMatch; rank: number }) {
  return (
    <div className={`p-4 rounded-lg border ${rank === 1 ? 'border-green-700 bg-green-950/20' : 'border-gray-800 bg-gray-900/50'}`}>
      <div className="flex items-start justify-between mb-2">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-gray-500">#{rank}</span>
            <span className="font-semibold text-white text-sm">
              {match.implant.manufacturer} {match.implant.system}
            </span>
          </div>
          <span className="text-xs text-gray-400">Size {match.implant.size}</span>
        </div>
        <FitBadge category={match.fit_category} />
      </div>

      <ScoreBar score={match.score} />

      <div className="grid grid-cols-2 gap-2 mt-3 text-xs">
        <div>
          <span className="text-gray-500">ML:</span>{' '}
          <span className={match.ml_mismatch_mm > 0 ? 'text-orange-400' : 'text-blue-400'}>
            {match.ml_mismatch_mm > 0 ? '+' : ''}{match.ml_mismatch_mm}mm
          </span>
        </div>
        <div>
          <span className="text-gray-500">AP:</span>{' '}
          <span className={match.ap_mismatch_mm > 0 ? 'text-orange-400' : 'text-blue-400'}>
            {match.ap_mismatch_mm > 0 ? '+' : ''}{match.ap_mismatch_mm}mm
          </span>
        </div>
        <div>
          <span className="text-gray-500">ML Coverage:</span>{' '}
          <span className="text-white">{match.ml_coverage_pct}%</span>
        </div>
        <div>
          <span className="text-gray-500">AP Coverage:</span>{' '}
          <span className="text-white">{match.ap_coverage_pct}%</span>
        </div>
      </div>

      <p className="text-xs text-gray-500 mt-3 leading-relaxed">{match.explanation}</p>
    </div>
  )
}

export default function ImplantMatching({ matches }: ImplantMatchingProps) {
  if (!matches) {
    return (
      <div className="card">
        <div className="card-header">Implant Matching</div>
        <p className="text-gray-500 text-sm">Run analysis to see implant recommendations</p>
      </div>
    )
  }

  return (
    <div className="card overflow-y-auto max-h-[700px]">
      <div className="card-header">Implant Matching</div>

      {matches.femoral && matches.femoral.length > 0 && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-blue-400 mb-3">Femoral Component</h4>
          <div className="space-y-3">
            {matches.femoral.map((match, i) => (
              <ImplantCard key={`f-${i}`} match={match} rank={i + 1} />
            ))}
          </div>
        </div>
      )}

      {matches.tibial && matches.tibial.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-green-400 mb-3">Tibial Component</h4>
          <div className="space-y-3">
            {matches.tibial.map((match, i) => (
              <ImplantCard key={`t-${i}`} match={match} rank={i + 1} />
            ))}
          </div>
        </div>
      )}

      {(!matches.femoral || matches.femoral.length === 0) && (!matches.tibial || matches.tibial.length === 0) && (
        <p className="text-gray-500 text-sm">
          Could not compute implant matches. Ensure bone measurements (condyle width, plateau width) are available.
        </p>
      )}
    </div>
  )
}
