/**
 * OA Assessment and Explainability panel.
 */

interface ComparisonItem {
  label?: string;
  patient_value_mm?: number;
  patient_mm?: number;
  z_score_vs_healthy?: number;
  interpretation?: string;
  healthy_mean_mm?: number;
}

interface ReferenceData {
  comparisons?: Record<string, ComparisonItem>;
  source?: string;
  population_size?: number;
}

interface OAAssessmentProps {
  assessment: {
    estimated_kl_grade: number;
    description: string;
    confidence: string;
    indicators: string[];
    disclaimer: string;
  } | null;
  referenceComparison: Record<string, unknown> | null;
}

function KLGauge({ grade }: { grade: number }) {
  const segments = [0, 1, 2, 3, 4]
  const colors = ['bg-green-500', 'bg-lime-500', 'bg-yellow-500', 'bg-orange-500', 'bg-red-500']

  return (
    <div className="flex items-center gap-1 mb-4">
      {segments.map((seg) => (
        <div
          key={seg}
          className={`h-3 flex-1 rounded-sm ${seg <= grade ? colors[seg] : 'bg-gray-800'}`}
        />
      ))}
    </div>
  )
}

export default function OAAssessment({ assessment, referenceComparison }: OAAssessmentProps) {
  if (!assessment) {
    return (
      <div className="card">
        <div className="card-header">OA Assessment</div>
        <p className="text-gray-500 text-sm">Run analysis to see OA assessment</p>
      </div>
    )
  }

  const ref = referenceComparison as ReferenceData | null
  const comparisons = ref?.comparisons

  return (
    <div className="card">
      <div className="card-header">OA Assessment &amp; Explainability</div>

      {/* KL Grade Estimate */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-1">
          <span className="text-sm text-gray-400">Estimated KL Grade</span>
          <span className="text-2xl font-bold text-white">{assessment.estimated_kl_grade}</span>
        </div>
        <KLGauge grade={assessment.estimated_kl_grade} />
        <p className="text-sm text-gray-300">{assessment.description}</p>
      </div>

      {/* Indicators */}
      {assessment.indicators.length > 0 && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-gray-400 mb-2">Clinical Indicators</h4>
          <ul className="space-y-1">
            {assessment.indicators.map((indicator, i) => (
              <li key={i} className="text-sm text-gray-300 flex items-start gap-2">
                <span className="text-yellow-500 mt-0.5">{'\u2022'}</span>
                <span>{indicator}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Reference Comparison */}
      {comparisons && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-gray-400 mb-2">Population Comparison</h4>
          {Object.entries(comparisons).map(([key, comp]) => (
            <div key={key} className="p-3 bg-gray-800/50 rounded-lg mb-2 last:mb-0">
              <div className="text-xs text-gray-400 mb-1">{comp.label ?? key}</div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-white">
                  {comp.patient_value_mm ?? comp.patient_mm ?? 'N/A'} mm
                </span>
                {comp.z_score_vs_healthy !== undefined && (
                  <span className={`text-xs ${comp.z_score_vs_healthy < -1 ? 'text-red-400' : 'text-green-400'}`}>
                    z = {comp.z_score_vs_healthy.toFixed(2)}
                  </span>
                )}
              </div>
              {comp.interpretation && (
                <p className="text-xs text-gray-500 mt-1">{comp.interpretation}</p>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Disclaimer */}
      <div className="p-3 bg-yellow-900/20 border border-yellow-800/50 rounded-lg">
        <p className="text-xs text-yellow-200/80">{assessment.disclaimer}</p>
      </div>
    </div>
  )
}
