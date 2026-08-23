/**
 * Patient profile card showing demographics and clinical data.
 */
import { CaseMetadata } from '../api'

interface PatientProfileProps {
  metadata: CaseMetadata | null;
}

function KLGradeBadge({ grade }: { grade: number }) {
  const colors: Record<number, string> = {
    0: 'bg-green-900/50 text-green-300 border-green-700',
    1: 'bg-lime-900/50 text-lime-300 border-lime-700',
    2: 'bg-yellow-900/50 text-yellow-300 border-yellow-700',
    3: 'bg-orange-900/50 text-orange-300 border-orange-700',
    4: 'bg-red-900/50 text-red-300 border-red-700',
  }

  const descriptions: Record<number, string> = {
    0: 'No OA',
    1: 'Doubtful',
    2: 'Mild',
    3: 'Moderate',
    4: 'Severe',
  }

  return (
    <span className={`badge border ${colors[grade] || colors[0]}`}>
      KL {grade} - {descriptions[grade] || 'Unknown'}
    </span>
  )
}

export default function PatientProfile({ metadata }: PatientProfileProps) {
  if (!metadata) {
    return (
      <div className="card">
        <div className="card-header">Patient Profile</div>
        <p className="text-gray-500 text-sm">No case selected</p>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="card-header mb-0">Patient Profile</h3>
        <KLGradeBadge grade={metadata.KLGrade} />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="stat-label">Case ID</div>
          <div className="stat-value text-lg">{metadata['CMT-ID']}</div>
        </div>
        <div>
          <div className="stat-label">Subject</div>
          <div className="stat-value text-lg">{metadata.SubjectID}</div>
        </div>
        <div>
          <div className="stat-label">Age</div>
          <div className="stat-value text-lg">{metadata.Age} yrs</div>
        </div>
        <div>
          <div className="stat-label">Gender</div>
          <div className="stat-value text-lg">{metadata.Gender_Label}</div>
        </div>
        <div>
          <div className="stat-label">BMI</div>
          <div className="stat-value text-lg">{metadata.BMI}</div>
        </div>
        <div>
          <div className="stat-label">Knee Side</div>
          <div className="stat-value text-lg capitalize">{metadata.side}</div>
        </div>
      </div>
    </div>
  )
}
