/**
 * Displays anatomical measurements from the measurement engine.
 */
import { MeasurementResult } from '../api'

interface MeasurementsPanelProps {
  measurements: MeasurementResult | null;
}

function MeasurementRow({ label, value, unit }: { label: string; value: number | undefined; unit: string }) {
  if (value === undefined || value === null) return null
  return (
    <div className="flex justify-between items-center py-1.5 border-b border-gray-800/50 last:border-0">
      <span className="text-sm text-gray-400">{label}</span>
      <span className="text-sm font-medium text-white">{value.toFixed(2)} {unit}</span>
    </div>
  )
}

function StructureSection({ title, children, color }: { title: string; children: React.ReactNode; color: string }) {
  return (
    <div className="mb-4 last:mb-0">
      <div className="flex items-center gap-2 mb-2">
        <div className={`w-3 h-3 rounded-full ${color}`} />
        <h4 className="text-sm font-semibold text-gray-300">{title}</h4>
      </div>
      <div className="pl-5">{children}</div>
    </div>
  )
}

export default function MeasurementsPanel({ measurements }: MeasurementsPanelProps) {
  if (!measurements) {
    return (
      <div className="card">
        <div className="card-header">Measurements</div>
        <p className="text-gray-500 text-sm">Run analysis to see measurements</p>
      </div>
    )
  }

  const { femur, tibia, femoral_cartilage, medial_tibial_cartilage, lateral_tibial_cartilage, derived } = measurements

  return (
    <div className="card overflow-y-auto max-h-[600px]">
      <div className="card-header">Anatomical Measurements</div>

      {femur && (
        <StructureSection title="Femur" color="bg-amber-200">
          <MeasurementRow label="Volume" value={femur.volume_cm3} unit="cm³" />
          <MeasurementRow label="Condyle ML Width" value={femur.condyle_ml_width_mm} unit="mm" />
          <MeasurementRow label="Condyle AP Depth" value={femur.condyle_ap_depth_mm} unit="mm" />
        </StructureSection>
      )}

      {tibia && (
        <StructureSection title="Tibia" color="bg-amber-100">
          <MeasurementRow label="Volume" value={tibia.volume_cm3} unit="cm³" />
          <MeasurementRow label="Plateau ML Width" value={tibia.plateau_ml_width_mm} unit="mm" />
          <MeasurementRow label="Plateau AP Depth" value={tibia.plateau_ap_depth_mm} unit="mm" />
        </StructureSection>
      )}

      {femoral_cartilage && (
        <StructureSection title="Femoral Cartilage" color="bg-blue-400">
          <MeasurementRow label="Mean Thickness" value={femoral_cartilage.thickness_mean_mm} unit="mm" />
          <MeasurementRow label="Max Thickness" value={femoral_cartilage.thickness_max_mm} unit="mm" />
          <MeasurementRow label="Volume" value={femoral_cartilage.volume_cm3} unit="cm³" />
          <MeasurementRow label="Coverage Area" value={femoral_cartilage.coverage_area_mm2} unit="mm²" />
        </StructureSection>
      )}

      {medial_tibial_cartilage && (
        <StructureSection title="Medial Tibial Cartilage" color="bg-red-400">
          <MeasurementRow label="Mean Thickness" value={medial_tibial_cartilage.thickness_mean_mm} unit="mm" />
          <MeasurementRow label="Max Thickness" value={medial_tibial_cartilage.thickness_max_mm} unit="mm" />
          <MeasurementRow label="Volume" value={medial_tibial_cartilage.volume_cm3} unit="cm³" />
          <MeasurementRow label="Coverage Area" value={medial_tibial_cartilage.coverage_area_mm2} unit="mm²" />
        </StructureSection>
      )}

      {lateral_tibial_cartilage && (
        <StructureSection title="Lateral Tibial Cartilage" color="bg-green-400">
          <MeasurementRow label="Mean Thickness" value={lateral_tibial_cartilage.thickness_mean_mm} unit="mm" />
          <MeasurementRow label="Max Thickness" value={lateral_tibial_cartilage.thickness_max_mm} unit="mm" />
          <MeasurementRow label="Volume" value={lateral_tibial_cartilage.volume_cm3} unit="cm³" />
          <MeasurementRow label="Coverage Area" value={lateral_tibial_cartilage.coverage_area_mm2} unit="mm²" />
        </StructureSection>
      )}

      {derived && (
        <StructureSection title="Derived Metrics" color="bg-purple-400">
          <MeasurementRow label="M/L Thickness Ratio" value={derived.medial_lateral_thickness_ratio} unit="" />
          <MeasurementRow label="Est. Joint Space" value={derived.estimated_medial_joint_space_mm} unit="mm" />
          <MeasurementRow label="Total Cart. Volume" value={derived.total_cartilage_volume_cm3} unit="cm³" />
        </StructureSection>
      )}
    </div>
  )
}
