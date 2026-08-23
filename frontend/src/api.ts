/**
 * API client for KneeAI backend.
 */

const BASE_URL = '/api';

export interface CaseMetadata {
  'CMT-ID': number;
  SubjectID: number;
  KLGrade: number;
  Gender_Label: string;
  Age: number;
  BMI: number;
  side: string;
  split: string;
}

export interface MeasurementResult {
  femur?: BoneMeasurement;
  tibia?: BoneMeasurement;
  femoral_cartilage?: CartilageMeasurement;
  medial_tibial_cartilage?: CartilageMeasurement;
  lateral_tibial_cartilage?: CartilageMeasurement;
  derived?: DerivedMeasurements;
}

export interface BoneMeasurement {
  extent_mm: { sagittal: number; coronal: number; axial: number };
  volume_mm3: number;
  volume_cm3: number;
  center_of_mass_mm: number[];
  condyle_ml_width_mm?: number;
  condyle_ap_depth_mm?: number;
  plateau_ml_width_mm?: number;
  plateau_ap_depth_mm?: number;
}

export interface CartilageMeasurement {
  thickness_mean_mm: number;
  thickness_max_mm: number;
  thickness_min_mm: number;
  thickness_std_mm: number;
  volume_mm3: number;
  volume_cm3: number;
  coverage_area_mm2: number;
  extent_mm: { sagittal: number; coronal: number; axial: number };
}

export interface DerivedMeasurements {
  medial_lateral_thickness_ratio?: number;
  estimated_medial_joint_space_mm?: number;
  total_cartilage_volume_mm3: number;
  total_cartilage_volume_cm3: number;
}

export interface ImplantMatch {
  implant: {
    manufacturer: string;
    system: string;
    component: string;
    size: string;
    ml_width_mm: number;
    ap_depth_mm: number;
    source: string;
  };
  score: number;
  ml_mismatch_mm: number;
  ap_mismatch_mm: number;
  ml_coverage_pct: number;
  ap_coverage_pct: number;
  fit_category: string;
  explanation: string;
}

export interface CaseAnalysis {
  case_id: number;
  metadata: CaseMetadata;
  measurements: MeasurementResult;
  implant_matches: Record<string, ImplantMatch[]>;
  oa_assessment: {
    estimated_kl_grade: number;
    description: string;
    confidence: string;
    indicators: string[];
    disclaimer: string;
  };
  reference_comparison: Record<string, unknown>;
  mesh_available: boolean;
}

export interface DatasetSummary {
  dataset_info: Record<string, unknown>;
  summary_stats: {
    total_cases: number;
    train_cases: number;
    test_cases: number;
    kl_grade_distribution: Record<string, number>;
    gender_distribution: Record<string, number>;
    age_stats: { mean: number; std: number; min: number; max: number };
    bmi_stats: { mean: number; std: number; min: number; max: number };
    corrupted_cases: number[];
  };
}

export interface CaseListItem {
  'CMT-ID': number;
  SubjectID: number;
  KLGrade: number;
  Gender_Label: string;
  Age: number;
  BMI: number;
  split: string;
  side: string;
}

export async function fetchDatasetSummary(): Promise<DatasetSummary> {
  const res = await fetch(`${BASE_URL}/dataset/summary`);
  if (!res.ok) throw new Error('Failed to fetch dataset summary');
  return res.json();
}

export async function fetchCases(params?: {
  split?: string;
  kl_grade?: number;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; cases: CaseListItem[] }> {
  const searchParams = new URLSearchParams();
  if (params?.split) searchParams.set('split', params.split);
  if (params?.kl_grade !== undefined) searchParams.set('kl_grade', String(params.kl_grade));
  if (params?.limit) searchParams.set('limit', String(params.limit));
  if (params?.offset) searchParams.set('offset', String(params.offset));

  const res = await fetch(`${BASE_URL}/dataset/cases?${searchParams}`);
  if (!res.ok) throw new Error('Failed to fetch cases');
  return res.json();
}

export async function analyzeCase(caseId: number): Promise<CaseAnalysis> {
  const res = await fetch(`${BASE_URL}/case/${caseId}/analyze`);
  if (!res.ok) throw new Error(`Failed to analyze case ${caseId}`);
  return res.json();
}

export function getMeshUrl(caseId: number): string {
  return `${BASE_URL}/case/${caseId}/mesh`;
}

export async function fetchImplantDatabase(): Promise<Record<string, unknown>> {
  const res = await fetch(`${BASE_URL}/implants/database`);
  if (!res.ok) throw new Error('Failed to fetch implant database');
  return res.json();
}

export async function fetchOACorrelations(): Promise<Record<string, unknown>> {
  const res = await fetch(`${BASE_URL}/research/correlations`);
  if (!res.ok) throw new Error('Failed to fetch OA correlations');
  return res.json();
}
