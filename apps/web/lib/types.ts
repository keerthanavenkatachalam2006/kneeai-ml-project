/**
 * Types mirroring the platform's ML -> application contract.
 *
 * These follow `docs/ML_CONTRACT.md` and `kneeai.ml.contract`. Two rules from that
 * contract are encoded structurally here rather than left to reviewers:
 *
 * 1. `CaseResult` has no accuracy field. Dice, IoU, HD95 and ASSD require ground
 *    truth, which does not exist at inference time on a patient scan. Reference
 *    metrics live on the model card and are typed separately as
 *    `ModelInfo.reference_metrics`, whose `scope` says out loud that they describe
 *    the validation set and not the case on screen.
 *
 * 2. `source` is required, not optional, so a component cannot render geometry
 *    without knowing whether a model produced it.
 */

export type SegmentationSource = "ai_prediction" | "ground_truth" | "phantom" | "mock";

export interface CaseListEntry {
  case_id: string;
  source: SegmentationSource | string;
  badge: string;
  is_ai: boolean;
  side: string | null;
  spacing_mm: number[] | null;
  image_shape: number[] | null;
  n_structures: number;
  structures_missing: string[];
  model: string | null;
  has_plan: boolean;
  has_meshes: boolean;
  has_simulation: boolean;
  n_warnings: number;
  modified: number;
}

export interface CaseResult {
  case_id: string;
  source: SegmentationSource | string;
  model: {
    model_id: string | null;
    version: string | null;
    architecture: string | null;
    model_card: string | null;
  };
  segmentation: {
    path: string;
    label_map: Record<string, number>;
    structures_present: string[];
    structures_expected_but_missing: string[];
  };
  geometry: {
    spacing_mm: number[];
    orientation: string;
    affine: number[][];
    image_shape: number[];
    side: string;
  };
  confidence: {
    per_structure: Record<string, number | null>;
    method: string;
    volume_mm3: Record<string, number>;
  };
  provenance?: unknown;
  warnings: string[];
}

export interface Quantity {
  value: number | null;
  unit: string | null;
  uncertainty?: number | null;
}

export interface Measurement {
  name: string;
  display_name: string;
  quantity: Quantity;
  formatted: string;
  method: string;
  structures: string[];
  landmarks: string[];
  origin: string;
  sign_convention?: string;
  /**
   * True inside the reference range, false outside it, and null when no range is
   * published for the measurement. The three-way distinction matters: "no reference
   * range exists" is not the same statement as "within range".
   */
  in_reference_range?: boolean | null;
  /** Both bounds nullable: several published ranges are one-sided. */
  reference_range?: { low: number | null; high: number | null } | null;
  /** A single sentence, empty when there is nothing to qualify. */
  caveat?: string;
}

export interface MeasurementSet {
  side: string;
  n_measurements: number;
  measurements: Measurement[];
  /** name -> reason it could not be computed. Never silently dropped. */
  unavailable: Record<string, string>;
  provenance?: unknown;
}

export interface MeshEntry {
  structure: string;
  category: string;
  path: string;
  frame: string;
  units: string;
  n_vertices: number;
  n_faces: number;
  volume_mm3: number | null;
  is_watertight: boolean;
  bounds_mm: [number[], number[]];
  compression: string;
  bytes_size: number;
  content_hash: string;
}

export interface MeshManifest {
  case_id: string;
  units: string;
  compression: string;
  total_faces: number;
  meshes: MeshEntry[];
}

/** One scoring term inside a fit, with the sentence that explains it. */
export interface FitTerm {
  name: string;
  metric?: number | null;
  unit?: string;
  score?: number | null;
  weight?: number | null;
  explanation?: string;
}

/**
 * How one component fits one bone.
 *
 * Femoral components are scored dimensionally (a five-cut component does not seat on
 * a single plane) so the polygon fields — coverage, overhang, undercoverage — are null
 * for them and the mismatch fields carry the information instead. Tibial components
 * are scored on the resected footprint, so it is the other way round.
 */
export interface ComponentFit {
  component_id?: string;
  size_label?: string;
  component_type?: string;
  method?: string;
  score?: number | null;
  acceptable?: boolean;
  violations?: string[];
  terms?: FitTerm[];
  coverage_pct?: number | null;
  overhang_max_mm?: number | null;
  overhang_mean_mm?: number | null;
  undercoverage_max_mm?: number | null;
  ml_mismatch_mm?: number | null;
  ap_mismatch_mm?: number | null;
  component_ml_mm?: number | null;
  component_ap_mm?: number | null;
  translation_mm?: number[];
  rotation_deg?: number | null;
}

export interface ImplantConfigurationRef {
  configuration_id?: string;
  label?: string;
  femoral?: string;
  tibial_baseplate?: string;
  tibial_insert?: string | null;
  bearing?: string;
  fixation?: string;
  total_construct_thickness_mm?: number;
}

export interface ImplantCandidate {
  configuration?: ImplantConfigurationRef;
  score?: number;
  acceptable?: boolean;
  femoral_fit?: ComponentFit;
  tibial_fit?: ComponentFit;
  violations?: string[];
  notes?: string[];
  [key: string]: unknown;
}

export interface ImplantResult {
  selected: ImplantCandidate | null;
  rationale: string | string[] | null;
  candidates: ImplantCandidate[];
  runner_up: ImplantCandidate | null;
  disclaimer: string;
  method: string;
}

export interface CutPlane {
  point: number[];
  normal: number[];
  frame: string;
  label: string;
}

export interface ResectionCut {
  name: string;
  structure: string;
  plane: CutPlane;
  depth_mm: number | null;
  reference: string;
  angles_deg: Record<string, number | null>;
  resected_volume_mm3: number | null;
  notes: string;
}

export interface SurgicalPlan {
  plan_id: string;
  case_id: string;
  side: string;
  version: number;
  label: string;
  strategy: string;
  targets: Record<string, unknown>;
  cuts: ResectionCut[];
  placements: Array<{
    component_id: string;
    structure: string;
    size_label: string;
    transform: { source: string; target: string; matrix: number[][]; label: string };
  }>;
  configuration_id: string;
  configuration_label: string;
  femoral_size: string;
  tibial_size: string;
  insert_thickness_mm: number | null;
}

export interface SimulationFinding {
  code?: string;
  severity?: string;
  message?: string;
  [key: string]: unknown;
}

export interface SimulationResult {
  plan_id: string;
  side: string;
  passed: boolean;
  n_failures: number;
  n_warnings: number;
  findings: SimulationFinding[];
  cuts?: unknown;
  resected_volume_mm3: number | null;
  interpenetration_mm: Record<string, number> | number | null;
  seating_gap_mm: Record<string, number> | number | null;
}

export interface Landmark {
  name: string;
  display_name?: string;
  position: number[];
  confidence: number;
  uncertainty_mm?: number | null;
  origin?: string;
}

export interface LandmarkSet {
  side: string;
  frame: string;
  mean_confidence: number;
  n_landmarks: number;
  n_surgeon_edited: number;
  landmarks: Landmark[];
}

export interface CaseSummary {
  case_id: string;
  source: SegmentationSource | string;
  badge: string;
  is_ai: boolean;
  result: CaseResult;
  landmarks: LandmarkSet | null;
  frame: Record<string, unknown> | null;
  measurements: MeasurementSet | null;
  implants: ImplantResult | null;
  plan: SurgicalPlan | null;
  simulation: SimulationResult | null;
  mesh_quality: unknown | null;
  meshes: MeshManifest | null;
  contract_valid: boolean;
  contract_errors: string[];
  simulation_state: string;
  banner: string;
  disclaimer: string;
  /**
   * Kellgren-Lawrence grade from the dataset's own radiographic reading, when the
   * cohort supplies one. A label, never a model output — nothing in this platform
   * predicts it.
   */
  kl_grade?: number | null;
  cohort?: Record<string, unknown> | null;
}

export interface ModelInfo {
  available: boolean;
  reason?: string;
  model_id?: string;
  version?: string;
  architecture?: string;
  task?: string;
  label_map?: Record<string, number>;
  approved_for_clinical_use?: boolean;
  reference_metrics?: {
    /** States that these describe the validation set, not the current patient. */
    scope: string;
    n_patients: number | null;
    mean_dice_excl_background: number | null;
    dice_per_structure: Record<string, number> | null;
    metric_definition: string | null;
  };
  dataset?: Record<string, unknown>;
  limitations?: string;
  contraindications?: string[];
}

export type SimulationStateName =
  | "planning"
  | "awaiting_surgeon_approval"
  | "approved_for_simulation"
  | "safety_check"
  | "simulation_running"
  | "simulation_complete"
  | "emergency_stopped"
  | "rejected";

export interface SafetyCheck {
  name: string;
  passed: boolean;
  detail: string;
  blocking: boolean;
}

export interface SimulationStateInfo {
  case_id: string;
  plan_id?: string;
  state: SimulationStateName | string;
  is_terminal?: boolean;
  allows_motion?: boolean;
  approver: string | null;
  approved_at?: string | null;
  safety_report: {
    passed: boolean;
    n_checks: number;
    n_blocking_failures: number;
    checks: SafetyCheck[];
  } | null;
  history?: Array<{ from: string; to: string; at: string; actor: string; reason: string }>;
  banner: string;
  hardware_control: boolean;
}

export interface ToolPose {
  position_mm: number[];
  axis: number[];
  plane_normal?: number[];
  frame: string;
  units: string;
  label?: string;
}

export interface RobotCommandEntry {
  sequence: number;
  action: string;
  target: string;
  status: string;
  pose?: ToolPose;
  parameters?: Record<string, unknown>;
  rationale?: string;
}

export interface RobotCommandSequence {
  case_id: string;
  plan_id: string;
  n_commands: number;
  frame: string;
  units: string;
  executable: false;
  hardware_control: false;
  note: string;
  safety_boundary: {
    type: string;
    min_mm: number[];
    max_mm: number[];
    size_mm: number[];
    margin_mm: number;
    derived_from: string[];
    frame: string;
    note: string;
  } | null;
  commands: RobotCommandEntry[];
  warnings: string[];
}

export interface SliceInfo {
  shape: number[];
  spacing_mm: number[];
  n_slices: Record<string, number>;
  default_index: Record<string, number>;
  planes: string[];
  /**
   * Whether the intensity volume the case was segmented from is still on disk.
   * The phantom is generated geometry with no scan behind it, so contours over
   * black would show nothing and the viewer falls back to filled labels.
   */
  has_source_image: boolean;
  note: string;
}
