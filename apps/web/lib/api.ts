/**
 * Typed client for the platform API.
 *
 * Requests go to relative `/api/...` paths, which `next.config.mjs` rewrites to
 * the FastAPI server. Same-origin from the browser's point of view, so there is no
 * CORS preflight and the API stays bound to loopback.
 */

import type {
  CaseListEntry,
  CaseSummary,
  MeshManifest,
  ModelInfo,
  RobotCommandSequence,
  SimulationStateInfo,
  SliceInfo,
  UploadJob,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });

  if (!response.ok) {
    // The API returns {detail: "..."} for domain errors. Surface that rather than
    // a bare status, because "409" alone tells a surgeon nothing about why a plan
    // was refused.
    let detail: string | undefined;
    try {
      const body = await response.json();
      detail = typeof body?.detail === "string" ? body.detail : JSON.stringify(body?.detail);
    } catch {
      detail = undefined;
    }
    throw new ApiError(detail ?? `${response.status} ${response.statusText}`, response.status, detail);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<{ status: string; banner: string; disclaimer: string }>("/api/health"),

  model: () => request<ModelInfo>("/api/model"),

  listCases: () => request<{ n_cases: number; cases: CaseListEntry[] }>("/api/cases"),

  case: (caseId: string) => request<CaseSummary>(`/api/cases/${encodeURIComponent(caseId)}`),

  meshManifest: (caseId: string) =>
    request<MeshManifest>(`/api/cases/${encodeURIComponent(caseId)}/meshes`),

  /** Absolute URL for a GLB, handed to the loader rather than fetched here. */
  meshUrl: (caseId: string, filename: string) =>
    `/api/cases/${encodeURIComponent(caseId)}/meshes/${encodeURIComponent(filename)}`,

  sliceInfo: (caseId: string) =>
    request<SliceInfo>(`/api/cases/${encodeURIComponent(caseId)}/slices`),

  /** URL for a rendered slice PNG, loaded by an <img> rather than fetched here. */
  sliceUrl: (caseId: string, plane: string, index: number, mode = "contour", zoom = 2) =>
    `/api/cases/${encodeURIComponent(caseId)}/slice.png` +
    `?plane=${plane}&index=${index}&mode=${mode}&zoom=${zoom}`,

  simulationState: (caseId: string) =>
    request<SimulationStateInfo>(`/api/cases/${encodeURIComponent(caseId)}/simulation/state`),

  approve: (caseId: string, approver: string, justification: string) =>
    request<SimulationStateInfo>(`/api/cases/${encodeURIComponent(caseId)}/simulation/approve`, {
      method: "POST",
      body: JSON.stringify({ approver, justification }),
    }),

  runSimulation: (caseId: string) =>
    request<{
      state: SimulationStateInfo;
      commands: RobotCommandSequence;
      robot_plan: Record<string, unknown>;
    }>(`/api/cases/${encodeURIComponent(caseId)}/simulation/run`, { method: "POST" }),

  stopSimulation: (caseId: string, actor = "operator", reason = "operator pressed stop") =>
    request<SimulationStateInfo>(`/api/cases/${encodeURIComponent(caseId)}/simulation/stop`, {
      method: "POST",
      body: JSON.stringify({ actor, reason }),
    }),

  resetSimulation: (caseId: string) =>
    request<{ case_id: string; state: string; reset: boolean }>(
      `/api/cases/${encodeURIComponent(caseId)}/simulation/reset`,
      { method: "POST" },
    ),

  robotCommands: (caseId: string) =>
    request<RobotCommandSequence>(`/api/cases/${encodeURIComponent(caseId)}/robot/commands`),

  // --- Pipeline trigger (bridge from existing segmentation model) ---

  /** List cases available in the dataset for pipeline processing. */
  availableCases: () =>
    request<{ n_cases: number; cases: { case_id: string; split: string }[] }>("/api/pipeline/cases"),

  /** Trigger the unified pipeline for a case. Runs synchronously (30-40s on CPU). */
  runPipeline: (caseId: string, strategy = "restricted_kinematic", side = "right") =>
    request<{
      case_id: string;
      case_dir: string;
      planned: boolean;
      total_time_s: number;
      warnings_count: number;
      artifact_paths: Record<string, string>;
    }>(`/api/pipeline/run`, {
      method: "POST",
      body: JSON.stringify({ case_id: caseId, strategy, side }),
    }),

  // --- Upload ---

  /** Upload a scan file and start processing. Returns immediately with a pollable job. */
  uploadScan: async (
    file: File,
    options: { side?: string; modality?: string; case_id?: string } = {},
  ): Promise<UploadJob> => {
    const form = new FormData();
    form.append("file", file);
    if (options.side) form.append("side", options.side);
    if (options.modality) form.append("modality", options.modality);
    if (options.case_id) form.append("case_id", options.case_id);
    // Upload directly to FastAPI (bypasses Next.js proxy body size limit)
    const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
    const response = await fetch(`${apiBase}/api/upload`, { method: "POST", body: form });
    if (!response.ok) {
      let detail: string | undefined;
      try {
        const body = await response.json();
        detail = typeof body?.detail === "string" ? body.detail : JSON.stringify(body?.detail);
      } catch {
        detail = undefined;
      }
      throw new ApiError(detail ?? `${response.status} ${response.statusText}`, response.status, detail);
    }
    return (await response.json()) as UploadJob;
  },

  /** Poll upload progress. */
  uploadStatus: async (caseId: string): Promise<UploadJob> => {
    const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
    const response = await fetch(`${apiBase}/api/uploads/${encodeURIComponent(caseId)}`);
    if (!response.ok) {
      throw new ApiError(`${response.status}`, response.status);
    }
    return (await response.json()) as UploadJob;
  },

  /** List all uploads. */
  listUploads: () =>
    request<{ jobs: UploadJob[]; queue_depth: number }>("/api/uploads"),
};

/** Format a measurement value, or say plainly that it is unavailable. */
export function fmt(value: number | null | undefined, unit?: string | null, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const suffix = unit ? (unit === "%" || unit === "deg" ? unit : ` ${unit}`) : "";
  return `${value.toFixed(digits)}${unit === "deg" ? "°" : suffix}`;
}

/** Colour per anatomical structure. Bone neutral, cartilage warm, meniscus cool. */
export const STRUCTURE_COLORS: Record<string, string> = {
  femur: "#e8e2d5",
  tibia: "#ded7c8",
  patella: "#d5cebe",
  fibula: "#cdc5b4",
  femoral_cartilage: "#5fb8e8",
  medial_tibial_cartilage: "#4fa8dd",
  lateral_tibial_cartilage: "#63c2ef",
  patellar_cartilage: "#7fcdf2",
  medial_meniscus: "#e0a94f",
  lateral_meniscus: "#e8bb68",
  fibula_resected: "#c8b9a0",
};

/**
 * Colour for an anatomical structure. Implant components do not come through here:
 * `Viewer3D` gives anything with `category === "component"` a polished metal
 * material, since a flat colour cannot read as metal regardless of the hex value.
 */
export function structureColor(structure: string): string {
  return STRUCTURE_COLORS[structure] ?? "#9aa7b4";
}

export function prettyStructure(structure: string): string {
  return structure
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}
