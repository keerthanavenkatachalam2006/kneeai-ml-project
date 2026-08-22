"use client";

/**
 * Interactive 3D knee viewer.
 *
 * Geometry arrives as GLB in millimetres, in the platform's IMAGE frame.
 *
 * **Meshes are keyed by `path`, not `structure`.** The manifest carries three
 * entries called `femur` — one `anatomy`, one `resected`, one `component` — so
 * `structure` is not unique and using it as a React key both warns and, worse,
 * renders all three superimposed. `category` is the discriminator; `path` is the
 * identity.
 *
 * **Millimetres stay millimetres.** The scene is not normalised to unit scale, so a
 * 78 mm femoral width is 78 world units and on-screen distance cannot drift from the
 * measurement table. The camera adapts to the mesh bounds instead.
 *
 * **The anatomy frame is LPS, not three.js Y-up.** One rotation on the parent group
 * maps superior to screen-up; doing it once on the parent keeps every child in a
 * single consistent frame, for the same reason `core/frames.py` tags frames rather
 * than trusting convention.
 */

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import {
  Bounds,
  ContactShadows,
  Environment,
  GizmoHelper,
  GizmoViewcube,
  Html,
  OrbitControls,
  useGLTF,
} from "@react-three/drei";
import * as THREE from "three";

import { api, prettyStructure, structureColor } from "@/lib/api";
import type {
  Landmark,
  MeshEntry,
  MeshManifest,
  ResectionCut,
  RobotCommandSequence,
} from "@/lib/types";

export interface ViewerSettings {
  /** Keyed by mesh `path`, which is unique; `structure` is not. */
  visible: Record<string, boolean>;
  opacity: Record<string, number>;
  showLandmarks: boolean;
  showCutPlanes: boolean;
  showImplant: boolean;
  showBoundary: boolean;
  showTrajectory: boolean;
  clipEnabled: boolean;
  clipPosition: number;
  clipAxis: 0 | 1 | 2;
  resectedView: boolean;
}

export interface PickInfo {
  structure: string;
  category: string;
  /** Anatomy-frame position in millimetres. */
  point: [number, number, number];
  entry: MeshEntry;
  nearestLandmark: { name: string; distance_mm: number } | null;
}

interface Props {
  caseId: string;
  manifest: MeshManifest;
  landmarks: Landmark[];
  cuts: ResectionCut[];
  commands: RobotCommandSequence | null;
  settings: ViewerSettings;
  animationProgress: number | null;
  onPick?: (pick: PickInfo | null) => void;
  onCanvasReady?: (gl: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera) => void;
}

/** Anatomy is LPS (+x left, +y posterior, +z superior); three.js wants Y up. */
const LPS_TO_VIEW = new THREE.Euler(-Math.PI / 2, 0, 0);

const SOFT_TISSUE = /cartilage|meniscus/;

/** Implant template colour. Teal reads as "planned overlay", not "photograph". */
const IMPLANT_TEAL = "#2dd4bf";

function materialFor(entry: MeshEntry, opacity: number, clip: THREE.Plane[] | undefined) {
  const isComponent = entry.category === "component";
  const isResected = entry.category === "resected";
  const isSoft = SOFT_TISSUE.test(entry.structure);

  if (isComponent) {
    /**
     * Rendered as a translucent teal template, not polished metal.
     *
     * The reference catalogue components are ~300 vertices — dimensionally correct
     * (67 mm femoral ML, 70 mm tibial ML) but geometrically schematic extruded
     * plates. A photoreal chrome shader on a 300-vertex plate looks like a bug,
     * because the eye expects metal to carry reflections it has no surface detail
     * to produce. A translucent template says "this is a planned position and an
     * envelope", which is exactly what this geometry honestly represents, and it
     * also lets the resected bone underneath stay visible.
     */
    return new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(IMPLANT_TEAL),
      metalness: 0.25,
      roughness: 0.25,
      clearcoat: 1.0,
      clearcoatRoughness: 0.1,
      transparent: true,
      opacity: Math.min(opacity, 0.62),
      depthWrite: false,
      side: THREE.DoubleSide,
      emissive: new THREE.Color("#0d9488"),
      emissiveIntensity: 0.28,
      envMapIntensity: 1.1,
      clippingPlanes: clip,
    });
  }

  return new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(structureColor(entry.structure)),
    metalness: 0,
    roughness: isSoft ? 0.34 : 0.5,
    clearcoat: isSoft ? 0.7 : 0.28,
    clearcoatRoughness: isSoft ? 0.25 : 0.5,
    sheen: isSoft ? 0.45 : 0,
    sheenColor: new THREE.Color("#bfe4ff"),
    // Resected bone is tinted slightly cooler so the cut surface is distinguishable
    // from intact cortex in the before/after comparison.
    emissive: isResected ? new THREE.Color("#16222e") : new THREE.Color("#000000"),
    emissiveIntensity: isResected ? 0.35 : 0,
    transparent: opacity < 0.995,
    opacity,
    depthWrite: opacity > 0.9,
    side: THREE.DoubleSide,
    clippingPlanes: clip,
    clipShadows: true,
    envMapIntensity: 0.55,
  });
}

function MeshPart({
  url,
  entry,
  opacity,
  clip,
  onPick,
}: {
  url: string;
  entry: MeshEntry;
  opacity: number;
  clip: THREE.Plane[] | undefined;
  onPick: (entry: MeshEntry, localPoint: THREE.Vector3) => void;
}) {
  const { scene } = useGLTF(url);
  const [hovered, setHovered] = useState(false);

  const object = useMemo(() => {
    const clone = scene.clone(true);
    const material = materialFor(entry, opacity, clip);
    const isComponent = entry.category === "component";

    clone.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (!mesh.isMesh) return;
      const geometry = mesh.geometry as THREE.BufferGeometry;
      if (!geometry.attributes.normal) geometry.computeVertexNormals();
      mesh.material = material;
      mesh.castShadow = !isComponent && !SOFT_TISSUE.test(entry.structure);
      mesh.receiveShadow = !isComponent;

      // A translucent low-poly plate has almost no silhouette. Wireframe edges give
      // it one, which is what makes the component legible at all against bone.
      if (isComponent) {
        const edges = new THREE.LineSegments(
          new THREE.EdgesGeometry(geometry, 25),
          new THREE.LineBasicMaterial({
            color: IMPLANT_TEAL,
            transparent: true,
            opacity: 0.95,
            toneMapped: false,
          }),
        );
        mesh.add(edges);
      }
    });
    return clone;
  }, [scene, entry, opacity, clip]);

  const handleClick = useCallback(
    (e: ThreeEvent<MouseEvent>) => {
      e.stopPropagation();
      // e.point is world space; the parent group is rotated, so convert back into
      // the anatomy frame before reporting millimetres.
      const local = e.object.worldToLocal(e.point.clone());
      onPick(entry, local);
    },
    [entry, onPick],
  );

  return (
    <primitive
      object={object}
      onClick={handleClick}
      onPointerOver={(e: ThreeEvent<PointerEvent>) => {
        e.stopPropagation();
        setHovered(true);
        document.body.style.cursor = "crosshair";
      }}
      onPointerOut={() => {
        setHovered(false);
        document.body.style.cursor = "";
      }}
    />
  );
}

function PickMarker({ pick, onClear }: { pick: PickInfo; onClear: () => void }) {
  return (
    <group position={pick.point}>
      <mesh>
        <sphereGeometry args={[1.8, 20, 20]} />
        <meshBasicMaterial color="#38bdf8" toneMapped={false} />
      </mesh>
      <mesh>
        <ringGeometry args={[3.2, 4.0, 32]} />
        <meshBasicMaterial color="#38bdf8" transparent opacity={0.55} side={THREE.DoubleSide} toneMapped={false} />
      </mesh>
      <Html distanceFactor={190} zIndexRange={[40, 0]}>
        <div className="w-56 -translate-x-1/2 translate-y-4 rounded-lg border border-sky-700/70 bg-slate-950/95 p-2.5 shadow-2xl backdrop-blur">
          <div className="mb-1 flex items-start justify-between gap-2">
            <span className="text-[12px] font-semibold leading-tight text-slate-50">
              {prettyStructure(pick.structure)}
            </span>
            <button
              type="button"
              onClick={onClear}
              className="-mr-1 -mt-1 px-1 text-slate-500 hover:text-slate-200"
              aria-label="Close"
            >
              ×
            </button>
          </div>
          <div className="mb-1.5 text-[10px] uppercase tracking-wider text-sky-400">
            {pick.category}
          </div>
          <dl className="space-y-0.5 font-mono text-[10.5px] text-slate-400">
            <div className="flex justify-between gap-2">
              <dt>x</dt>
              <dd className="text-slate-200">{pick.point[0].toFixed(1)} mm</dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt>y</dt>
              <dd className="text-slate-200">{pick.point[1].toFixed(1)} mm</dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt>z</dt>
              <dd className="text-slate-200">{pick.point[2].toFixed(1)} mm</dd>
            </div>
            {pick.entry.volume_mm3 != null && (
              <div className="flex justify-between gap-2 border-t border-slate-800 pt-0.5">
                <dt>volume</dt>
                <dd className="text-slate-200">
                  {(pick.entry.volume_mm3 / 1000).toFixed(1)} cm³
                </dd>
              </div>
            )}
            <div className="flex justify-between gap-2">
              <dt>faces</dt>
              <dd className="text-slate-200">{pick.entry.n_faces.toLocaleString()}</dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt>watertight</dt>
              <dd className={pick.entry.is_watertight ? "text-emerald-400" : "text-amber-400"}>
                {pick.entry.is_watertight ? "yes" : "no"}
              </dd>
            </div>
          </dl>
          {pick.nearestLandmark && (
            <p className="mt-1.5 border-t border-slate-800 pt-1.5 text-[10px] leading-snug text-slate-500">
              nearest landmark:{" "}
              <span className="text-slate-300">
                {prettyStructure(pick.nearestLandmark.name)}
              </span>{" "}
              at {pick.nearestLandmark.distance_mm.toFixed(1)} mm
            </p>
          )}
        </div>
      </Html>
    </group>
  );
}

function LandmarkMarkers({ landmarks }: { landmarks: Landmark[] }) {
  const [hovered, setHovered] = useState<string | null>(null);
  return (
    <group>
      {landmarks.map((lm) => {
        if (!lm.position || lm.position.length < 3) return null;
        const active = hovered === lm.name;
        const color = new THREE.Color().setHSL(
          0.33 * Math.min(1, Math.max(0, lm.confidence)),
          0.8,
          0.55,
        );
        return (
          <group key={lm.name} position={lm.position as [number, number, number]}>
            <mesh
              onPointerOver={(e) => {
                e.stopPropagation();
                setHovered(lm.name);
              }}
              onPointerOut={() => setHovered(null)}
            >
              <sphereGeometry args={[active ? 2.4 : 1.5, 20, 20]} />
              <meshBasicMaterial color={color} toneMapped={false} />
            </mesh>
            {active && (
              <Html distanceFactor={200} zIndexRange={[30, 0]}>
                <div className="pointer-events-none -translate-x-1/2 translate-y-3 whitespace-nowrap rounded-md border border-slate-600 bg-slate-950/95 px-2.5 py-1.5 shadow-xl">
                  <div className="text-[11px] font-semibold text-slate-100">
                    {lm.display_name ?? prettyStructure(lm.name)}
                  </div>
                  <div className="font-mono text-[10px] text-slate-400">
                    confidence {lm.confidence.toFixed(2)}
                    {lm.uncertainty_mm != null && ` · ±${lm.uncertainty_mm.toFixed(1)} mm`}
                  </div>
                </div>
              </Html>
            )}
          </group>
        );
      })}
    </group>
  );
}

function CutPlanes({ cuts }: { cuts: ResectionCut[] }) {
  return (
    <group>
      {cuts.map((cut, i) => {
        const p = cut.plane?.point;
        const n = cut.plane?.normal;
        if (!p || !n || p.length < 3 || n.length < 3) return null;
        const normal = new THREE.Vector3(n[0], n[1], n[2]).normalize();
        const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
        // 90 mm squares overlapped into visual noise and hid the joint. A resection
        // plane only needs to read where it meets the bone, so this is sized just
        // past the widest condyle and drawn as an outline with a faint fill.
        const SIDE = 58;
        const geo = new THREE.PlaneGeometry(SIDE, SIDE);
        return (
          <group key={`${cut.name}-${i}`} position={p as [number, number, number]} quaternion={quat}>
            <mesh>
              <planeGeometry args={[SIDE, SIDE]} />
              <meshBasicMaterial
                color="#38bdf8"
                transparent
                opacity={0.06}
                side={THREE.DoubleSide}
                depthWrite={false}
                toneMapped={false}
              />
            </mesh>
            <lineSegments>
              <edgesGeometry args={[geo]} />
              <lineBasicMaterial color="#7dd3fc" transparent opacity={0.5} toneMapped={false} />
            </lineSegments>
          </group>
        );
      })}
    </group>
  );
}

function BoundaryBox({ min, max }: { min: number[]; max: number[] }) {
  const geometry = useMemo(
    () => new THREE.BoxGeometry(max[0] - min[0], max[1] - min[1], max[2] - min[2]),
    [min, max],
  );
  return (
    <group position={[(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2]}>
      <lineSegments>
        <edgesGeometry args={[geometry]} />
        <lineBasicMaterial color="#fbbf24" transparent opacity={0.5} toneMapped={false} />
      </lineSegments>
    </group>
  );
}

function Trajectory({
  commands,
  progress,
}: {
  commands: RobotCommandSequence;
  progress: number | null;
}) {
  const poses = useMemo(
    () => commands.commands.filter((c) => c.pose?.position_mm?.length === 3),
    [commands],
  );
  const points = useMemo(
    () => poses.map((c) => new THREE.Vector3(...(c.pose!.position_mm as [number, number, number]))),
    [poses],
  );
  const geometry = useMemo(() => new THREE.BufferGeometry().setFromPoints(points), [points]);
  const tool = useRef<THREE.Group>(null);

  useFrame(() => {
    if (progress === null || points.length < 2 || !tool.current) return;
    const t = Math.min(0.999, Math.max(0, progress)) * (points.length - 1);
    const i = Math.floor(t);
    tool.current.position.lerpVectors(points[i], points[Math.min(i + 1, points.length - 1)], t - i);
    const axis = poses[i].pose!.axis;
    if (axis?.length === 3) {
      tool.current.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3(axis[0], axis[1], axis[2]).normalize(),
      );
    }
  });

  if (points.length < 2) return null;

  return (
    <group>
      <line>
        <primitive object={geometry} attach="geometry" />
        <lineBasicMaterial color="#34d399" transparent opacity={0.65} toneMapped={false} />
      </line>
      {points.map((p, i) => (
        <mesh key={i} position={p}>
          <sphereGeometry args={[1.1, 12, 12]} />
          <meshBasicMaterial color="#34d399" transparent opacity={0.5} toneMapped={false} />
        </mesh>
      ))}
      {progress !== null && (
        <group ref={tool}>
          {/* A cutting guide, not a robot arm. This system has no kinematic model,
              and drawing a plausible arm would imply one it does not have. */}
          <mesh position={[0, 30, 0]} castShadow>
            <cylinderGeometry args={[2.6, 2.6, 60, 24]} />
            <meshPhysicalMaterial color="#dfe6ee" metalness={1} roughness={0.16} clearcoat={1} envMapIntensity={2} />
          </mesh>
          <mesh position={[0, 4, 0]} castShadow>
            <cylinderGeometry args={[5.0, 2.8, 8, 24]} />
            <meshPhysicalMaterial color="#94a3b8" metalness={1} roughness={0.3} />
          </mesh>
          <mesh castShadow>
            <coneGeometry args={[2.9, 8, 24]} />
            <meshPhysicalMaterial color="#f1f5f9" metalness={1} roughness={0.1} clearcoat={1} envMapIntensity={2.4} />
          </mesh>
        </group>
      )}
    </group>
  );
}

function CanvasBridge({ onReady }: { onReady: Props["onCanvasReady"] }) {
  const { gl, scene, camera } = useThree();
  useEffect(() => {
    gl.localClippingEnabled = true;
    onReady?.(gl, scene, camera);
  }, [gl, scene, camera, onReady]);
  return null;
}

export default function Viewer3D({
  caseId,
  manifest,
  landmarks,
  cuts,
  commands,
  settings,
  animationProgress,
  onPick,
  onCanvasReady,
}: Props) {
  const [pick, setPick] = useState<PickInfo | null>(null);

  /**
   * Selection by `category`, which is the actual discriminator. Testing the
   * structure name for a `_resected` suffix matched nothing, so every intact bone,
   * its resected twin and the implant rendered on top of one another.
   */
  const entries = useMemo(() => {
    const all = manifest.meshes ?? [];
    const hasResected = new Set(
      all.filter((m) => m.category === "resected").map((m) => m.structure),
    );
    return all.filter((m) => {
      if (m.category === "component") return settings.showImplant;
      if (m.category === "resected") return settings.resectedView;
      // Intact bone yields to its resected twin in the "after" view.
      if (settings.resectedView && hasResected.has(m.structure)) return false;
      return true;
    });
  }, [manifest.meshes, settings.resectedView, settings.showImplant]);

  const clip = useMemo(() => {
    if (!settings.clipEnabled) return undefined;
    const n = new THREE.Vector3();
    n.setComponent(settings.clipAxis, 1);
    return [new THREE.Plane(n, -settings.clipPosition)];
  }, [settings.clipEnabled, settings.clipAxis, settings.clipPosition]);

  const handlePick = useCallback(
    (entry: MeshEntry, local: THREE.Vector3) => {
      let nearest: PickInfo["nearestLandmark"] = null;
      for (const lm of landmarks) {
        if (!lm.position || lm.position.length < 3) continue;
        const d = local.distanceTo(new THREE.Vector3(...(lm.position as [number, number, number])));
        if (!nearest || d < nearest.distance_mm) {
          nearest = { name: lm.name, distance_mm: d };
        }
      }
      const info: PickInfo = {
        structure: entry.structure,
        category: entry.category,
        point: [local.x, local.y, local.z],
        entry,
        nearestLandmark: nearest,
      };
      setPick(info);
      onPick?.(info);
    },
    [landmarks, onPick],
  );

  const clearPick = useCallback(() => {
    setPick(null);
    onPick?.(null);
  }, [onPick]);

  return (
    <Canvas
      shadows
      camera={{ position: [260, 150, 260], fov: 35, near: 1, far: 5000 }}
      dpr={[1, 2]}
      gl={{
        preserveDrawingBuffer: true,
        antialias: true,
        localClippingEnabled: true,
        toneMapping: THREE.ACESFilmicToneMapping,
        toneMappingExposure: 1.05,
      }}
      onPointerMissed={clearPick}
    >
      <CanvasBridge onReady={onCanvasReady} />

      {/* Image-based lighting does most of the work; the directional light adds a
          highlight direction so condylar curvature and implant polish both read. */}
      <Environment preset="studio" environmentIntensity={0.55} />
      <ambientLight intensity={0.32} />
      <directionalLight
        position={[180, 240, 160]}
        intensity={1.5}
        castShadow
        shadow-mapSize={[1024, 1024]}
      />
      <directionalLight position={[-160, -60, -140]} intensity={0.4} />

      <group rotation={LPS_TO_VIEW}>
        <Suspense fallback={null}>
          <Bounds fit clip observe margin={1.25}>
            <group>
              {entries.map((entry) =>
                settings.visible[entry.path] === false ? null : (
                  <MeshPart
                    key={entry.path}
                    url={api.meshUrl(caseId, entry.path)}
                    entry={entry}
                    opacity={settings.opacity[entry.path] ?? 1}
                    clip={clip}
                    onPick={handlePick}
                  />
                ),
              )}
            </group>
          </Bounds>
        </Suspense>

        {settings.showLandmarks && <LandmarkMarkers landmarks={landmarks} />}
        {settings.showCutPlanes && <CutPlanes cuts={cuts} />}
        {settings.showBoundary && commands?.safety_boundary && (
          <BoundaryBox
            min={commands.safety_boundary.min_mm}
            max={commands.safety_boundary.max_mm}
          />
        )}
        {settings.showTrajectory && commands && (
          <Trajectory commands={commands} progress={animationProgress} />
        )}
        {pick && <PickMarker pick={pick} onClear={clearPick} />}
      </group>

      <ContactShadows
        position={[0, -95, 0]}
        opacity={0.35}
        scale={420}
        blur={2.6}
        far={180}
        color="#000000"
      />

      <GizmoHelper alignment="bottom-right" margin={[68, 68]}>
        <GizmoViewcube
          color="#1e293b"
          strokeColor="#475569"
          textColor="#cbd5e1"
          hoverColor="#0ea5e9"
        />
      </GizmoHelper>

      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.07}
        rotateSpeed={0.75}
        zoomSpeed={0.85}
        minDistance={50}
        maxDistance={1400}
      />
    </Canvas>
  );
}
