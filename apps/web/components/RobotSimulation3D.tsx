"use client";

/**
 * Robot cell simulation: a 3-axis gantry driving the planned tool poses.
 *
 * The command sequence already carries a position and a tool axis per step, in image
 * millimetres. This renders a gantry that reaches those poses so the motion is legible
 * as machine motion — rails, carriage, column, spindle — rather than a marker teleporting
 * between points.
 *
 * The gantry is **illustrative**. No robot geometry exists in the plan, so its rails are
 * sized from the safety boundary and its axes are Cartesian XYZ. A real Mako or ROSA is
 * a serial arm with rotary joints and a different workspace. What is faithful here is the
 * tool: its position, its axis, and the order it visits them all come from
 * ``robot_commands``, and the boundary drawn around it is the one the safety report
 * computed.
 *
 * Presentation notes: the keep-out volume is drawn as corner brackets rather than a full
 * wireframe box, because a closed box reads as an object and competes with the machine;
 * brackets read as an annotation. Orientation is carried by a corner gizmo instead of a
 * labelled triad sitting in the scene, so nothing overlaps the tool path. The environment
 * is built from lightformers rather than a preset HDRI so the metal reads as metal with
 * no network fetch.
 *
 * SIMULATION ONLY. Nothing here can drive hardware; there is no hardware attached.
 */

import { Suspense, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import {
  Billboard,
  Environment,
  GizmoHelper,
  GizmoViewport,
  Grid,
  Lightformer,
  Line,
  OrbitControls,
  useGLTF,
} from "@react-three/drei";
import * as THREE from "three";

import { api } from "@/lib/api";
import type {
  MeshEntry,
  MeshManifest,
  RobotCommandEntry,
  RobotCommandSequence,
} from "@/lib/types";

/** Image (LPS) axes into the view's Y-up convention. */
const LPS_TO_VIEW = new THREE.Euler(-Math.PI / 2, 0, 0);

const STRUCTURE = "#13596e"; // anodised machine body
const STRUCTURE_DARK = "#0d4657";
const ACCENT = "#2dd4bf";
const CUT_ACCENT = "#fbbf24";
const TOOL_METAL = "#dfe6ee";
const BONE = "#ded8ca";

/** Commands that represent cutting, so the tool can be shown as active. */
const CUTTING = /RESECTION/i;

interface Pose {
  position: THREE.Vector3;
  axis: THREE.Vector3;
  command: RobotCommandEntry;
}

/** Quaternion that points the spindle's local +Y down the tool axis. */
function orientationFor(axis: THREE.Vector3): THREE.Quaternion {
  return new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    axis.clone().normalize(),
  );
}

/**
 * Shared material for every structural member of the gantry.
 *
 * Metalness is deliberately moderate. At 0.85 the members reflect almost nothing but the
 * environment, and in a dark cell that renders as black-on-black: the whole machine
 * disappeared into the background. Half-metal keeps the diffuse term that makes the
 * members readable.
 */
function MachineMaterial({ dark = false }: { dark?: boolean }) {
  return (
    <meshStandardMaterial
      color={dark ? STRUCTURE_DARK : STRUCTURE}
      metalness={0.55}
      roughness={0.4}
      envMapIntensity={0.7}
    />
  );
}

/**
 * The bone under the tool.
 *
 * The poses and the meshes are both in image millimetres, so the anatomy drops straight
 * into the same frame with no registration step. Without it the cell was a gantry moving
 * through an empty box, which is the least informative way to show a resection: the whole
 * point of a tool path is where it sits relative to the bone. Resected surfaces are
 * preferred when the plan produced them, since that is the geometry the sequence cuts.
 */
function BonePart({ url }: { url: string }) {
  const { scene } = useGLTF(url);

  const object = useMemo(() => {
    const clone = scene.clone(true);
    const material = new THREE.MeshStandardMaterial({
      color: BONE,
      roughness: 0.66,
      metalness: 0.04,
      envMapIntensity: 0.5,
      transparent: true,
      opacity: 0.93,
    });
    clone.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (!mesh.isMesh) return;
      if (!mesh.geometry.attributes.normal) mesh.geometry.computeVertexNormals();
      mesh.material = material;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    });
    return clone;
  }, [scene]);

  return <primitive object={object} />;
}

/** Bone only: cartilage and meniscus masks are fragmented and would read as noise here. */
function boneEntries(manifest: MeshManifest | null | undefined): MeshEntry[] {
  const all = manifest?.meshes ?? [];
  const isBone = (m: MeshEntry) => /^(femur|tibia|patella|fibula)/.test(m.structure);
  const resected = all.filter((m) => m.category === "resected" && isBone(m));
  return resected.length ? resected : all.filter((m) => m.category === "anatomy" && isBone(m));
}

function Gantry({
  poses,
  progress,
  bounds,
  onStep,
}: {
  poses: Pose[];
  progress: number;
  bounds: { min: THREE.Vector3; max: THREE.Vector3 };
  onStep: (index: number, position: THREE.Vector3, axis: THREE.Vector3) => void;
}) {
  const carriage = useRef<THREE.Group>(null);
  const column = useRef<THREE.Group>(null);
  const spindle = useRef<THREE.Group>(null);
  const xBridge = useRef<THREE.Group>(null);
  const tip = useRef<THREE.Mesh>(null);

  const lastStep = useRef(-1);

  useFrame(({ clock }) => {
    if (poses.length === 0) return;

    // Interpolate along the pose list so travel between commands is continuous.
    const t = Math.max(0, Math.min(1, progress)) * (poses.length - 1);
    const i = Math.floor(t);
    const frac = t - i;
    const a = poses[i];
    const b = poses[Math.min(i + 1, poses.length - 1)];

    const pos = a.position.clone().lerp(b.position, frac);
    const axis = a.axis.clone().lerp(b.axis, frac).normalize();

    // Gantry kinematics: bridge carries X, carriage carries Y, column carries Z.
    if (xBridge.current) xBridge.current.position.x = pos.x;
    if (carriage.current) carriage.current.position.y = pos.y;
    if (column.current) column.current.position.z = pos.z;
    if (spindle.current) spindle.current.quaternion.copy(orientationFor(axis));

    // The tip glows while the active command is a resection, so cutting is visible
    // in the scene and not only in the readout.
    const step = Math.round(t);
    if (tip.current) {
      const active = poses[Math.min(step, poses.length - 1)]?.command;
      const cutting = active ? CUTTING.test(active.action) : false;
      const mat = tip.current.material as THREE.MeshBasicMaterial;
      const pulse = 0.55 + 0.45 * Math.sin(clock.elapsedTime * 9);
      mat.color.set(cutting ? CUT_ACCENT : ACCENT);
      tip.current.scale.setScalar(cutting ? 1 + pulse * 0.5 : 1);
    }

    if (step !== lastStep.current) {
      lastStep.current = step;
      const at = poses[Math.min(step, poses.length - 1)];
      onStep(step, at.position, at.axis);
    }
  });

  const size = bounds.max.clone().sub(bounds.min);
  const mid = bounds.min.clone().add(size.clone().multiplyScalar(0.5));
  const railZ = bounds.min.z - 8;

  return (
    <group>
      {/* --- Static frame: two X rails at the workspace floor --- */}
      {[bounds.min.y, bounds.max.y].map((y) => (
        <group key={y} position={[mid.x, y, railZ]}>
          <mesh castShadow>
            <boxGeometry args={[size.x + 40, 5, 5]} />
            <MachineMaterial dark />
          </mesh>
          {/* Linear-guide highlight, so the travel direction is readable. */}
          <mesh position={[0, 0, 2.8]}>
            <boxGeometry args={[size.x + 40, 1.2, 0.8]} />
            <meshBasicMaterial color={ACCENT} transparent opacity={0.5} toneMapped={false} />
          </mesh>
        </group>
      ))}

      {/* --- X bridge: spans Y, travels in X --- */}
      <group ref={xBridge}>
        <mesh position={[0, mid.y, railZ]} castShadow>
          <boxGeometry args={[7, size.y + 14, 7]} />
          <MachineMaterial />
        </mesh>

        {/* --- Y carriage: rides the bridge --- */}
        <group ref={carriage}>
          <mesh position={[0, 0, railZ]} castShadow>
            <boxGeometry args={[16, 13, 11]} />
            <MachineMaterial />
          </mesh>

          {/* --- Z column + spindle ---
              The column is positioned relative to the tool, not to the frame: the group
              is already translated to the current Z, so the member runs back from the tool
              towards the rails and is made long enough to cover the whole travel. */}
          <group ref={column}>
            <mesh position={[0, 0, -size.z / 2 - 8]} castShadow>
              <boxGeometry args={[9, 9, size.z + 16]} />
              <MachineMaterial dark />
            </mesh>

            <group ref={spindle}>
              {/* Spindle housing. */}
              <mesh position={[0, 26, 0]} castShadow>
                <cylinderGeometry args={[5, 4.4, 14, 32]} />
                <MachineMaterial />
              </mesh>
              {/* Tool shaft along local +Y, tip at the origin. */}
              <mesh position={[0, 12, 0]} castShadow>
                <cylinderGeometry args={[2.6, 2.6, 22, 32]} />
                <meshStandardMaterial color={TOOL_METAL} metalness={1} roughness={0.18} />
              </mesh>
              {/* Cutting tip. */}
              <mesh position={[0, 2.4, 0]} castShadow>
                <coneGeometry args={[3, 6, 32]} />
                <meshStandardMaterial color={TOOL_METAL} metalness={1} roughness={0.1} />
              </mesh>
              {/* Tool centre point. */}
              <mesh ref={tip}>
                <sphereGeometry args={[1.5, 20, 20]} />
                <meshBasicMaterial color={ACCENT} toneMapped={false} />
              </mesh>
            </group>
          </group>
        </group>
      </group>
    </group>
  );
}

/**
 * The keep-out volume, drawn as corner brackets from the report's own extents.
 * Brackets read as an annotation around the workspace; a closed wireframe box reads
 * as an object and fights the machine for attention.
 */
function SafetyBrackets({ min, max }: { min: THREE.Vector3; max: THREE.Vector3 }) {
  const segments = useMemo(() => {
    const size = max.clone().sub(min);
    const arm = Math.min(size.x, size.y, size.z) * 0.16;
    const out: Array<[THREE.Vector3, THREE.Vector3]> = [];

    for (const x of [min.x, max.x]) {
      for (const y of [min.y, max.y]) {
        for (const z of [min.z, max.z]) {
          const corner = new THREE.Vector3(x, y, z);
          const sx = x === min.x ? 1 : -1;
          const sy = y === min.y ? 1 : -1;
          const sz = z === min.z ? 1 : -1;
          out.push([corner, corner.clone().add(new THREE.Vector3(arm * sx, 0, 0))]);
          out.push([corner, corner.clone().add(new THREE.Vector3(0, arm * sy, 0))]);
          out.push([corner, corner.clone().add(new THREE.Vector3(0, 0, arm * sz))]);
        }
      }
    }
    return out;
  }, [min, max]);

  return (
    <group>
      {segments.map((points, i) => (
        <Line key={i} points={points} color={ACCENT} lineWidth={1.6} transparent opacity={0.55} />
      ))}
    </group>
  );
}

/** Pulsing ring on the pose currently being served. */
function ActiveMarker({ at }: { at: THREE.Vector3 }) {
  const ring = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    // Scale only. The position lives on the billboard, whose local axes are rotated to
    // face the camera — moving the child inside it would displace the ring.
    if (ring.current) ring.current.scale.setScalar(1 + 0.35 * Math.sin(clock.elapsedTime * 5));
  });

  return (
    <Billboard position={at}>
      <mesh ref={ring}>
        <ringGeometry args={[3.4, 4.1, 32]} />
        <meshBasicMaterial
          color={ACCENT}
          transparent
          opacity={0.7}
          toneMapped={false}
          side={THREE.DoubleSide}
        />
      </mesh>
    </Billboard>
  );
}

/** The tool path as a polyline, so the whole route is visible at once. */
function ToolPath({ poses, progress }: { poses: Pose[]; progress: number }) {
  const points = useMemo(() => poses.map((p) => p.position), [poses]);

  const done = points.length >= 2 ? Math.max(1, Math.floor(progress * (points.length - 1)) + 1) : 1;
  const active = Math.min(points.length - 1, Math.max(0, done - 1));

  if (points.length < 2) return null;

  return (
    <>
      {/* Whole route, dim. */}
      <Line points={points} color="#0e7490" lineWidth={1} transparent opacity={0.45} dashed dashSize={4} gapSize={3} />
      {/* Route travelled so far. */}
      {done >= 2 && <Line points={points.slice(0, done)} color={ACCENT} lineWidth={2.4} />}

      {poses.map((p, i) => {
        const cutting = CUTTING.test(p.command.action);
        const visited = i < done;
        return (
          <mesh key={i} position={p.position}>
            <octahedronGeometry args={[cutting ? 2.4 : 1.5, 0]} />
            <meshBasicMaterial
              color={visited ? (cutting ? CUT_ACCENT : ACCENT) : "#14505f"}
              toneMapped={false}
            />
          </mesh>
        );
      })}

      <ActiveMarker at={points[active]} />
    </>
  );
}

/** Lightformer rig: polished metal needs something to reflect, and this needs no network. */
function StudioEnvironment() {
  return (
    <Environment resolution={256} frames={1}>
      <color attach="background" args={["#04161f"]} />
      <Lightformer intensity={2.2} position={[0, 6, 0]} scale={[12, 3, 1]} color="#bff5ef" />
      <Lightformer intensity={0.9} position={[6, 2, 4]} scale={[6, 6, 1]} color="#3fb9d4" />
      <Lightformer intensity={0.7} position={[-6, 1, -4]} scale={[6, 6, 1]} color="#0d5f74" />
    </Environment>
  );
}

export default function RobotSimulation3D({
  commands,
  running,
  progress,
  /** Case and mesh manifest, so the cell can show the bone being cut. Optional. */
  caseId,
  manifest,
  /** Fill the parent instead of using the panel's fixed height. */
  fill = false,
  /** The provenance note under the scene. Off in the standalone window, which states it in the footer. */
  showNote = true,
}: {
  commands: RobotCommandSequence | null;
  running: boolean;
  /** 0..1 through the sequence. */
  progress: number;
  caseId?: string | null;
  manifest?: MeshManifest | null;
  fill?: boolean;
  showNote?: boolean;
}) {
  const [live, setLive] = useState({ index: 0, x: 0, y: 0, z: 0 });

  const poses = useMemo<Pose[]>(() => {
    if (!commands) return [];
    return commands.commands
      .filter((c) => c.pose)
      .map((c) => ({
        position: new THREE.Vector3(...(c.pose!.position_mm as [number, number, number])),
        axis: new THREE.Vector3(...(c.pose!.axis as [number, number, number])).normalize(),
        command: c,
      }));
  }, [commands]);

  const bounds = useMemo(() => {
    const b = commands?.safety_boundary;
    if (b) {
      return {
        min: new THREE.Vector3(...(b.min_mm as [number, number, number])),
        max: new THREE.Vector3(...(b.max_mm as [number, number, number])),
      };
    }
    if (poses.length === 0) {
      return { min: new THREE.Vector3(0, 0, 0), max: new THREE.Vector3(100, 100, 100) };
    }
    const box = new THREE.Box3();
    poses.forEach((p) => box.expandByPoint(p.position));
    box.expandByScalar(20);
    return { min: box.min.clone(), max: box.max.clone() };
  }, [commands?.safety_boundary, poses]);

  const centre = useMemo(
    () => bounds.min.clone().add(bounds.max).multiplyScalar(0.5),
    [bounds],
  );
  const span = useMemo(() => bounds.max.clone().sub(bounds.min).length(), [bounds]);
  /**
   * Floor height in view space. The scene group is rotated -90° about X, so image Z
   * becomes view Y: the bottom of the workspace sits at -size.z/2.
   */
  const floorY = useMemo(() => -(bounds.max.z - bounds.min.z) / 2 - 14, [bounds]);

  const bones = useMemo(() => boneEntries(manifest), [manifest]);

  const activeCommand = poses[Math.min(live.index, Math.max(0, poses.length - 1))]?.command;
  const isCutting = activeCommand ? CUTTING.test(activeCommand.action) : false;

  if (!commands || poses.length === 0) {
    return (
      <div className={`flex items-center px-4 ${fill ? "h-full justify-center text-center" : "py-3"}`}>
        <p className="max-w-sm text-[12px] leading-relaxed text-teal-100/50">
          No posed commands for this case yet. The sequence is derived from the plan —
          approve the plan and it will appear here.
        </p>
      </div>
    );
  }

  return (
    <div className={fill ? "flex h-full flex-col" : undefined}>
      <div className={`relative w-full bg-clinic-950 ${fill ? "min-h-0 flex-1" : "h-72"}`}>
        <Canvas
          camera={{ position: [span * 0.85, span * 0.6, span * 0.95], fov: 34, near: 1, far: 6000 }}
          dpr={[1, 2]}
          shadows
          gl={{ antialias: true }}
        >
          <color attach="background" args={["#04161f"]} />
          <fog attach="fog" args={["#04161f", span * 1.6, span * 3.6]} />

          <ambientLight intensity={0.6} />
          {/* The shadow camera has to be sized to the cell: the default orthographic
              frustum is a few units across and would clip the machine entirely. */}
          <directionalLight
            position={[span * 0.6, span * 0.9, span * 0.5]}
            intensity={1.5}
            castShadow
            shadow-mapSize={[1024, 1024]}
            shadow-camera-left={-span}
            shadow-camera-right={span}
            shadow-camera-top={span}
            shadow-camera-bottom={-span}
            shadow-camera-near={1}
            shadow-camera-far={span * 4}
            shadow-bias={-0.0006}
          />
          <directionalLight position={[-span * 0.5, -span * 0.2, -span * 0.4]} intensity={0.55} />

          <Suspense fallback={null}>
            <StudioEnvironment />
          </Suspense>

          {/* Catches the gantry's shadow and nothing else, so the machine reads as
              standing on the floor rather than floating over a grid. */}
          <mesh
            position={[0, floorY + 0.1, 0]}
            rotation={[-Math.PI / 2, 0, 0]}
            receiveShadow
          >
            <planeGeometry args={[span * 3, span * 3]} />
            <shadowMaterial transparent opacity={0.4} />
          </mesh>

          {/* Floor. Gives the machine somewhere to stand and the eye a sense of scale:
              one cell is 10 mm, one section 50 mm. */}
          <Grid
            position={[0, floorY, 0]}
            args={[span * 2.4, span * 2.4]}
            cellSize={10}
            cellThickness={0.6}
            cellColor="#155a70"
            sectionSize={50}
            sectionThickness={1.2}
            sectionColor="#2189a3"
            fadeDistance={span * 2.6}
            fadeStrength={1.6}
            infiniteGrid
            followCamera={false}
          />

          {/* Two nested groups, not one. A single group with both `rotation` and
              `position` applies the translation in world space *after* the rotation, which
              left the cell about 140 mm off the orbit target — the machine sat in the
              corner of the frame and the camera looked at empty space. Rotating the
              already-centred geometry puts the workspace on the origin. */}
          <group rotation={LPS_TO_VIEW}>
            <group position={centre.clone().negate()}>
              {caseId && bones.length > 0 && (
                <Suspense fallback={null}>
                  {bones.map((entry) => (
                    <BonePart key={entry.path} url={api.meshUrl(caseId, entry.path)} />
                  ))}
                </Suspense>
              )}
              <SafetyBrackets min={bounds.min} max={bounds.max} />
              <ToolPath poses={poses} progress={progress} />
              <Gantry
                poses={poses}
                progress={progress}
                bounds={bounds}
                onStep={(index, position) =>
                  setLive({ index, x: position.x, y: position.y, z: position.z })
                }
              />
            </group>
          </group>

          <OrbitControls
            makeDefault
            enableDamping
            dampingFactor={0.1}
            target={[0, 0, 0]}
            minDistance={span * 0.25}
            maxDistance={span * 2.5}
          />

          {/* Orientation lives in the corner rather than as a labelled triad in the
              scene, where it used to sit on top of the tool path. */}
          <GizmoHelper alignment="bottom-right" margin={[58, 58]}>
            <GizmoViewport
              axisColors={["#f87171", "#4ade80", "#60a5fa"]}
              labelColor="#cbe9ef"
              hideNegativeAxes
            />
          </GizmoHelper>
        </Canvas>

        {/* Vignette. Pulls the eye to the middle of the cell. */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              "radial-gradient(120% 90% at 50% 45%, transparent 55%, rgba(2,14,20,0.5) 100%)",
          }}
        />

        {/* --- Digital readout --- */}
        <div className="pointer-events-none absolute left-3 top-3 rounded-xl border border-white/10 bg-clinic-950/80 px-3 py-2 font-mono text-[11px] shadow-float backdrop-blur-md">
          <div className="mb-1.5 flex items-center gap-2">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                running ? "animate-pulse bg-teal-400" : "bg-teal-800"
              }`}
            />
            <span className="text-[9.5px] uppercase tracking-[0.14em] text-teal-300/70">
              tool centre point
            </span>
          </div>
          {(
            [
              ["X", live.x, "text-rose-300"],
              ["Y", live.y, "text-emerald-300"],
              ["Z", live.z, "text-sky-300"],
            ] as Array<[string, number, string]>
          ).map(([label, value, tone]) => (
            <div key={label} className="flex items-baseline gap-2">
              <span className={`w-3 ${tone}`}>{label}</span>
              <span className="w-16 text-right tabular-nums text-white">{value.toFixed(1)}</span>
              <span className="text-teal-200/40">mm</span>
            </div>
          ))}
        </div>

        {/* --- Cell facts --- */}
        <div className="pointer-events-none absolute right-3 top-3 rounded-xl border border-white/10 bg-clinic-950/80 px-2.5 py-1.5 text-right font-mono text-[10px] leading-relaxed text-teal-200/50 shadow-float backdrop-blur-md">
          <div>
            <span className="text-teal-300/80">{poses.length}</span> poses
          </div>
          {commands.safety_boundary && (
            <div>
              keep-out {commands.safety_boundary.size_mm.map((v) => v.toFixed(0)).join("×")} mm
            </div>
          )}
          <div>
            <span className="text-amber-400/80">amber</span> = resection
          </div>
        </div>

        {/* --- Active command --- */}
        <div className="pointer-events-none absolute inset-x-3 bottom-3 rounded-xl border border-white/10 bg-clinic-950/80 px-3 py-2 shadow-float backdrop-blur-md">
          <div className="flex items-baseline justify-between gap-3">
            <span
              className={`truncate font-mono text-[11px] ${
                isCutting ? "text-amber-300" : "text-teal-300"
              }`}
            >
              {activeCommand?.action.replace(/_/g, " ") ?? "idle"}
            </span>
            <span className="shrink-0 font-mono text-[10px] text-teal-200/45">
              {Math.min(live.index + 1, poses.length)}/{poses.length}
            </span>
          </div>
          {activeCommand?.target && (
            <p className="mt-0.5 truncate font-mono text-[10px] text-teal-100/45">
              {activeCommand.target.replace(/_/g, " ")}
            </p>
          )}
          <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-clinic-800">
            <div
              className={`h-full transition-all duration-200 ${
                isCutting ? "bg-amber-400" : "bg-teal-400"
              }`}
              style={{ width: `${Math.max(0, Math.min(100, progress * 100))}%` }}
            />
          </div>
        </div>
      </div>

      {showNote && (
        <div className="px-3 py-2.5">
          <p className="text-[10px] leading-relaxed text-teal-100/45">
            The tool position, axis and visit order are the planner&apos;s own, from{" "}
            <span className="font-mono text-teal-200/70">robot_commands</span>, and the
            boundary is the one the safety report computed. The gantry itself is
            illustrative: no robot geometry exists in the plan, so its rails are sized from
            that boundary and its axes are Cartesian. A clinical system is a serial arm with
            a different workspace. Nothing here can drive hardware.
          </p>
        </div>
      )}
    </div>
  );
}
