"use client";

/**
 * Standalone implant preview: the components on their own, no anatomy.
 *
 * Comparing sizes on the bone does not work. The bone is the thing that stays the same
 * between two sizes, so it contributes nothing to the comparison while occluding the
 * part that changed, and on this segmentation its surface is rough enough to read as
 * noise. Here the components sit alone against a neutral background with a millimetre
 * grid behind them, so a 4 mm step between sizes is actually visible.
 *
 * Two components are shown as they relate to each other — femoral above, tibial below,
 * separated by the insert thickness — rather than in their planned world positions,
 * because the question this panel answers is "how big is this construct", not "where
 * does it sit".
 *
 * The geometry is the planned size scaled to the selected size's catalogue ML/AP. The
 * outline is therefore right and the internal detail is stretched; the numbers in the
 * sizing panel remain the authority.
 */

import { Suspense, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { Bounds, Environment, OrbitControls, useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { toCreasedNormals } from "three/examples/jsm/utils/BufferGeometryUtils.js";

import { api } from "@/lib/api";
import type { MeshEntry, MeshManifest } from "@/lib/types";

const IMPLANT_SILVER = "#e2e8f0";
const IMPLANT_EDGE = "#93c5fd";

/** Anatomy is LPS; three.js wants Y up. */
const LPS_TO_VIEW = new THREE.Euler(-Math.PI / 2, 0, 0);

function Component({
  url,
  scale,
  ghost = false,
}: {
  url: string;
  scale: { ml: number; ap: number };
  /** Draws the planned size as a translucent reference behind the preview. */
  ghost?: boolean;
}) {
  const { scene } = useGLTF(url);

  const object = useMemo(() => {
    const clone = scene.clone(true);

    const material = ghost
      ? new THREE.MeshBasicMaterial({
          color: new THREE.Color("#64748b"),
          transparent: true,
          opacity: 0.16,
          side: THREE.BackSide,
          depthWrite: false,
        })
      : new THREE.MeshPhysicalMaterial({
          color: new THREE.Color(IMPLANT_SILVER),
          metalness: 1.0,
          roughness: 0.14,
          clearcoat: 0.8,
          clearcoatRoughness: 0.08,
          reflectivity: 1.0,
          envMapIntensity: 2.1,
        });

    clone.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (!mesh.isMesh) return;
      let geometry = mesh.geometry as THREE.BufferGeometry;
      if (!geometry.attributes.normal) geometry.computeVertexNormals();
      geometry = toCreasedNormals(geometry, (40 * Math.PI) / 180);
      mesh.geometry = geometry;
      mesh.material = material;

      if (!ghost) {
        const edges = new THREE.LineSegments(
          new THREE.EdgesGeometry(geometry, 48),
          new THREE.LineBasicMaterial({
            color: IMPLANT_EDGE,
            transparent: true,
            opacity: 0.85,
            toneMapped: false,
          }),
        );
        mesh.add(edges);
      }
    });

    // Centre the component on the origin, then scale. Centring first is what lets two
    // different sizes be compared concentrically instead of drifting apart.
    const box = new THREE.Box3().setFromObject(clone);
    const centre = box.getCenter(new THREE.Vector3());
    clone.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (!mesh.isMesh) return;
      const geometry = mesh.geometry as THREE.BufferGeometry;
      geometry.translate(-centre.x, -centre.y, -centre.z);
      if (!ghost) geometry.scale(scale.ml, scale.ap, 1);
      geometry.computeBoundingSphere();
    });

    return clone;
  }, [scene, scale.ml, scale.ap, ghost]);

  return <primitive object={object} />;
}

export default function ImplantPreview3D({
  caseId,
  manifest,
  scale,
  sizeLabel,
  isPlanned,
  showGhost = true,
}: {
  caseId: string;
  manifest: MeshManifest | null;
  scale: { femoral: { ml: number; ap: number }; tibial: { ml: number; ap: number } };
  sizeLabel: string;
  isPlanned: boolean;
  showGhost?: boolean;
}) {
  const components = useMemo<MeshEntry[]>(
    () => (manifest?.meshes ?? []).filter((m) => m.category === "component"),
    [manifest],
  );

  if (components.length === 0) {
    return (
      <div className="card">
        <div className="card-head">
          <span className="card-title">Implant preview</span>
        </div>
        <div className="card-body">
          <p className="text-[12px] text-slate-500">
            No component geometry was exported for this case.
          </p>
        </div>
      </div>
    );
  }

  const scaleFor = (entry: MeshEntry) =>
    /tibia/.test(entry.structure) ? scale.tibial : scale.femoral;

  return (
    <div className="card">
      <div className="card-head">
        <span className="card-title">Implant preview</span>
        <span
          className={`chip ${
            isPlanned
              ? "border-emerald-600 bg-emerald-500/15 text-emerald-300"
              : "border-sky-600 bg-sky-500/15 text-sky-300"
          }`}
        >
          size {sizeLabel}
          {isPlanned ? " · planned" : " · preview"}
        </span>
      </div>

      <div className="card-body !px-0 !py-0">
        <div className="relative h-56 w-full overflow-hidden rounded-b-lg bg-slate-950">
          <Canvas
            camera={{ position: [110, 70, 110], fov: 32, near: 1, far: 2000 }}
            dpr={[1, 2]}
            gl={{ antialias: true, preserveDrawingBuffer: true }}
          >
            <color attach="background" args={["#020617"]} />
            <ambientLight intensity={0.55} />
            <directionalLight position={[120, 140, 90]} intensity={1.5} />
            <directionalLight position={[-110, -50, -90]} intensity={0.45} />
            <Suspense fallback={null}>
              <Environment preset="studio" />
            </Suspense>

            <group rotation={LPS_TO_VIEW}>
              <Suspense fallback={null}>
                <Bounds fit clip observe margin={1.35}>
                  <group>
                    {components.map((entry) => (
                      <group key={entry.path}>
                        {/* Planned size behind, as a translucent reference. */}
                        {showGhost && !isPlanned && (
                          <Component
                            url={api.meshUrl(caseId, entry.path)}
                            scale={{ ml: 1, ap: 1 }}
                            ghost
                          />
                        )}
                        <Component
                          url={api.meshUrl(caseId, entry.path)}
                          scale={scaleFor(entry)}
                        />
                      </group>
                    ))}
                  </group>
                </Bounds>
              </Suspense>
            </group>

            <OrbitControls
              enablePan={false}
              enableDamping
              dampingFactor={0.1}
              minDistance={40}
              maxDistance={400}
            />
          </Canvas>

          {showGhost && !isPlanned && (
            <div className="pointer-events-none absolute bottom-2 left-2 rounded bg-slate-950/80 px-2 py-1 text-[10px] leading-snug text-slate-400 backdrop-blur">
              <span className="text-slate-500">ghost</span> = planned size ·{" "}
              <span className="text-sky-300">solid</span> = size {sizeLabel}
            </div>
          )}
          <div className="pointer-events-none absolute right-2 top-2 rounded bg-slate-950/80 px-2 py-1 text-[10px] text-slate-500 backdrop-blur">
            drag to rotate
          </div>
        </div>
      </div>
    </div>
  );
}
