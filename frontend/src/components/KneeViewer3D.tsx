/**
 * Interactive 3D Knee Viewer using React Three Fiber.
 * Loads GLB meshes from the backend and renders with orbit controls.
 */
import { Suspense, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, useGLTF, Html, Center } from '@react-three/drei'
import * as THREE from 'three'

interface KneeViewer3DProps {
  meshUrl: string | null;
  isLoading?: boolean;
}

function KneeModel({ url }: { url: string }) {
  const { scene } = useGLTF(url)
  const ref = useRef<THREE.Group>(null)

  // Center and scale the model
  const box = new THREE.Box3().setFromObject(scene)
  const center = box.getCenter(new THREE.Vector3())
  const size = box.getSize(new THREE.Vector3())
  const maxDim = Math.max(size.x, size.y, size.z)
  const scale = 4 / maxDim

  return (
    <group ref={ref} scale={scale} position={[-center.x * scale, -center.y * scale, -center.z * scale]}>
      <primitive object={scene} />
    </group>
  )
}

function LoadingIndicator() {
  const ref = useRef<THREE.Mesh>(null)
  useFrame((_, delta) => {
    if (ref.current) {
      ref.current.rotation.y += delta * 2
    }
  })

  return (
    <mesh ref={ref}>
      <torusGeometry args={[1, 0.3, 8, 30]} />
      <meshStandardMaterial color="#38bdf8" wireframe />
    </mesh>
  )
}

function Placeholder() {
  return (
    <Html center>
      <div className="text-center text-gray-400">
        <div className="text-4xl mb-2">🦴</div>
        <div className="text-sm">Select a case to view 3D reconstruction</div>
      </div>
    </Html>
  )
}

export default function KneeViewer3D({ meshUrl, isLoading }: KneeViewer3DProps) {
  const [_error] = useState<string | null>(null)

  return (
    <div className="w-full h-full min-h-[500px] rounded-xl overflow-hidden bg-gray-900 border border-gray-800">
      <Canvas
        camera={{ position: [0, 0, 8], fov: 50 }}
        gl={{ antialias: true, alpha: true }}
      >
        <color attach="background" args={['#0f172a']} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[5, 5, 5]} intensity={0.8} />
        <directionalLight position={[-5, -5, -3]} intensity={0.3} />
        <pointLight position={[0, 5, 0]} intensity={0.5} />

        <Suspense fallback={<LoadingIndicator />}>
          {meshUrl && !isLoading && !_error ? (
            <Center>
              <KneeModel url={meshUrl} />
            </Center>
          ) : isLoading ? (
            <LoadingIndicator />
          ) : (
            <Placeholder />
          )}
        </Suspense>

        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          rotateSpeed={0.8}
          zoomSpeed={1.2}
          minDistance={2}
          maxDistance={20}
        />
        <gridHelper args={[10, 10, '#1e293b', '#1e293b']} position={[0, -3, 0]} />
      </Canvas>

      {/* Controls overlay */}
      <div className="absolute bottom-3 left-3 text-xs text-gray-500">
        Drag to rotate • Scroll to zoom • Right-click to pan
      </div>
    </div>
  )
}
