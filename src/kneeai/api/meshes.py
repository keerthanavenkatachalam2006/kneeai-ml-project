"""GLB export for the 3D viewer, with optional Draco compression.

The contract calls for ``meshes/*.glb`` in millimetres. Two things are handled
carefully here:

**Units and frame are preserved, not converted.** Vertices are written exactly
as the reconstruction produced them: millimetres, in the frame the
:class:`~kneeai.recon.mesh.AnatomicalMesh` declares. glTF is nominally a
Y-up/metre format, and quietly rescaling or rotating geometry on the way out
would mean an on-mesh measurement in the viewer no longer matched the
measurement engine. The manifest states the units and frame instead, and the
viewer applies a display rotation it owns.

**Draco is reported, never assumed.** trimesh cannot write Draco-compressed
glTF, so compression is applied by an external tool when one is available. If
none is, the GLB is written uncompressed and the manifest says
``"compression": "none"``. Claiming Draco when it was not applied would be a
fabricated property of a delivered artefact.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from kneeai.core.anatomy import Structure
from kneeai.core.errors import ValidationError
from kneeai.core.provenance import hash_file
from kneeai.logging import get_logger
from kneeai.recon.mesh import AnatomicalMesh, MeshSet

__all__ = [
    "MeshExport",
    "MeshManifest",
    "MESH_MANIFEST_FILENAME",
    "draco_tool",
    "export_mesh",
    "export_meshes",
]

log = get_logger("api.meshes")

MESH_MANIFEST_FILENAME = "manifest.json"

#: Where to look for a Draco-capable glTF processor, in order.
#:
#: ``@gltf-transform/cli`` is a devDependency of ``apps/web``, so a developer who
#: has run ``make web-install`` gets Draco compression with no extra step.
_WEB_NODE_BIN = Path("apps/web/node_modules/.bin")


@dataclass(frozen=True)
class MeshExport:
    """One exported mesh file."""

    structure: str
    category: str
    path: Path
    frame: str
    n_vertices: int
    n_faces: int
    volume_mm3: float
    is_watertight: bool
    bounds_mm: tuple[tuple[float, float, float], tuple[float, float, float]]
    compression: str
    bytes_size: int
    content_hash: str
    units: str = "mm"

    def to_dict(self, *, relative_to: Path | None = None) -> dict[str, Any]:
        path = self.path
        if relative_to is not None:
            try:
                path = self.path.relative_to(relative_to)
            except ValueError:
                pass
        return {
            "structure": self.structure,
            "category": self.category,
            "path": str(path).replace(os.sep, "/"),
            "frame": self.frame,
            "units": self.units,
            "n_vertices": self.n_vertices,
            "n_faces": self.n_faces,
            "volume_mm3": round(self.volume_mm3, 2),
            "is_watertight": self.is_watertight,
            "bounds_mm": [list(self.bounds_mm[0]), list(self.bounds_mm[1])],
            "compression": self.compression,
            "bytes_size": self.bytes_size,
            "content_hash": self.content_hash,
        }


@dataclass
class MeshManifest:
    """Everything the viewer needs to load a case's geometry."""

    case_id: str
    meshes: list[MeshExport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_faces(self) -> int:
        return sum(m.n_faces for m in self.meshes)

    @property
    def compression_used(self) -> str:
        kinds = {m.compression for m in self.meshes}
        if not kinds:
            return "none"
        return sorted(kinds)[0] if len(kinds) == 1 else "mixed"

    def by_category(self, category: str) -> list[MeshExport]:
        return [m for m in self.meshes if m.category == category]

    def to_dict(self, *, relative_to: Path | None = None) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "units": "mm",
            "compression": self.compression_used,
            "total_faces": self.total_faces,
            "meshes": [m.to_dict(relative_to=relative_to) for m in self.meshes],
            "warnings": list(self.warnings),
        }

    def write(self, directory: str | Path) -> Path:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / MESH_MANIFEST_FILENAME
        target.write_text(json.dumps(self.to_dict(relative_to=out_dir), indent=2) + "\n")
        return target


# ---------------------------------------------------------------------------
# Draco
# ---------------------------------------------------------------------------


def draco_tool() -> tuple[str, list[str]] | None:
    """Locate a Draco-capable glTF processor.

    Returns:
        ``(name, argv_prefix)`` or ``None`` when no tool is installed. Callers
        must treat ``None`` as "write uncompressed and say so".
    """
    override = os.environ.get("KNEEAI_GLTF_TRANSFORM")
    if override and Path(override).exists():
        return ("gltf-transform", [override])

    local = _WEB_NODE_BIN / "gltf-transform"
    if local.exists():
        return ("gltf-transform", [str(local)])

    found = shutil.which("gltf-transform")
    if found:
        return ("gltf-transform", [found])

    packed = shutil.which("gltfpack")
    if packed:
        return ("gltfpack", [packed])

    return None


def _compress_draco(path: Path) -> str:
    """Compress ``path`` in place. Returns the compression actually applied."""
    tool = draco_tool()
    if tool is None:
        return "none"
    name, argv = tool
    temp = path.with_suffix(".draco.glb")
    if name == "gltf-transform":
        command = [*argv, "draco", str(path), str(temp)]
    else:  # gltfpack
        command = [*argv, "-i", str(path), "-o", str(temp), "-cc"]
    try:
        completed = subprocess.run(  # noqa: S603 - argv list, no shell
            command,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning(
            "Draco compression tool failed to run; writing uncompressed GLB",
            fields={"tool": name, "error": str(exc)},
        )
        temp.unlink(missing_ok=True)
        return "none"

    if completed.returncode != 0 or not temp.exists() or temp.stat().st_size == 0:
        log.warning(
            "Draco compression failed; keeping the uncompressed GLB",
            fields={
                "tool": name,
                "returncode": completed.returncode,
                "stderr": (completed.stderr or "")[-400:],
            },
        )
        temp.unlink(missing_ok=True)
        return "none"

    temp.replace(path)
    return "draco"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _sanitise(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")
    if not safe:
        raise ValidationError(f"cannot derive a safe filename from {name!r}")
    return safe


def export_mesh(
    mesh: AnatomicalMesh,
    directory: str | Path,
    *,
    name: str | None = None,
    category: str = "anatomy",
    compress: bool = True,
) -> MeshExport:
    """Write one mesh as GLB in millimetres, optionally Draco-compressed."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_sanitise(name or mesh.structure.value)}.glb"
    target = out_dir / filename

    # Export through trimesh directly rather than AnatomicalMesh.export(), which
    # also writes a .json sidecar per mesh; the manifest carries that information
    # once for the whole case instead.
    exported = mesh.mesh.export(file_type="glb")
    target.write_bytes(exported)

    compression = _compress_draco(target) if compress else "none"
    bounds = mesh.bounds
    return MeshExport(
        structure=mesh.structure.value,
        category=category,
        path=target,
        frame=mesh.frame.value,
        n_vertices=mesh.n_vertices,
        n_faces=mesh.n_faces,
        volume_mm3=mesh.volume_mm3,
        is_watertight=mesh.is_watertight,
        bounds_mm=(
            tuple(round(float(v), 4) for v in bounds[0]),  # type: ignore[arg-type]
            tuple(round(float(v), 4) for v in bounds[1]),  # type: ignore[arg-type]
        ),
        compression=compression,
        bytes_size=target.stat().st_size,
        content_hash=hash_file(target),
    )


def export_meshes(
    case_id: str,
    directory: str | Path,
    *,
    anatomy: MeshSet | Mapping[Structure, AnatomicalMesh] | None = None,
    resected: Mapping[Structure, AnatomicalMesh] | None = None,
    components: Mapping[str, AnatomicalMesh] | None = None,
    extra: Iterable[tuple[str, str, AnatomicalMesh]] = (),
    compress: bool = True,
) -> MeshManifest:
    """Export a case's geometry and write ``meshes/manifest.json``.

    Categories are kept distinct because the viewer treats them differently:
    ``anatomy`` is the intact bone, ``resected`` is what remains after the
    planned cuts, and ``component`` is the implant seated on that resection.
    """
    out_dir = Path(directory)
    manifest = MeshManifest(case_id=case_id)

    for structure, mesh in sorted(
        dict(anatomy or {}).items(), key=lambda kv: kv[0].value
    ):
        manifest.meshes.append(
            export_mesh(mesh, out_dir, category="anatomy", compress=compress)
        )

    for structure, mesh in sorted(
        dict(resected or {}).items(), key=lambda kv: kv[0].value
    ):
        manifest.meshes.append(
            export_mesh(
                mesh,
                out_dir,
                name=f"{structure.value}_resected",
                category="resected",
                compress=compress,
            )
        )

    for label, mesh in sorted(dict(components or {}).items()):
        manifest.meshes.append(
            export_mesh(
                mesh,
                out_dir,
                name=f"implant_{label}",
                category="component",
                compress=compress,
            )
        )

    for name, category, mesh in extra:
        manifest.meshes.append(
            export_mesh(mesh, out_dir, name=name, category=category, compress=compress)
        )

    if compress and manifest.compression_used == "none" and manifest.meshes:
        manifest.warnings.append(
            "GLB meshes are NOT Draco-compressed: no glTF processor was found. "
            "Install the web devDependencies (make web-install) or set "
            "KNEEAI_GLTF_TRANSFORM to a gltf-transform binary. The files are "
            "valid, uncompressed glTF and load correctly; only transfer size is "
            "affected."
        )

    non_watertight = [m.structure for m in manifest.meshes if not m.is_watertight]
    if non_watertight:
        manifest.warnings.append(
            "not watertight (volume undefined, unusable for boolean cuts): "
            + ", ".join(sorted(non_watertight))
        )

    manifest.write(out_dir)
    log.info(
        "meshes exported",
        fields={
            "case_id": case_id,
            "files": len(manifest.meshes),
            "total_faces": manifest.total_faces,
            "compression": manifest.compression_used,
        },
    )
    return manifest


def verify_roundtrip(export: MeshExport, *, tolerance_mm: float = 1e-3) -> None:
    """Re-read an exported GLB and confirm its geometry survived unchanged.

    Guards against a silent unit or axis conversion in the glTF writer: a viewer
    showing millimetres that are really metres, or a mesh rotated into Y-up,
    would produce on-screen measurements that disagree with the measurement
    engine.

    Raises:
        ValidationError: if bounds moved by more than ``tolerance_mm``.
    """
    import trimesh

    loaded = trimesh.load(str(export.path), process=False, force="mesh")
    actual = np.asarray(loaded.bounds, dtype=np.float64)
    expected = np.asarray(export.bounds_mm, dtype=np.float64)
    if not np.allclose(actual, expected, atol=tolerance_mm):
        raise ValidationError(
            f"{export.path.name} changed on export: bounds were "
            f"{expected.tolist()} mm, re-read as {actual.tolist()} mm. The glTF "
            "writer applied a unit or axis conversion; on-mesh measurements would "
            "not match the measurement engine."
        )
