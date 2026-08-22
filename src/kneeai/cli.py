"""``kneeai`` command line interface.

The entry point declared by ``pyproject.toml`` (``kneeai = "kneeai.cli:app"``)
and the commands the Makefile calls.

``kneeai demo run`` is the spine of the platform: it drives the whole
deterministic pipeline on the synthetic phantom knee —

    phantom -> post-process -> reconstruct -> landmarks -> measure
            -> implant match -> plan -> simulate

— and writes a contract-compliant case directory. Every manifest it emits is
validated through :func:`kneeai.ml.contract.validate_result_payload` before it
lands, so a breach of the ML contract fails the command rather than reaching the
API or the viewer.

This module is presentation only. Backend selection lives in
:mod:`kneeai.api.adapter` and stage ordering in :mod:`kneeai.api.pipeline`, so the
CLI, the HTTP API and the worker all produce identical case directories.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from kneeai.api.adapter import AdapterConfig, MLInferenceAdapter
from kneeai.api.pipeline import DISCLAIMER, CaseArtifacts, PipelineOptions, run_case_pipeline
from kneeai.config import get_settings
from kneeai.core.anatomy import Side
from kneeai.core.errors import KneeAIError
from kneeai.logging import case_context, configure_logging, get_logger
from kneeai.ml.contract import ArtifactSource
from kneeai.version import PLATFORM_VERSION

app = typer.Typer(
    name="kneeai",
    help="Knee Intelligence Platform — patient-specific planning toolchain.",
    no_args_is_help=True,
    add_completion=False,
)
demo_app = typer.Typer(no_args_is_help=True, help="End-to-end demo on synthetic data.")
db_app = typer.Typer(no_args_is_help=True, help="Database lifecycle.")
implants_app = typer.Typer(no_args_is_help=True, help="Implant catalog.")
datasets_app = typer.Typer(no_args_is_help=True, help="Dataset registry (read-only).")
app.add_typer(demo_app, name="demo")
app.add_typer(db_app, name="db")
app.add_typer(implants_app, name="implants")
app.add_typer(datasets_app, name="datasets")

console = Console()
log = get_logger("cli")


# ---------------------------------------------------------------------------
# Option parsing
# ---------------------------------------------------------------------------


def _parse_side(value: str) -> Side:
    try:
        return Side(value.strip().lower())
    except ValueError as exc:
        raise typer.BadParameter(
            f"side must be one of {[s.value for s in Side]}, got {value!r}"
        ) from exc


def _parse_backend(value: str) -> ArtifactSource:
    try:
        return ArtifactSource(value.strip().lower())
    except ValueError as exc:
        valid = [s.value for s in ArtifactSource]
        raise typer.BadParameter(f"unknown source {value!r}; choose from {valid}") from exc


def _stage_printer(quiet: bool) -> Any:
    if quiet:
        return None

    def announce(label: str) -> None:
        console.print(f"[cyan]->[/cyan] {label}")

    return announce


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_summary(artifacts: CaseArtifacts) -> None:
    from kneeai.api.contract import read_result_json

    view = read_result_json(artifacts.case_dir)
    console.rule(f"[bold]{view.badge}[/bold]  case {artifacts.case_id}")
    console.print(view.summary())

    if artifacts.mesh_manifest is not None:
        table = Table(title="Surfaces", show_edge=False, header_style="bold")
        for column in ("structure", "category", "faces", "volume mm3", "watertight"):
            table.add_column(column)
        for mesh in artifacts.mesh_manifest.meshes:
            table.add_row(
                mesh.structure,
                mesh.category,
                f"{mesh.n_faces:,}",
                f"{mesh.volume_mm3:,.1f}",
                "yes" if mesh.is_watertight else "NO",
            )
        console.print(table)
        console.print(
            f"total {artifacts.mesh_manifest.total_faces:,} faces, "
            f"compression: {artifacts.mesh_manifest.compression_used}"
        )

    if artifacts.measurements is not None:
        console.print()
        console.print(artifacts.measurements.summary())

    if artifacts.truth and artifacts.measurements is not None:
        _print_truth_table(artifacts)

    if artifacts.fits:
        best = artifacts.fits[0]
        console.print()
        console.rule("Implant selection")
        console.print("\n".join(best.rationale()))
        if len(artifacts.fits) > 1:
            runner_up = artifacts.fits[1]
            console.print(
                f"\nrunner-up: {runner_up.configuration.label} scored "
                f"{runner_up.score:.1f} vs {best.score:.1f} "
                f"({runner_up.score - best.score:+.1f})"
            )
            for violation in runner_up.violations:
                console.print(f"  - {violation}")

    if artifacts.plan is not None:
        console.print()
        console.rule("Plan")
        console.print(artifacts.plan.summary())
    if artifacts.simulation is not None:
        console.print()
        console.rule("Simulation")
        console.print(artifacts.simulation.summary())
    if artifacts.kinematics is not None:
        console.print()
        console.print(artifacts.kinematics.summary())

    console.print()
    console.rule("Artifacts")
    console.print(f"case directory : {artifacts.case_dir}")
    for name, path in sorted(artifacts.artifact_paths().items()):
        console.print(f"  {name:<14} {path.name}")
    if artifacts.timings_s:
        stages = "  ".join(f"{k} {v:.1f}s" for k, v in artifacts.timings_s.items())
        console.print(f"timings ({artifacts.total_seconds:.1f}s total): {stages}")
    if artifacts.warnings:
        console.print()
        console.print("[yellow]warnings:[/yellow]")
        for warning in artifacts.warnings:
            console.print(f"  - {warning}")
    console.print()
    console.print(f"[bold red]{DISCLAIMER}[/bold red]")


def _print_truth_table(artifacts: CaseArtifacts) -> None:
    """Compare measurements against the phantom's analytic truth.

    Shown only for phantom-backed runs, where a closed-form answer exists. It is
    a pipeline self-check, not a clinical accuracy claim.
    """
    from kneeai.measure import MeasurementName as M

    checks: list[tuple[Any, str, float]] = [
        (M.FEMORAL_ML_WIDTH, "femoral_ml_mm", 2.0),
        (M.FEMORAL_AP_SIZE, "femoral_ap_mm", 2.0),
        (M.TIBIAL_ML_WIDTH, "tibial_ml_mm", 2.5),
        (M.TIBIAL_AP_DEPTH, "tibial_ap_mm", 2.5),
        (M.POSTERIOR_TIBIAL_SLOPE, "posterior_tibial_slope_deg", 1.5),
        (M.CONDYLAR_TWIST_ANGLE, "condylar_twist_deg", 2.0),
        (M.HKA_ANGLE, "hka_varus_deg", 1.5),
        (M.FEMORAL_ASPECT_RATIO, "aspect_ratio_femur", 0.06),
        (M.FEMORAL_CARTILAGE_MEAN_THICKNESS, "femoral_cartilage_thickness_mm", 0.6),
    ]
    table = Table(
        title="Measured vs analytic phantom truth (pipeline self-check)",
        show_edge=False,
        header_style="bold",
    )
    for column in ("measurement", "measured", "truth", "error", "tol", ""):
        table.add_column(column)
    outside = 0
    for name, key, tolerance in checks:
        measurement = artifacts.measurements.get(name)
        if measurement is None or key not in artifacts.truth:
            continue
        truth = artifacts.truth[key]
        error = abs(measurement.value - truth)
        ok = error <= tolerance
        outside += 0 if ok else 1
        table.add_row(
            name.value,
            f"{measurement.value:9.3f}",
            f"{truth:9.3f}",
            f"{error:6.3f}",
            f"{tolerance:5.2f}",
            "ok" if ok else "OUTSIDE",
        )
    console.print()
    console.print(table)
    style = "green" if outside == 0 else "red"
    console.print(f"[{style}]{outside} check(s) outside tolerance[/{style}]")


# ---------------------------------------------------------------------------
# Pipeline commands
# ---------------------------------------------------------------------------


def _execute(
    config: AdapterConfig,
    *,
    case_id: str,
    out_root: Path,
    options: PipelineOptions,
) -> CaseArtifacts:
    adapter = MLInferenceAdapter(config)
    with case_context(case_id, actor="cli"):
        outcome = adapter.segment()
        if outcome.degraded:
            console.print(
                f"[yellow]backend degraded:[/yellow] {outcome.degradation_reason}"
            )
            console.print(
                f"[yellow]output is {outcome.source.badge}, not "
                f"{(outcome.requested_backend or config.backend).badge}[/yellow]"
            )
        return run_case_pipeline(
            outcome, case_id=case_id, out_root=out_root, options=options
        )


@demo_app.command("run")
def demo_run(
    out: Path = typer.Option(
        Path("artifacts"),
        "--out",
        "-o",
        help="Artifact root. The case is written to <out>/cases/<case-id>/.",
    ),
    case_id: str = typer.Option("demo-phantom-right", "--case-id", help="Case id."),
    side: str = typer.Option("right", "--side", help="Operative side: left or right."),
    spacing: float = typer.Option(0.8, "--spacing", help="Phantom voxel size in mm."),
    cartilage_loss: float = typer.Option(
        0.0,
        "--cartilage-loss",
        min=0.0,
        max=1.0,
        help="Medial cartilage loss (0 healthy, 1 full thickness).",
    ),
    strategy: str = typer.Option(
        "restricted_kinematic", "--strategy", help="Alignment strategy."
    ),
    optimize: bool = typer.Option(True, "--optimize/--no-optimize"),
    compress: bool = typer.Option(
        True, "--compress/--no-compress", help="Draco-compress the GLB meshes."
    ),
    flexion_step: int = typer.Option(20, "--flexion-step", help="Flexion step (deg)."),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Run the full pipeline on the synthetic phantom knee (no patient data)."""
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)

    console.print(f"[bold]kneeai {PLATFORM_VERSION}[/bold] — phantom demo")
    console.print(f"[red]{DISCLAIMER}[/red]\n")

    config = AdapterConfig(
        backend=ArtifactSource.PHANTOM,
        side=_parse_side(side),
        spacing_mm=spacing,
        cartilage_loss=cartilage_loss,
    )
    options = PipelineOptions(
        strategy=strategy,
        optimize=optimize,
        flexion_step_deg=flexion_step,
        compress_meshes=compress,
        on_stage=_stage_printer(quiet),
    )
    try:
        artifacts = _execute(
            config, case_id=case_id, out_root=out.resolve(), options=options
        )
    except KneeAIError as exc:
        console.print(f"[bold red]failed:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    _print_summary(artifacts)


@app.command("infer")
def infer(
    case_id: str = typer.Option(..., "--case-id", help="Case id."),
    source: str = typer.Option(
        "phantom",
        "--source",
        help="Backend: ai_prediction | ground_truth | phantom | mock.",
    ),
    image: Optional[Path] = typer.Option(
        None, "--image", help="Input image (required for ai_prediction)."
    ),
    mask: Optional[Path] = typer.Option(
        None, "--mask", help="Reference mask (required for ground_truth)."
    ),
    out: Optional[Path] = typer.Option(None, "--out", help="Artifact root."),
    side: str = typer.Option("right", "--side"),
    spacing: float = typer.Option(0.8, "--spacing", help="Phantom/mock voxel size (mm)."),
    model: str = typer.Option("seg_oaizib", "--model", help="Registry reference."),
    strategy: str = typer.Option("restricted_kinematic", "--strategy"),
    optimize: bool = typer.Option(True, "--optimize/--no-optimize"),
    compress: bool = typer.Option(True, "--compress/--no-compress"),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail instead of degrading to phantom when the model is unavailable.",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Segment and plan one case through the selected backend.

    ``ai_prediction`` degrades to the phantom backend — with ``source`` relabelled
    to ``phantom`` — when no model card or checkpoint is available. Pass
    ``--strict`` to make that a failure instead.
    """
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)

    config = AdapterConfig(
        backend=_parse_backend(source),
        side=_parse_side(side),
        spacing_mm=spacing,
        image_path=image,
        mask_path=mask,
        model_reference=model,
        allow_degradation=not strict,
    )
    options = PipelineOptions(
        strategy=strategy,
        optimize=optimize,
        compress_meshes=compress,
        on_stage=_stage_printer(quiet),
    )
    try:
        artifacts = _execute(
            config,
            case_id=case_id,
            out_root=(out or settings.artifact_root).resolve(),
            options=options,
        )
    except KneeAIError as exc:
        console.print(f"[bold red]failed:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    _print_summary(artifacts)


@app.command("validate")
def validate_case(
    case_dir: Path = typer.Argument(..., help="Case directory containing result.json."),
) -> None:
    """Validate a case directory against the ML contract."""
    from kneeai.ml.contract import validate_result_json

    target = case_dir / "result.json" if case_dir.is_dir() else case_dir
    report = validate_result_json(target)
    console.print(report.summary())
    if not report.valid:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Datasets (read-only)
# ---------------------------------------------------------------------------


@datasets_app.command("validate")
def datasets_validate(
    registry: Optional[Path] = typer.Option(
        None, "--registry", help="Path to dataset_registry.csv."
    ),
) -> None:
    """Validate the dataset registry. Strictly read-only.

    ``data/`` belongs to the ML side. This command never creates or modifies
    anything under it, including the directory itself.
    """
    settings = get_settings()
    path = registry or (settings.data_root / "registry" / "dataset_registry.csv")

    if not path.exists():
        console.print(f"[bold red]dataset registry not found:[/bold red] {path}")
        console.print(
            "This file is owned by the ML/data side and is not created here. "
            "Expected a CSV at data/registry/dataset_registry.csv describing each "
            "dataset (id, name, licence, access, patient count)."
        )
        raise typer.Exit(1)

    try:
        rows = list(csv.DictReader(path.read_text().splitlines()))
    except (OSError, csv.Error) as exc:
        console.print(f"[bold red]could not read {path}:[/bold red] {exc}")
        raise typer.Exit(1) from exc

    if not rows:
        console.print(f"[bold red]{path} contains no data rows.[/bold red]")
        raise typer.Exit(1)

    fields = [f for f in (rows[0].keys()) if f]
    id_field = next(
        (f for f in fields if f.strip().lower() in ("dataset_id", "id", "name")),
        fields[0] if fields else "",
    )
    problems: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=2):
        identifier = (row.get(id_field) or "").strip()
        if not identifier:
            problems.append(f"line {index}: empty {id_field!r}")
            continue
        if identifier in seen:
            problems.append(f"line {index}: duplicate identifier {identifier!r}")
        seen.add(identifier)

    # Show a readable subset. The registry carries ~23 columns (citations,
    # checksums, URLs); rendering all of them collapses every column to an
    # ellipsis and tells the reader nothing.
    preferred = [
        "dataset_id",
        "dataset_name",
        "role",
        "access_status",
        "license",
        "commercial_use",
        "patient_count",
        "modality",
    ]
    columns = [c for c in preferred if c in fields] or fields[:6]
    table = Table(title=path.name, show_edge=False, header_style="bold")
    for column in columns:
        table.add_column(column, overflow="fold" if column.endswith("name") else "ellipsis")
    for row in rows:
        table.add_row(*[str(row.get(column, "") or "") for column in columns])
    console.print(table)
    console.print(
        f"{len(rows)} dataset(s), {len(seen)} unique identifier(s), "
        f"{len(fields)} columns ({len(columns)} shown)"
    )

    if problems:
        console.print("[bold red]problems:[/bold red]")
        for problem in problems:
            console.print(f"  - {problem}")
        raise typer.Exit(1)
    console.print("[green]registry is internally consistent[/green] (read-only check)")


# ---------------------------------------------------------------------------
# Database / catalog
# ---------------------------------------------------------------------------


@db_app.command("init")
def db_init(
    url: Optional[str] = typer.Option(None, "--url", help="Override the database URL."),
) -> None:
    """Create the database tables."""
    try:
        from kneeai.db import init_database
    except ImportError as exc:
        console.print(
            f"[bold red]the persistence layer is unavailable:[/bold red] {exc}"
        )
        console.print("Install the api extra:  pip install -e '.[api]'")
        raise typer.Exit(1) from exc

    settings = get_settings()
    target = url or settings.database_url
    created = init_database(target)
    console.print(f"[green]initialised[/green] {_redact_url(target)}")
    for name in created:
        console.print(f"  - {name}")


@implants_app.command("seed")
def implants_seed(
    catalog_id: Optional[str] = typer.Option(None, "--catalog", help="Catalog id."),
    url: Optional[str] = typer.Option(None, "--url", help="Override the database URL."),
) -> None:
    """Load the reference (non-clinical) implant catalog into the database."""
    from kneeai.implants import load_catalog

    settings = get_settings()
    catalog = load_catalog(catalog_id or settings.implant_catalog)
    console.print(json.dumps(catalog.summary(), indent=2))
    console.print(
        "[yellow]This catalog is synthetic reference geometry for development. It "
        "must never be used to plan a real operation (PRD §21).[/yellow]"
    )

    try:
        from kneeai.db import seed_implant_catalog
    except ImportError as exc:
        console.print(
            f"[bold red]the persistence layer is unavailable:[/bold red] {exc}"
        )
        console.print("Install the api extra:  pip install -e '.[api]'")
        raise typer.Exit(1) from exc

    counts = seed_implant_catalog(catalog, url or settings.database_url)
    for label, count in sorted(counts.items()):
        console.print(f"  {label}: {count}")


def _redact_url(url: str) -> str:
    """Hide credentials before printing a database URL."""
    if "@" not in url or "//" not in url:
        return url
    scheme, _, rest = url.partition("//")
    _credentials, _, host = rest.rpartition("@")
    return f"{scheme}//<redacted>@{host}"


@app.command("version")
def version() -> None:
    """Print version information."""
    from kneeai.ml.contract import CONTRACT_VERSION
    from kneeai.version import COMPONENT_VERSIONS, SCHEMA_VERSION

    console.print(f"platform  {PLATFORM_VERSION}")
    console.print(f"schema    {SCHEMA_VERSION}")
    console.print(f"contract  v{CONTRACT_VERSION}")
    for name, component in sorted(COMPONENT_VERSIONS.items()):
        console.print(f"  {name:<24} {component}")


if __name__ == "__main__":  # pragma: no cover
    app()
