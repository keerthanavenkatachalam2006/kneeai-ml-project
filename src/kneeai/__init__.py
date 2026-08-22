"""Knee Intelligence Platform.

A patient-specific knee intelligence, surgical planning and (future) robotic
assistance platform. Converts a knee imaging study into an explainable,
surgeon-editable 3D surgical plan.

Layers, in dependency order::

    core         domain vocabulary, coordinate frames, provenance
    imaging      DICOM ingest, validation, anonymization, preprocessing
    ml           segmentation / landmark / pathology models (MONAI, Colab)
    recon        voxel masks -> validated 3D surfaces
    landmarks    anatomical landmark detection
    measure      clinical measurement engine
    implants     implant catalog + deterministic geometric matching
    planning     alignment, resections, plan optimisation
    simulation   virtual cuts, collision, kinematics
    robotics     independent safety validation (no direct AI->actuator path)
    llm          explanation layer, guardrailed
    reporting    clinical report generation
    db / api     persistence and HTTP surface

Nothing in ``core`` .. ``planning`` requires a GPU. Model training runs on Google
Colab; see ``notebooks/``.

This software is an engineering blueprint implementation, not a cleared medical
device. See ``docs/regulatory.md``.
"""

from __future__ import annotations

from kneeai.version import PLATFORM_VERSION as __version__

__all__ = ["__version__"]
