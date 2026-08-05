"""Compatibility wrapper for the v2 Nepali OCR pipeline.

This module exposes the newer 128x320, external-vocab pipeline under a
stable import name so launchers can use `from ocr_pipeline_v2 import
OCRPipeline` without depending on the dotted filename `ocr_pipeline_2.0.py`.

The implementation still lives in `ocr_pipeline_2.0.py` so the existing
training-aligned code stays in one place.
"""

from __future__ import annotations

import runpy
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LEGACY_PATH = _HERE / "ocr_pipeline_2.0.py"

if not _LEGACY_PATH.exists():
    raise FileNotFoundError(f"Expected v2 pipeline at {_LEGACY_PATH}")

_legacy_globals = runpy.run_path(str(_LEGACY_PATH), run_name="ocr_pipeline_v2_legacy")

OCRPipeline = _legacy_globals["OCRPipeline"]
CRNNRecognizer = _legacy_globals["CRNNRecognizer"]
CRNNDevanagari = _legacy_globals["CRNNDevanagari"]
render_reference_template = _legacy_globals["render_reference_template"]
load_and_preprocess = _legacy_globals["load_and_preprocess"]
align_to_template = _legacy_globals["align_to_template"]
crop_field_rois = _legacy_globals["crop_field_rois"]
IMG_H = _legacy_globals["IMG_H"]
IMG_W = _legacy_globals["IMG_W"]

__all__ = [
    "OCRPipeline",
    "CRNNRecognizer",
    "CRNNDevanagari",
    "render_reference_template",
    "load_and_preprocess",
    "align_to_template",
    "crop_field_rois",
    "IMG_H",
    "IMG_W",
]


if __name__ == "__main__":
    runpy.run_path(str(_LEGACY_PATH), run_name="__main__")