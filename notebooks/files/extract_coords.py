"""
One-off utility: extract exact field box coordinates from the template PDF.

Run this whenever the template PDF changes layout, to regenerate the
coordinates that belong in template_config.py -- rather than hand-measuring
pixels on a rendered image, which is error-prone and hard to keep in sync.

USAGE
    python3 extract_coords.py path/to/template.pdf

OUTPUT
    Prints:
      1. All text spans with their bounding boxes (to help you match labels
         to fields if you add/rename/reorder fields).
      2. All filled rectangles with a light-gray fill (0.9569, 0.9569,
         0.9569) -- these are the input value-boxes on this specific
         template. If you re-export from a different design tool, the
         exact fill color may differ; adjust GRAY_FILL below to match.
"""

from __future__ import annotations

import sys

import fitz  # PyMuPDF

GRAY_FILL = (0.9569, 0.9569, 0.9569)
FILL_TOLERANCE = 0.01


def extract(pdf_path: str) -> None:
    doc = fitz.open(pdf_path)
    page = doc[0]

    print(f"Page size (pt): {page.rect}\n")

    print("=== TEXT SPANS ===")
    for block in page.get_text("dict")["blocks"]:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                bbox = tuple(round(v, 1) for v in span["bbox"])
                print(f"  text={span['text']!r:30} bbox={bbox}")

    print("\n=== INPUT BOXES (gray-filled rectangles) ===")
    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        if fill is None:
            continue
        if all(abs(fill[i] - GRAY_FILL[i]) < FILL_TOLERANCE for i in range(3)):
            rect = tuple(round(v, 1) for v in drawing["rect"])
            print(f"  box={rect}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 extract_coords.py path/to/template.pdf")
        sys.exit(1)
    extract(sys.argv[1])
