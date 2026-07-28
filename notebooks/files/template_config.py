"""
Template configuration for the Nepali OCR pipeline.

COORDINATE SYSTEM
------------------
Origin: top-left corner of the page, (0, 0).
Units:  PDF points, matching the source file
        `Red_Minimalist_Membership_Form_A4.pdf` (page size 595.5 x 842.25 pt,
        i.e. standard A4 at 72pt/inch).
All coordinates below were extracted DIRECTLY from that PDF's vector content
(text spans + drawn rectangles) via PyMuPDF -- not measured by eye. This
means they are exact, not estimates.

If you regenerate the PDF (e.g. re-export from Canva/Figma/etc.), these
coordinates only stay valid if the layout doesn't shift. If you change the
template at all, re-run the extraction (see `extract_coords.py`) rather than
hand-editing these numbers.

WHY A "ZONE OF INTEREST" (ROI) SEPARATE FROM THE FULL BOX
------------------------------------------------------------
The drawn input boxes on the form are ~382pt wide (151.2 -> 533.8) -- sized
to look proportional on the printed form, not sized to actual handwritten
content. Feeding the CRNN a crop that's 90% blank space hurts recognition:
the model has to "find" the word in a sea of padding, and CTC alignment
degrades on mostly-empty inputs.

So each field defines TWO regions:
  - "box"  : the full drawn rectangle (useful for visualization/debugging,
             and as a fallback region for alignment sanity-checks).
  - "roi"  : a tighter zone of interest, left-anchored to the box's start
             (where handwriting realistically begins), sized to a
             reasonable max word/phrase width. This is what actually gets
             cropped and fed to the CRNN.

ROI width is currently a fixed 180pt (~2.5 inches), which comfortably fits
a Nepali first name, surname, or short place name in the sample handwriting
size implied by the box height (~24-25pt tall). If real filled samples show
people writing longer than this (e.g. long compound place names), widen
ROI_WIDTH_PT for that specific field below.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Page / rendering reference
# --------------------------------------------------------------------------

# THIS_DIR is the folder this config file itself lives in (e.g.
# .../notebooks/files/). Anchoring paths to THIS_DIR instead of a bare
# relative string means the pipeline works no matter what folder you're
# in when you run python3 -- you no longer have to `cd` into the exact
# script folder first. Put "template/<pdf>" as a subfolder next to this
# file and it will always be found.
THIS_DIR = Path(__file__).resolve().parent

REFERENCE_TEMPLATE_PATH = str(THIS_DIR / "template" / "Red_Minimalist_Membership_Form_A4.pdf")
PAGE_SIZE_PT = (595.5, 842.25)   # (width, height) in PDF points, from the source file

# DPI used when rasterizing the PDF to an image for alignment/cropping.
# All pixel-space conversions in the pipeline derive from this.
RENDER_DPI = 150
_PT_TO_PX = RENDER_DPI / 72.0  # PDF points are defined at 72 pt/inch

# --------------------------------------------------------------------------
# Field definitions
# --------------------------------------------------------------------------
# box: (x1, y1, x2, y2) in PDF points -- the full drawn input rectangle.
# roi_width_pt: width of the tight crop zone, anchored to the box's left
#               edge (x1), vertically centered on the box.

DEFAULT_ROI_WIDTH_PT = 180.0
DEFAULT_ROI_HEIGHT_PADDING_PT = 2.0  # small vertical padding beyond box height

FIELDS = {
    "first_name": {
        "label_np": "पहिलो नाम",
        "box": (151.2, 247.4, 533.8, 272.0),
    },
    "last_name": {
        "label_np": "थर",
        "box": (151.2, 284.2, 533.8, 308.8),
    },
    "place_of_birth": {
        "label_np": "जन्मस्थान",
        "box": (151.2, 329.0, 533.8, 353.6),
        "roi_width_pt": 220.0,  # place names can run longer
    },
    "father_first_name": {
        "label_np": "बुबाको पहिलो नाम",
        "box": (151.2, 371.3, 533.8, 395.9),
    },
    "father_last_name": {
        "label_np": "बुबाको थर",
        "box": (151.2, 412.6, 533.8, 437.1),
    },
    "mother_first_name": {
        "label_np": "आमाको पहिलो नाम",
        "box": (151.2, 460.0, 533.8, 484.6),
    },
    "mother_last_name": {
        "label_np": "आमाको थर",
        "box": (151.2, 507.5, 533.8, 532.0),
    },
    "nationality": {
        "label_np": "राष्ट्रियता",
        "box": (151.2, 551.6, 533.8, 576.1),
    },
    "city_district": {
        "label_np": "जिल्ला",
        "box": (151.2, 589.4, 533.8, 614.0),
    },
}


def get_roi_pt(field_def: dict) -> tuple[float, float, float, float]:
    """Compute the zone-of-interest rectangle (in PDF points) for a field,
    left-anchored to its box's start and padded slightly beyond box height.
    """
    x1, y1, x2, y2 = field_def["box"]
    roi_width = field_def.get("roi_width_pt", DEFAULT_ROI_WIDTH_PT)
    pad = DEFAULT_ROI_HEIGHT_PADDING_PT

    roi_x1 = x1
    roi_x2 = min(x2, x1 + roi_width)  # never exceed the drawn box's right edge
    roi_y1 = y1 - pad
    roi_y2 = y2 + pad
    return (roi_x1, roi_y1, roi_x2, roi_y2)


def pt_to_px(rect_pt: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """Convert a (x1, y1, x2, y2) rectangle from PDF points to pixel
    coordinates at RENDER_DPI.
    """
    x1, y1, x2, y2 = rect_pt
    return (
        round(x1 * _PT_TO_PX),
        round(y1 * _PT_TO_PX),
        round(x2 * _PT_TO_PX),
        round(y2 * _PT_TO_PX),
    )


CROP_PADDING_PX = 4  # extra pixel padding added after pt->px conversion, to avoid clipping matras
