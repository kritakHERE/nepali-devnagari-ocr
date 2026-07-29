"""
Nepali Devanagari OCR Pipeline -- built around the fixed template
`Red_Minimalist_Membership_Form_A4.pdf`.

STAGE OVERVIEW
--------------
1. Reference setup : rasterize the clean template PDF once, at RENDER_DPI,
                      to serve as the alignment reference and the source of
                      truth for pixel-space field coordinates.
2. Preprocess       : load an incoming scan, convert to grayscale, deskew.
3. Align            : register the scan against the reference render
                      (ORB features + homography), so template_config's
                      fixed coordinates line up regardless of scan
                      skew/shift/scale.
4. Crop ROIs        : slice out each field's ZONE OF INTEREST -- a tight,
                      left-anchored sub-region of the drawn input box, not
                      the full oversized box (see template_config.py for
                      why this matters).
5. OCR              : run the CRNN+CTC model on each cropped ROI.
6. Assemble         : return {field_name: recognized_text} as a dict/JSON.

USAGE
-----
    from ocr_pipeline import OCRPipeline

    pipeline = OCRPipeline(model_path="models/p2_epoch19_acc49.1.pth")
    result = pipeline.run("path/to/scanned_form.jpg")
    print(result)   # {"first_name": "राम", "last_name": "शर्मा", ...}

Note there is no char_list_path argument -- the vocabulary is saved
INSIDE the .pth checkpoint itself (see CRNNRecognizer below), matching
how `phase2_nepali_finetune.ipynb` saves checkpoints.

WHAT'S REAL VS. WHAT'S STILL UNVALIDATED
--------------------------------------------
REAL (verified against your actual files, not guessed):
  - All 9 field box coordinates (template_config.py), extracted directly
    from the template PDF's vector content.
  - ROI (zone-of-interest) cropping logic, addressing the oversized-box issue.
  - CRNN architecture: VGG16 + BiLSTM + CTC, copied exactly from
    `phase2_nepali_finetune.ipynb` (Cell 9).
  - Preprocessing (resize to 64x256, grayscale->3-channel, ImageNet
    normalization) and CTC greedy decode, copied exactly from the same
    notebook (Cells 7 and 11).
  - Vocab loading directly from the checkpoint dict (`ckpt['vocab']`,
    `ckpt['num_classes']`) -- matches how the notebook saves checkpoints.

STILL UNVALIDATED (this is a structural/architectural match, not yet a
tested end-to-end run):
  - No real filled-sample testing has been done -- validate against actual
    filled scans once available. Field crops from the FORM (64px-tall
    boxes at 150 DPI) may need resizing/scale tuning to look like what the
    model saw during training (font-rendered word images), especially
    once real handwriting is involved instead of synthetic/printed text.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz  # PyMuPDF, for rendering the reference PDF
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

import template_config as cfg
from nlp_postprocessor import NLPPostProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ocr_pipeline")


# --------------------------------------------------------------------------
# Reference template setup
# --------------------------------------------------------------------------

def render_reference_template(pdf_path: str, dpi: int = cfg.RENDER_DPI) -> np.ndarray:
    """Rasterize page 1 of the clean template PDF to a grayscale image at
    the given DPI. This becomes the alignment target and defines pixel
    space for all field coordinates.
    """
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = img[:, :, 0]
    return gray


# --------------------------------------------------------------------------
# Stage: Preprocessing (incoming scans)
# --------------------------------------------------------------------------

def load_and_preprocess(image_path: str) -> np.ndarray:
    """Load a scanned/filled document image and apply basic cleanup:
    grayscale + deskew. Returns a single-channel image.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return _deskew(gray)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Correct small rotation using minAreaRect of thresholded foreground
    pixels. Handles scanner-induced skew; larger shifts/perspective are
    handled separately by the homography alignment step.
    """
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle

    if abs(angle) < 0.3:
        return gray

    (h, w) = gray.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        gray, rot_matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


# --------------------------------------------------------------------------
# Stage: Alignment to reference template
# --------------------------------------------------------------------------

def align_to_template(scan_gray: np.ndarray, reference_gray: np.ndarray) -> np.ndarray:
    """Warp `scan_gray` to match `reference_gray`'s layout via ORB keypoint
    matching + homography, so fixed ROI coordinates remain valid.
    """
    orb = cv2.ORB_create(nfeatures=3000)
    kp1, des1 = orb.detectAndCompute(scan_gray, None)
    kp2, des2 = orb.detectAndCompute(reference_gray, None)

    if des1 is None or des2 is None:
        raise RuntimeError(
            "Could not find enough features to align scan to template. "
            "Check image quality / contrast."
        )

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)

    num_good = max(int(len(matches) * 0.15), 10)
    good_matches = matches[:num_good]
    if len(good_matches) < 4:
        raise RuntimeError(
            "Too few reliable matches between scan and reference template "
            "to compute alignment."
        )

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if homography is None:
        raise RuntimeError("Homography estimation failed.")

    h, w = reference_gray.shape[:2]
    return cv2.warpPerspective(scan_gray, homography, (w, h))


# --------------------------------------------------------------------------
# Stage: ROI cropping
# --------------------------------------------------------------------------

def crop_field_rois(aligned_image: np.ndarray) -> dict[str, np.ndarray]:
    """Slice out each field's zone-of-interest (tight crop, not the full
    oversized input box) from an already-aligned image.
    """
    crops = {}
    h, w = aligned_image.shape[:2]
    pad = cfg.CROP_PADDING_PX

    for field_name, field_def in cfg.FIELDS.items():
        roi_pt = cfg.get_roi_pt(field_def)
        x1, y1, x2, y2 = cfg.pt_to_px(roi_pt)

        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

        crops[field_name] = aligned_image[y1:y2, x1:x2]

    return crops


# --------------------------------------------------------------------------
# Stage: CRNN + CTC inference
# --------------------------------------------------------------------------

class CRNN(nn.Module):
    """VGG16 + BiLSTM + CTC.

    This EXACTLY matches the architecture defined in the training notebook
    `phase2_nepali_finetune.ipynb` (Cell 9) that produced the
    p2_epoch*.pth checkpoints. Do not modify this class without also
    updating the training notebook, or checkpoints will stop loading.

    Key details that must stay in sync with training:
      - Backbone: torchvision vgg16 .features (ImageNet-pretrained)
      - Pooling: AdaptiveAvgPool2d((1, None)) -- collapses height to 1,
        keeps width variable (sequence length for the RNN)
      - RNN: 2-layer bidirectional LSTM, hidden_size=256, dropout=0.3
      - Output: log_softmax over classes (NOT raw logits) -- CTC decode
        must account for this, though softmax/log_softmax don't change
        the argmax used for greedy decoding.
    """

    def __init__(self, num_classes: int, hidden_size: int = 256):
        super().__init__()
        # weights=None here: we always load a fine-tuned state_dict right
        # after construction, so ImageNet init would be immediately
        # overwritten anyway. Avoids an unnecessary download at inference time.
        vgg = models.vgg16(weights=None)
        self.cnn = vgg.features

        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=0.3,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)
        f = self.pool(f)
        f = f.squeeze(2)          # (B, C, W)
        f = f.permute(2, 0, 1)    # (W, B, C) -- time-major, matches training
        r, _ = self.rnn(f)
        out = self.fc(r)
        return torch.nn.functional.log_softmax(out, dim=2)


@dataclass
class CRNNRecognizer:
    """Loads a p2_epoch*.pth checkpoint and runs inference.

    IMPORTANT: unlike a generic setup, this checkpoint format (as saved by
    the training notebook) bundles the vocabulary INSIDE the .pth file
    itself (`ckpt['vocab']`, `ckpt['num_classes']`). There is no separate
    char_list.txt to manage -- the checkpoint is self-contained.
    """

    model_path: str
    img_height: int = 64   # must match training IMG_HEIGHT
    img_width: int = 256   # must match training IMG_WIDTH
    device: str = "cpu"

    def __post_init__(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Checkpoints save vocab + num_classes directly -- no external file needed.
        self.vocab: list[str] = checkpoint["vocab"]
        num_classes: int = checkpoint["num_classes"]
        # idx2char must match training exactly: index 0 is reserved for the
        # CTC blank token, so vocab[i] corresponds to class index i (not i+1) --
        # this mirrors ctc_decode() in the training notebook (Cell 11), where
        # idx2char = {i: c for i, c in enumerate(vocab)} and blank=0 is skipped
        # during decoding, not omitted from the vocab list itself.
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}

        self.model = CRNN(num_classes=num_classes)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "Loaded checkpoint: epoch=%s val_acc=%s vocab_size=%d",
            checkpoint.get("epoch", "?"),
            checkpoint.get("val_acc", "?"),
            len(self.vocab),
        )

    def _preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """Matches the training-time transform exactly:
        grayscale crop -> 3-channel RGB -> resize -> tensor -> ImageNet
        normalization. Training used PIL's Grayscale(num_output_channels=3),
        i.e. the same single-channel values duplicated across R/G/B, not a
        true color image -- so we replicate the crop across 3 channels here.
        """
        if crop.size == 0:
            raise ValueError("Empty crop passed to recognizer -- check ROI coordinates.")

        resized = cv2.resize(crop, (self.img_width, self.img_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB) if resized.ndim == 2 else resized
        normalized = rgb.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std

        # HWC -> CHW -> add batch dim
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    def _ctc_greedy_decode(self, log_probs: torch.Tensor) -> str:
        """Mirrors ctc_decode() from the training notebook (Cell 11) exactly:
        argmax over classes, collapse repeats, drop blanks (index 0).
        """
        preds = log_probs.argmax(dim=2).permute(1, 0)  # (B, T) -- B=1 here
        seq = preds[0].tolist()

        chars, prev = [], None
        for idx in seq:
            if idx != prev and idx != 0:
                chars.append(self.idx_to_char.get(idx, ""))
            prev = idx
        return "".join(chars)

    def predict(self, crop: np.ndarray) -> str:
        with torch.no_grad():
            tensor = self._preprocess_crop(crop)
            log_probs = self.model(tensor)
            return self._ctc_greedy_decode(log_probs)


# --------------------------------------------------------------------------
# Full pipeline orchestration
# --------------------------------------------------------------------------

class OCRPipeline:
    def __init__(
        self,
        model_path: str,
        template_pdf_path: str | None = None,
        vocab_path: str = "vocab.json",
        device: str = "cpu",
    ):
        pdf_path = template_pdf_path or cfg.REFERENCE_TEMPLATE_PATH
        self.reference_gray = self._load_reference(pdf_path)
        self.recognizer = CRNNRecognizer(
            model_path=model_path,
            device=device,
        )
        self.postprocessor = NLPPostProcessor(vocab_path=vocab_path)

    @staticmethod
    def _load_reference(pdf_path: str) -> np.ndarray:
        p = Path(pdf_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Reference template PDF not found at {p}. Update "
                "template_config.REFERENCE_TEMPLATE_PATH or pass "
                "template_pdf_path explicitly."
            )
        return render_reference_template(str(p))

    def run(self, image_path: str) -> dict[str, str]:
        logger.info("Preprocessing scan: %s", image_path)
        scan_gray = load_and_preprocess(image_path)

        logger.info("Aligning scan to reference template")
        aligned = align_to_template(scan_gray, self.reference_gray)

        logger.info("Cropping %d field ROIs", len(cfg.FIELDS))
        crops = crop_field_rois(aligned)

        results: dict[str, str] = {}
        for field_name, crop in crops.items():
            try:
                text = self.recognizer.predict(crop)
            except Exception as e:  # noqa: BLE001
                logger.warning("Field '%s' failed OCR: %s", field_name, e)
                text = ""

            corrected = self.postprocessor.correct_field(field_name, text)
            results[field_name] = corrected

            if corrected != text:
                logger.info("  %-20s -> %r  (corrected from %r)", field_name, corrected, text)
            else:
                logger.info("  %-20s -> %r", field_name, corrected)

        return results

    def run_to_json(self, image_path: str, output_path: str | None = None) -> str:
        result = self.run(image_path)
        payload = json.dumps(result, ensure_ascii=False, indent=2)
        if output_path:
            Path(output_path).write_text(payload, encoding="utf-8")
            logger.info("Wrote output to %s", output_path)
        return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Nepali OCR pipeline on a scanned form.")
    parser.add_argument("image", help="Path to the scanned/filled document image")
    parser.add_argument("--model", required=True, help="Path to a p2_epoch*.pth checkpoint")
    parser.add_argument("--template-pdf", default=None, help="Path to the clean template PDF")
    parser.add_argument("--vocab", default="vocab.json", help="Path to vocab.json for NLP post-processing")
    parser.add_argument("--output", default=None, help="Optional path to write JSON output")
    args = parser.parse_args()

    pipeline = OCRPipeline(
        model_path=args.model,
        template_pdf_path=args.template_pdf,
        vocab_path=args.vocab,
    )
    print(pipeline.run_to_json(args.image, args.output))
