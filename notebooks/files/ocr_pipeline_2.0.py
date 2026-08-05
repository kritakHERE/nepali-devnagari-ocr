"""
Nepali Devanagari OCR Pipeline v2.0 -- for lean phase-2 checkpoints
produced by `nepali-handwritten-word-recognition-ocr_2_1.ipynb`.

WHAT CHANGED FROM v1.0
-----------------------
1. Model architecture  : CRNNDevanagari (VGG16 with ASYMMETRIC pooling in
                         blocks 4 & 5) replaces the old CRNN (standard
                         VGG16 features + AdaptiveAvgPool).
                         Asymmetric pooling = MaxPool2d((2,1),(2,1)) at
                         features[23] and features[30], preserving temporal
                         width for CTC (40 timesteps at 320 px wide input).

2. Input resolution    : IMG_H=128, IMG_W=320  (was 64×256).

3. Checkpoint format   : v2 checkpoints are LEAN -- they only contain:
                           {'epoch', 'model_state', 'val_loss', 'cer', 'word_acc'}
                         No 'vocab' or 'num_classes' key is embedded.
                         Vocab is loaded from vocab.json (same directory).

4. Vocab loading       : Always from vocab.json (external). The old fallback
                         logic that read vocab from inside the checkpoint is
                         kept for backward compatibility with v1 checkpoints
                         that DO embed vocab.

STAGE OVERVIEW
--------------
1. Reference setup : rasterize the clean template PDF once, at RENDER_DPI,
                     to serve as the alignment reference.
2. Preprocess      : load an incoming scan, convert to grayscale, deskew.
3. Align           : register the scan against the reference render
                     (ORB features + homography).
4. Crop ROIs       : slice out each field's zone of interest.
5. OCR             : run the CRNN+CTC model on each cropped ROI.
6. Assemble        : return {field_name: recognized_text}.

USAGE
-----
    from ocr_pipeline_2 import OCRPipeline

    pipeline = OCRPipeline(
        model_path="../../models/best_models/phase2_BEST.pth",
        vocab_path="vocab.json",   # same directory as this file
    )
    result = pipeline.run("path/to/scanned_form.jpg")
    print(result)   # {"first_name": "राम", "last_name": "शर्मा", ...}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

import template_config as cfg
from nlp_postprocessor import NLPPostProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ocr_pipeline_2")

# ---------------------------------------------------------------------------
# Global constants -- must match training notebook exactly
# ---------------------------------------------------------------------------

IMG_H: int = 128          # nepali-handwritten-word-recognition-ocr_2_1.ipynb § 1
IMG_W: int = 320          # "Input resolution: 320 × 128 → 40 timesteps"
BILSTM_HIDDEN: int = 256
BILSTM_LAYERS: int = 2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Reference template setup
# ---------------------------------------------------------------------------

def render_reference_template(pdf_path: str, dpi: int = cfg.RENDER_DPI) -> np.ndarray:
    """Rasterize page 1 of the clean template PDF to a grayscale image."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = img[:, :, 0]
    return gray


# ---------------------------------------------------------------------------
# Stage: Preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess(image_path: str) -> np.ndarray:
    """Load a scanned/filled document image, convert to grayscale, deskew."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return _deskew(gray)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Correct small rotation using minAreaRect of thresholded foreground pixels."""
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


# ---------------------------------------------------------------------------
# Stage: Alignment
# ---------------------------------------------------------------------------

def align_to_template(scan_gray: np.ndarray, reference_gray: np.ndarray) -> np.ndarray:
    """Warp scan_gray to match reference_gray via ORB + homography."""
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


# ---------------------------------------------------------------------------
# Stage: ROI cropping
# ---------------------------------------------------------------------------

def crop_field_rois(aligned_image: np.ndarray) -> dict[str, np.ndarray]:
    """Slice out each field's zone-of-interest from an aligned image."""
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


# ---------------------------------------------------------------------------
# Stage: CRNN + CTC inference
# ---------------------------------------------------------------------------

class CRNNDevanagari(nn.Module):
    """VGG16 + asymmetric pooling + BiLSTM + CTC.

    Matches CRNNDevanagari in nepali-handwritten-word-recognition-ocr_2_1.ipynb (§ 6)
    exactly. The two critical departures from stock VGG16:

      - features[23] and features[30] (block 4 & 5 max-pools) are replaced
        with MaxPool2d(kernel_size=(2,1), stride=(2,1)), pooling height but
        NOT width. This keeps temporal resolution at 40 timesteps for a
        320-px-wide input, versus 10 with standard (2,2) pooling.
      - AdaptiveAvgPool2d((1, None)) collapses the remaining height (4 px)
        to 1 before the LSTM.

    Do NOT modify this class without also retraining, or the saved
    state_dict will refuse to load.
    """

    def __init__(self, num_classes: int, lstm_dropout: float = 0.0, fc_dropout: float = 0.0):
        super().__init__()

        # Load VGG16 feature extractor (weights=None: we load a fine-tuned
        # state_dict immediately after, so ImageNet init is wasted work)
        vgg = models.vgg16(weights=None)
        features = list(vgg.features.children())   # 31 layers, indices 0-30

        # Replace block-4 pool (index 23) and block-5 pool (index 30)
        # with asymmetric pooling: pool height but preserve width
        for idx in [23, 30]:
            features[idx] = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.cnn = nn.Sequential(*features)

        # Collapse remaining height (4 px) → 1, keep width (40) intact
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=512,
            hidden_size=BILSTM_HIDDEN,
            num_layers=BILSTM_LAYERS,
            bidirectional=True,
            dropout=lstm_dropout,
            batch_first=False,
        )

        self.fc_drop = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(BILSTM_HIDDEN * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 128, 320]
        feat = self.cnn(x)                  # [B, 512, 4, 40]
        feat = self.adaptive_pool(feat)     # [B, 512, 1, 40]
        feat = feat.squeeze(2)              # [B, 512, 40]
        feat = feat.permute(2, 0, 1)        # [40, B, 512]  time-major for LSTM
        out, _ = self.bilstm(feat)          # [40, B, 512]
        out = self.fc_drop(out)
        out = self.fc(out)                  # [40, B, num_classes]
        return out.log_softmax(2)           # required by CTCLoss / greedy decode


@dataclass
class CRNNRecognizer:
    """Loads a v2 lean checkpoint and runs inference.

    Checkpoint format (from 2_1.ipynb EarlyStopper.save):
        {'epoch', 'model_state', 'val_loss', 'cer', 'word_acc'}

    Vocab is NOT embedded in v2 checkpoints. It is loaded from vocab.json
    (same directory). For backward compat, if the checkpoint DOES contain
    a 'vocab' key (v1 format), that takes priority over vocab_path.
    """

    model_path: str
    vocab_path: str | None = None        # path to vocab.json
    img_height: int = IMG_H              # 128
    img_width: int = IMG_W               # 320
    device: str = "cpu"

    def __post_init__(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # ── Vocab: embedded (v1 compat) or external JSON (v2 default) ────────
        if "vocab" in checkpoint:
            self.vocab: list[str] = checkpoint["vocab"]
            logger.info("Vocab loaded from checkpoint (v1 format, %d chars)", len(self.vocab))
        elif self.vocab_path:
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
            logger.info("Vocab loaded from %s (%d chars)", self.vocab_path, len(self.vocab))
        else:
            raise ValueError(
                "Checkpoint has no embedded 'vocab' key and no vocab_path was provided. "
                "Pass vocab_path='vocab.json' (or the correct path) when constructing "
                "CRNNRecognizer / OCRPipeline."
            )

        # ── num_classes: embedded (v1) or derived from vocab length (v2) ─────
        num_classes: int = checkpoint.get("num_classes", len(self.vocab))

        # idx2char: index 0 is the CTC blank token -- matches Vocabulary class
        # in the training notebook where blank=0 and vocab[0] == '<BLANK>'
        self.idx_to_char: dict[int, str] = {i: c for i, c in enumerate(self.vocab)}

        # ── Build model and load weights ──────────────────────────────────────
        self.model = CRNNDevanagari(num_classes=num_classes)

        # v2 checkpoints use 'model_state'; guard against bare state_dict too
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            # bare state dict (someone called torch.save(model.state_dict(), path))
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "Loaded checkpoint: epoch=%s  val_loss=%s  cer=%s  vocab_size=%d",
            checkpoint.get("epoch", "?"),
            checkpoint.get("val_loss", "?"),
            checkpoint.get("cer", "?"),
            len(self.vocab),
        )

    def _preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """Matches the training-time transform from 2_1.ipynb make_transform():
            Resize((IMG_H, IMG_W))
            → Grayscale(num_output_channels=3)   [same value across R/G/B]
            → ToTensor()
            → Normalize(IMAGENET_MEAN, IMAGENET_STD)
        """
        if crop.size == 0:
            raise ValueError("Empty crop passed to recognizer -- check ROI coordinates.")

        # Resize to training resolution: (width, height) for cv2
        resized = cv2.resize(crop, (self.img_width, self.img_height))

        # Grayscale → 3-channel (duplicate channel, matching PIL Grayscale(3))
        if resized.ndim == 2:
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            rgb = resized

        # Normalize: float32 [0,1] then ImageNet stats
        normalized = rgb.astype(np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std  = np.array(IMAGENET_STD,  dtype=np.float32)
        normalized = (normalized - mean) / std

        # HWC → CHW → batch dim
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    def _ctc_greedy_decode(self, log_probs: torch.Tensor) -> str:
        """Greedy CTC decode -- mirrors ctc_decode() in 2_1.ipynb:
        argmax over classes, collapse repeats, drop blanks (index 0).
        """
        preds = log_probs.argmax(dim=2).permute(1, 0)   # (B, T), B=1 here
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


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------

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
            vocab_path=vocab_path,   # forwarded from CLI / caller
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nepali OCR pipeline v2.0 (lean checkpoint format).")
    parser.add_argument("image", help="Path to the scanned/filled document image")
    parser.add_argument("--model", required=True, help="Path to phase2_BEST.pth checkpoint")
    parser.add_argument("--template-pdf", default=None, help="Path to the clean template PDF")
    parser.add_argument("--vocab", default="vocab.json", help="Path to vocab.json (default: vocab.json)")
    parser.add_argument("--output", default=None, help="Optional path to write JSON output")
    args = parser.parse_args()

    pipeline = OCRPipeline(
        model_path=args.model,
        template_pdf_path=args.template_pdf,
        vocab_path=args.vocab,
    )
    print(pipeline.run_to_json(args.image, args.output))
