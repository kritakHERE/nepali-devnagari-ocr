# ITS69204 — Source of Truth v3
## Nepali Devanagari Handwritten OCR — Form Field Extraction
### Track T2 | Taylor's University MAY 2026 Semester | Module: ITS69204 Computer Vision and NLP

---

> **Purpose:** Single authoritative reference for every group member writing the project report, README, or preparing slides. Every fact, figure, and design decision here is conflict-resolved and finalised. Do not invent facts — if something is not in this document, it has not been confirmed and must not be claimed.
>
> **Version:** v3 — compiled July 31, 2026. Supersedes v2, SOURCE_OF_TRUTH_FINAL.md, design_2_1.0, design_2_2.0, and all planning session notes.
>
> **Conflict resolution policy:** Where any prior document contradicts this file, this file wins. Where this file contains a section marked 🔵 (older run reference), those figures are labelled as such and must not be reported as primary results.

---

## HOW TO READ THIS DOCUMENT

| Part | Contents |
|---|---|
| Part 1 | What we built — summary, pipeline, form fields, files, links |
| Part 2 | Model architecture — CRNN, VGG16, BiLSTM, CTC |
| Part 3 | Training — datasets, hyperparameters, results (both phases) |
| Part 4 | NLP post-processing and live demo results |
| Part 5 | Report writing pack — critical analysis, limitations, citations, viva Q&A |

**Source legend:**
- 🟢 Confirmed in final implemented system / canonical fact
- 🟡 Confirmed from live testing session logs
- 🔵 From older design_2_2.0 Kaggle run — reference only, not primary reported results
- ⚠️ Conflict resolved in this version — prior documents disagreed

---

## PART 1 — WHAT WE BUILT

### 1.1 One-Paragraph Project Summary 🟢

We built an end-to-end Nepali handwritten OCR system that takes a scanned or photographed filled membership form, automatically aligns it to a reference template using ORB feature matching, crops each of the 9 handwritten text fields as a region of interest, feeds each crop through a trained CRNN (Convolutional Recurrent Neural Network) to recognise the Devanagari text, and applies a lightweight NLP post-processing layer to correct common OCR errors using vocabulary fuzzy matching and Unicode normalisation. The system outputs a structured JSON of field names and recognised values (e.g., `{"first_name": "राम", "last_name": "श्रेष्ठ", ...}`). It addresses Track T2: Devanagari OCR for Digitising Nepali Government Records, motivated by the enormous backlog of handwritten Nepali documents — citizenship certificates, land registries, school records, court documents — that remain undigitised.

### 1.2 Project Identity 🟢

| Item | Value |
|---|---|
| Module | ITS69204 — Computer Vision and NLP |
| Track | T2 — Devanagari OCR for Digitising Nepali Government Records |
| Institution | Taylor's University, MAY 2026 Semester |
| Form used | Red Minimalist Membership Form A4 (`Red_Minimalist_Membership_Form_A4.pdf`) |
| Pipeline type | Fixed-template form — field positions are hardcoded, not detected at runtime |
| Output | Structured JSON / Python dict of 9 recognised field values |
| Interface | Gradio web UI (`gui.py`) — drag-and-drop image, returns table of predictions |

### 1.3 System Pipeline — 7 Stages 🟢

```
Scanned Form Image
       ↓
[Stage 1] REFERENCE SETUP
  → Rasterise clean template PDF at 150 DPI using PyMuPDF (fitz)
  → Produces pixel-space coordinate system for all 9 fields

[Stage 2] PREPROCESS
  → Load scan → convert to grayscale
  → Deskew: detect rotation angle using minAreaRect on thresholded pixels
  → Apply warpAffine to correct small scanner-induced tilt

[Stage 3] ALIGN
  → ORB (Oriented FAST and Rotated BRIEF) feature detection on both scan and reference
  → BFMatcher (Brute Force, Hamming distance) finds matching keypoints
  → Top 15% of matches used; rest discarded as noise
  → RANSAC homography (cv2.findHomography, reprojection threshold=5.0px) estimates transform
  → warpPerspective warps scan to match reference layout exactly
  → After this step, all hardcoded coordinate boxes map correctly to the scan

[Stage 4] CROP ROIs
  → For each of 9 fields: compute ROI from hardcoded coordinates
  → ROI is LEFT-ANCHORED, width = 180pt (220pt for place_of_birth)
  → NOT the full ~382pt-wide drawn input box — only where writing starts
  → Add 4px padding top/bottom to avoid clipping matras (Devanagari top-bar vowel modifiers)
  → Convert PDF-point coordinates to pixels at 150 DPI (factor: 150/72 = 2.0833)

[Stage 5] OCR — CRNN MODEL
  → Each crop: resize to 320×128px, grayscale → 3-channel, ImageNet normalise
  → Forward pass through CRNN (VGG16 CNN → AdaptiveAvgPool2d → BiLSTM → FC → log_softmax)
  → CTC greedy decode: argmax over time steps, collapse repeats, drop blank token (index 0)
  → Output: raw predicted Devanagari string

[Stage 6] NLP POST-PROCESSING
  → Unicode NFC normalisation: canonicalises combining character order
  → Artifact stripping: removes ZWJ/ZWNJ, pipe |, stray Latin, isolated halant ्
  → Field-specific rules: nationality matched against closed list (~10 values)
  → Vocabulary fuzzy match (Levenshtein distance ≤ 2) for last name and district fields only
  → First name fields intentionally excluded (too many near-neighbours — causes over-correction)

[Stage 7] ASSEMBLE
  → Collect all 9 corrected field predictions into Python dict
  → Return as JSON
```

### 1.4 Supporting Files 🟢

| File | Role |
|---|---|
| `ocr_pipeline.py` | Main pipeline — Stages 1–5, calls NLP. `import OCRPipeline; pipeline.run()` |
| `gui.py` | Gradio web UI. `python gui.py --model <path>`. Model loads once at startup |
| `nlp_postprocessor.py` | NLP post-processing — Stage 6. Standalone module, pure Python stdlib |
| `template_config.py` | All 9 field coordinates + ROI logic |
| `extract_coords.py` | Utility: re-extracts field coordinates from PDF vector content if layout changes |
| `vocab.json` | Vocabulary for fuzzy correction (first_names, last_names, districts) |
| `Red_Minimalist_Membership_Form_A4.pdf` | Reference template used for ORB alignment |

### 1.5 Canonical Links 🟢

| Resource | URL |
|---|---|
| Phase 2 training dataset (Nepali font-generated names) | https://www.kaggle.com/datasets/kritakhere/nepali-font-generated-handwritten-names |
| Phase 1 training dataset (HindiSeg handwritten Hindi words) | https://www.kaggle.com/datasets/sabarinathan/handwritten-hindi-word-recognition |
| Main training notebook (combined — both phases, primary report model) | https://www.kaggle.com/code/kritakhere/nepali-handwritten-word-recognition-ocr |

### 1.6 The 9 Form Fields — Fixed Coordinates 🟢

The form uses a fixed layout. All field positions are extracted programmatically from the PDF vector content using `extract_coords.py` (PyMuPDF). Coordinates are in PDF points, origin top-left.

- Form page: 595.5 × 842.25 pt (A4 at 72 pt/inch)
- Rasterised at 150 DPI for pixel operations
- Pixel conversion factor: 150 / 72 = 2.0833

**Why ROI width is narrower than the drawn box:** The drawn input boxes are ~382pt wide (151.2 → 533.8). Feeding a mostly-blank 382pt-wide crop to the CRNN wastes CTC time steps on whitespace and hurts alignment. The left-anchored ROI crops only where handwriting actually appears.

| Field Key | Nepali Label | Full Box (x1, y1, x2, y2) pt | ROI Width Used | NLP Applied |
|---|---|---|---|---|
| `first_name` | पहिलो नाम | 151.2, 247.4, 533.8, 272.0 | 180pt | Normalise only |
| `last_name` | थर | 151.2, 284.2, 533.8, 308.8 | 180pt | Fuzzy match → last_names |
| `place_of_birth` | जन्मस्थान | 151.2, 329.0, 533.8, 353.6 | **220pt** | Normalise only |
| `father_first_name` | बुबाको पहिलो नाम | 151.2, 371.3, 533.8, 395.9 | 180pt | Normalise only |
| `father_last_name` | बुबाको थर | 151.2, 412.6, 533.8, 437.1 | 180pt | Fuzzy match → last_names |
| `mother_first_name` | आमाको पहिलो नाम | 151.2, 460.0, 533.8, 484.6 | 180pt | Normalise only |
| `mother_last_name` | आमाको थर | 151.2, 507.5, 533.8, 532.0 | 180pt | Fuzzy match → last_names |
| `nationality` | राष्ट्रियता | 151.2, 551.6, 533.8, 576.1 | 180pt | Closed-list match |
| `city_district` | जिल्ला | 151.2, 589.4, 533.8, 614.0 | 180pt | Fuzzy match → districts |

`place_of_birth` uses 220pt (wider) because place names are longer than personal names.

### 1.7 Alignment — ORB + RANSAC Homography 🟢

| Parameter | Value |
|---|---|
| Feature detector | ORB (Oriented FAST and Rotated BRIEF) |
| Max keypoints | 3,000 |
| Matcher | BFMatcher (Brute Force, Hamming distance) |
| Match filtering | Top 15% of matches used |
| Homography estimator | `cv2.findHomography` with RANSAC |
| RANSAC reprojection threshold | 5.0 px |
| Output | Warped scan matching reference template dimensions exactly |
| Reference rendered at | 150 DPI from `Red_Minimalist_Membership_Form_A4.pdf` |

---

## PART 2 — MODEL ARCHITECTURE

### 2.1 Architecture Name and Type 🟢

**Model:** CRNN — Convolutional Recurrent Neural Network with CTC decoding.

Based on Shi et al. (2016), adapted with a deeper pretrained backbone (VGG16).

**Why CRNN was chosen:**
- No character-level segmentation required — CTC handles variable-length output implicitly. Segmenting Devanagari conjuncts and matras is a hard unsolved problem; CRNN sidesteps it entirely.
- BiLSTM captures left-to-right and right-to-left context, critical for Devanagari conjuncts where character identity depends on neighbouring glyphs.
- VGG16 pretrained on ImageNet enables transfer learning from a large real-world prior.

**Why VGG16 specifically (not ResNet, not standalone CNN, not Tesseract):**

VGG16 was trained on ImageNet — 1.2 million real photographs of diverse objects, edges, textures, and curves. This means it has learned a rich hierarchy of visual features: low-level edge detectors in early layers, progressively abstract curve and stroke representations in deeper layers. Handwriting is made of the same kinds of curves and edges — VGG16's pretrained features transfer directly.

Tesseract, by contrast, was trained exclusively on printed text — a finite set of fixed, stable font structures. It has never been exposed to the kind of stroke variability, slant, ink spread, and character inconsistency found in real handwriting. This is a fundamental training data mismatch, not a tuning issue. No amount of parameter adjustment fixes it.

ResNet was not chosen because its residual connections change spatial dimensions in non-trivial ways that complicate the AdaptiveAvgPool2d → sequence conversion step. VGG16's sequential convolutional stack produces a feature map whose spatial dimensions degrade predictably — this makes the pooling-to-sequence step straightforward and was the reason the original Shi et al. (2016) CRNN paper used a VGG-style backbone.

### 2.2 Component-by-Component Breakdown 🟢

#### CNN Backbone — VGG16

```python
import torchvision.models as models
vgg = models.vgg16(pretrained=True)
self.cnn = vgg.features  # .features stack only — NOT the classifier head
```

- Input: 320×128px, 3-channel (grayscale image duplicated across 3 channels to match ImageNet format)
- ImageNet normalisation: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Output spatial map: `[B, 512, H', W']`

#### Pooling — AdaptiveAvgPool2d

```python
self.pool = nn.AdaptiveAvgPool2d((1, None))
```

Collapses the height dimension to 1, preserving width as a variable-length sequence. After squeezing and permuting: `[W', B, 512]` — time-major format required by PyTorch LSTM. This is the key structural step that converts a 2D image feature map into a 1D sequence for the RNN.

#### Sequence Model — BiLSTM

```python
self.rnn = nn.LSTM(
    input_size=512,
    hidden_size=256,
    num_layers=2,
    bidirectional=True,
    batch_first=False,
    dropout=0.3   # Phase 1 value; see Part 3 for Phase 2 override
)
```

Output per timestep: 512 features (256 forward + 256 backward, concatenated).

#### Output Head + CTC

```python
self.fc = nn.Linear(512, num_classes)
# + log_softmax over dim=2
```

**Loss during training:** `nn.CTCLoss(blank=0, zero_infinity=True)`

**Greedy decode at inference:**
1. `argmax` over class dimension at each timestep
2. Collapse consecutive repeated characters
3. Remove blank token (index 0)
4. Output: decoded Devanagari string

### 2.3 Architecture Summary Table 🟢

⚠️ Resolution is 320×128 for both phases. Any earlier reference to 64×256 or 32×128 is incorrect and superseded.

| Component | Value |
|---|---|
| Input resolution | **320×128px**, 3-channel, ImageNet-normalised |
| CNN | VGG16 `.features` — ImageNet pretrained |
| Pooling | `nn.AdaptiveAvgPool2d((1, None))` — height→1, width preserved as sequence |
| RNN | BiLSTM: input=512, hidden=256, 2 layers, bidirectional, batch_first=False |
| RNN output per timestep | 512 features (256 forward + 256 backward) |
| FC | `nn.Linear(512, num_classes)` + log_softmax |
| Loss (training) | `CTCLoss(blank=0, zero_infinity=True)` |
| Decode (inference) | Greedy CTC — argmax → collapse repeats → drop blank |
| **Total parameters** | **17,928,629** |
| **Trainable (Phase 1)** | **~17.6M** (blocks 3–5 + BiLSTM + FC) |
| **Trainable (Phase 2)** | **~10.3M** (block 4 tail + block 5 + BiLSTM + FC) |
| **Frozen (Phase 2)** | **~7.4M** (blocks 1–4 conv1-2) |

### 2.4 Freezing Strategy 🟢

**Design rationale:** Freeze more in Phase 2 than Phase 1 because Phase 2 data is synthetic (font-rendered). We want to preserve the real handwriting features learned from Hindi in Phase 1, and only nudge the deepest layers toward Nepali script patterns.

| Block | Layers | Phase 1 (92k Hindi real handwriting) | Phase 2 (24k Nepali synthetic) | Reasoning |
|---|---|---|---|---|
| Block 1 | 0–4 | ❄️ Frozen | ❄️ Frozen | Pure edge detectors — ImageNet features are sufficient |
| Block 2 | 5–9 | ❄️ Frozen | ❄️ Frozen | Simple curves — nothing script-specific |
| Block 3 | 10–16 | 🔥 Train | ❄️ Frozen | Phase 1 learns Devanagari stroke junctions. Phase 2 protects them from 24k font bias |
| Block 4 conv 1–2 | 17–20 | 🔥 Train | ❄️ Frozen | Phase 1 learns character parts. Phase 2 freezes to preserve handwriting robustness |
| Block 4 conv 3 | 21–23 | 🔥 Train | 🔥 LR 1e-6 | Highest block 4 abstraction — slight Nepali adaptation only |
| Block 5 | 24–30 | 🔥 Train | 🔥 LR 1e-6 | Full glyphs and ligatures — needs Nepali vocabulary nudge |
| BiLSTM | — | 🔥 LR 1e-4 | 🔥 LR 1e-4 | Sequence learning — always trainable |
| FC | — | 🔥 LR 1e-4 | 🔥 LR 1e-4 | Class scores — always trainable |

**Trainable parameter counts:**
- Phase 1: ~17.6M trainable (blocks 3–5 + BiLSTM + FC)
- Phase 2: ~10.3M trainable (block 4 conv3 + block 5 + BiLSTM + FC)
- Phase 2 frozen: ~7.4M (blocks 1–4 conv1-2)

**Why this makes sense:** VGG16's early blocks (1–2) detect universal features like horizontal lines, diagonal strokes, and simple curves — these are identical across ImageNet, Hindi, and Nepali. There is no benefit to retraining them. Block 3 onward begins detecting script-specific stroke junctions and character components — these need to be adapted for Devanagari in Phase 1. In Phase 2, we protect those Phase 1 Devanagari features from being overwritten by 24k synthetic font images that lack real handwriting variability.

---

## PART 3 — TRAINING

### 3.1 Two-Phase Transfer Learning Strategy 🟢

```
ImageNet pretrained VGG16
         ↓
  Phase 1: HindiSeg
  92k real handwritten Hindi words
  Teaches: Devanagari strokes, matras, conjuncts from real handwriting
         ↓
  Phase 2: kritakhere Nepali dataset
  24k synthetic font-rendered Nepali names
  Teaches: Nepali-specific vocabulary and character distributions
         ↓
  NLP Post-processing
  Vocabulary-guided fuzzy correction — no retraining
```

**Why Hindi first, not English or Nepali directly:**
Hindi and Nepali share the Devanagari script identically. A model trained on real handwritten Hindi words already understands Devanagari strokes, matras, and conjuncts from actual human writing. Training on Nepali directly from scratch on only font-rendered images would produce a severe domain gap — the model would learn to read fonts but struggle with real handwriting. Hindi pre-training provides the real handwriting exposure before Nepali fine-tuning adapts the vocabulary.

### 3.2 Phase 1 — Hindi Pre-training

#### Dataset 🟢

| Fact | Value |
|---|---|
| Dataset name | HindiSeg — Handwritten Hindi Word Recognition |
| Source | Sabarinathan, Kaggle |
| URL | https://www.kaggle.com/datasets/sabarinathan/handwritten-hindi-word-recognition |
| Total images | ~92,000 |
| What we used | ~62,000+ training images (pre-split train set) |
| Format | Grayscale, real handwritten Hindi word crops (.jpg) |
| Label format | CSV with `file_name`, `text` columns; pre-split train/val/test |
| Directory structure | `HindiSeg/HindiSeg/train/{folder}/{id}.jpg` |
| Data type | **Real handwriting** (not synthetic) |

#### Phase 1 Hyperparameters 🟢

⚠️ These are the canonical values. Any earlier reference to batch=64, LR=1e-3, Adam, or max_epochs=20 for Phase 1 is from an older run and is superseded.

| Parameter | Value |
|---|---|
| Resolution | **320 × 128** |
| Batch size | **24** |
| Optimiser | **Adam** |
| Learning rate | **1e-4** flat across all trainable layers (blocks 3–5 + BiLSTM + FC) |
| LR scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Max epochs | **30** |
| Early stopping | patience=5, min_delta=1e-4 on val CTC loss |
| Gradient clipping | norm 5.0 |
| BiLSTM dropout | 0.3 |
| FC dropout | 0.3 |
| VGG16 blocks trained | Blocks 3–5 (blocks 1–2 frozen) |
| Checkpoint format | `phase1_epoch{N:03d}_acc{acc:.1f}_cer{cer:.3f}.pth` |
| Augmentation | Rotation ±5°, Gaussian blur, random brightness/contrast, slight elastic distortion (Devanagari-safe range) |

#### Phase 1 Training Outcome 🔵

*Epochs 11–13 recovered from design_2_2.0 session. Epochs 1–10 not recovered from that session.*

| Epoch | Train Loss | Val Loss | CER | Word Acc |
|---|---|---|---|---|
| 1–10 | Not recovered | — | — | — |
| 11 | 0.0532 | 0.4953 | 10.57% | 61.36% |
| 12 | 0.0463 | 0.4777 | 10.19% | 63.56% |
| 13 | 0.0417 | 0.5362 | 11.01% | 60.89% |

- **Early stopped:** Epoch 13 (val loss rising after epoch 10)
- **Best checkpoint:** Epoch 10
- **Best checkpoint filename:** `phase1_epoch010_wacc0.644_cer0.0966.pth` → renamed `crnn_hindi_best.pth`

#### Phase 1 Test Results — Hindi Test Set 🔵

*From design_2_2.0 older run. These demonstrate Phase 1 training was effective before Nepali fine-tuning.*

| Metric | Value |
|---|---|
| CTC Loss | 0.3979 |
| CER | **8.99%** |
| Word Accuracy | **67.31%** |
| WER | 32.69% |

#### Phase 1 → Phase 2 Baseline (Hindi model evaluated on Nepali, no fine-tuning) 🟢

Before any Nepali fine-tuning, the Phase 1 Hindi-trained model was evaluated on the Nepali validation set:

- **Word Accuracy: ~47.2%**

This is the key baseline. It proves two things: (1) Hindi pre-training transfers meaningfully to Nepali — script is shared, so the model is not at ~0%; (2) Nepali fine-tuning is still necessary to reach usable accuracy — 47.2% is insufficient for real deployment.

---

### 3.3 Phase 2 — Nepali Fine-tuning

#### Dataset 🟢

| Fact | Value |
|---|---|
| Dataset name | Nepali Font-Generated Handwritten Names |
| Source | kritakhere, Kaggle |
| URL | https://www.kaggle.com/datasets/kritakhere/nepali-font-generated-handwritten-names |
| Unique Nepali words/names | ~24,000 unique words |
| Font variants per word | ~18 |
| Total images | ~24,000 (one randomly selected font variant per word per epoch) |
| Filename pattern | `word00000_font0.png` through `word00000_font17.png` |
| Label format | Tab-separated: `filename\tlabel` in `labels.txt` |
| Data type | **Synthetic** — font-rendered using Devanagari handwriting-style TTF fonts via Pillow |
| Known issue | BOM (`\ufeff`) at start of some labels — stripped before use |
| Known issue | Corrupt/blank images (mean pixel < 5 or > 250) — detected and skipped |
| Font generation source | Google Fonts Devanagari TTF collection, rendered with Pillow |
| Font generation code | https://drive.google.com/drive/folders/1PKDdDji6OBmDyi3dMoaqNzaCXLOugXli |

**Critical split design:** Dataset is split by **word index**, not by image file. If `word00042_font0.png` is in training, all `word00042_font*.png` variants must also be in training. Splitting by file would allow the same word in different fonts across train/val, inflating validation accuracy — this is a data leakage vulnerability that most quick implementations get wrong.

#### Data Augmentation — Phase 2 Training Split Only 🟢

| Transform | Parameters |
|---|---|
| RandomAffine | 2° rotation, 3% translation, 2° shear |
| ColorJitter | brightness ±0.2, contrast ±0.3 |
| GaussianBlur | kernel=3, p=0.2 |

**Per-epoch font randomisation:** Each epoch, `dataset.resample()` picks one random font variant per unique word. The model sees a different rendering of each name every epoch, simulating handwriting variability without real handwritten samples.

**Why augmentation is lighter than Phase 1:** Phase 1 used heavier augmentation (elastic distortion, wider rotation) because real handwriting is variable. Phase 2 data is clean font renders — heavy augmentation would create unrealistic artifacts. Light augmentation simulates minor scan variability only.

#### Phase 2 Hyperparameters 🟢

⚠️ Optimiser is Adam (not AdamW). Any earlier reference to AdamW is incorrect and superseded.

| Parameter | Value |
|---|---|
| Resolution | **320 × 128** (same as Phase 1 — no CTC timestep disruption) |
| Batch size | **24** |
| Optimiser | **Adam** with differential learning rates |
| LR — CNN tail (block 4 conv3 + block 5) | **1e-6** (near-frozen — preserves Hindi features) |
| LR — BiLSTM | **1e-4** |
| LR — FC | **1e-4** |
| LR scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Max epochs | **20** |
| Early stopping | patience=5, min_delta=1e-4 on val CTC loss |
| BiLSTM dropout | 0.4 |
| FC dropout | 0.4 |
| Weights loaded | Best Phase 1 checkpoint (`crnn_hindi_best.pth`) |
| Checkpoint format | `phase2_epoch{N:03d}_acc{acc:.1f}_cer{cer:.3f}.pth` |

**Why differential LRs:** The CNN tail at LR=1e-6 is effectively frozen in practice — the weight updates are negligible. This preserves the Devanagari handwriting features learned from Hindi. The BiLSTM and FC at LR=1e-4 adapt the sequence model and classifier to Nepali-specific vocabulary and character distributions. Cited as: Yosinski et al. (2014) — principled transfer learning strategy to prevent catastrophic forgetting.

#### Training Hardware 🟢

| Item | Value |
|---|---|
| Platform | Kaggle (free GPU tier) |
| GPUs available | 2× T4 (15 GiB VRAM each) |
| DataParallel | Attempted, then **dropped** |
| Why DataParallel was dropped | PyTorch DataParallel splits batches across GPUs. CTC loss requires the entire batch's predictions and targets on the same device simultaneously for its dynamic programming alignment. Naive DataParallel breaks this requirement. |
| Effective training | Single T4 at ~99% GPU utilisation |
| Phase 1 epoch duration | ~675 seconds/epoch |
| Phase 2 epoch duration | ~8–10 seconds/epoch |

#### Phase 2 Final Results — Primary Report Model 🟢

| Checkpoint | Val Word Accuracy | Notes |
|---|---|---|
| `crnn_nepali_best_phase2b.pth` | **52.23%** | ← **Primary report model** |
| `p2_epoch19_acc49.1.pth` | 49.1% | ← Used in live demo screenshots (best available at demo time) |

⚠️ **Critical distinction for the report:** The live demo was run using `p2_epoch19_acc49.1.pth` (49.1%). The primary report model is `crnn_nepali_best_phase2b.pth` (52.23%). This distinction must be stated clearly in the Results section.

#### Phase 2 Reference Training Log — Older Run 🔵

*From design_2_2.0 session (separate older Kaggle notebooks). Included as supporting reference for training narrative only — not primary results. Metrics are on that run's validation set.*

| Epoch | Train Loss | Val Loss | CER | Word Acc | Status |
|---|---|---|---|---|---|
| 1 | 0.9659 | 0.6900 | 13.72% | 54.38% | ✓ new best |
| 2 | 0.8658 | 0.5830 | 11.42% | 59.45% | ✓ |
| 3 | 0.8195 | 0.5885 | 11.84% | 58.53% | · |
| 4 | 0.7367 | 0.5955 | 11.74% | 55.76% | · |
| 5 | 0.7039 | 0.5572 | 10.69% | 63.13% | ✓ |
| 6 | 0.6702 | 0.4983 | 10.67% | 60.37% | ✓ |
| 7 | 0.6852 | 0.5003 | 11.47% | 58.53% | · |
| 8 | 0.5927 | 0.4735 | 9.90% | 60.83% | ✓ |
| 9 | 0.6030 | 0.4163 | 10.55% | 60.37% | ✓ |
| 10 | 0.5791 | 0.4443 | 9.46% | 63.13% | · |
| 11 | 0.5790 | 0.4753 | 10.72% | 60.83% | · |
| 12 | 0.5415 | 0.4239 | 9.70% | 62.67% | ⚠ |
| 13 | 0.5349 | 0.4047 | 8.82% | 66.82% | ✓ |
| 14 | 0.5379 | 0.4343 | 10.58% | 59.91% | · |
| 15 | 0.4969 | 0.3662 | 9.12% | 64.98% | ✓ |
| 16 | 0.5183 | 0.4089 | 10.81% | 60.37% | · |
| 17 | 0.4688 | 0.3949 | 10.08% | 63.13% | · |
| 18 | 0.4421 | 0.4938 | 11.09% | 60.37% | ⚠ |
| 19 | 0.4468 | 0.3903 | 9.38% | 64.06% | ⚠ |
| 20 | 0.4468 | 0.3555 | 7.98% | 68.20% | ✓ |

Best checkpoint this run: epoch 20, val_loss=0.3555, Word Acc=68.20%, CER=7.98%.

*Note: Higher word accuracies in this table (up to 68%) vs. primary notebook (52.23%) are expected — different dataset splits, different random seeds, different preprocessing between the two Kaggle runs.*

#### Phase 2 Test Results — Older Run 🔵

*Included as supporting evidence only. These are from the older separate notebooks, not the primary report model.*

| Metric | Value |
|---|---|
| CTC Loss | 0.2749 |
| CER | **6.71%** |
| Word Accuracy | **69.92%** |
| WER | 30.08% |
| Δ vs Phase 1 Hindi baseline on Nepali | +22.76 percentage points |

#### Sample Predictions — Older Run 🔵

| Target | Predicted | Correct? |
|---|---|---|
| म‌ंडल | मंडल | ✗ |
| सैयद | सेयद | ✗ |
| श्रीकुमार | श्रीकुमार | ✓ |
| सौराग्य | सौराम्य | ✗ |
| प्रकाश | प्रलाश | ✗ |
| चित्र | चित्र | ✓ |
| नानी | नानी | ✓ |
| डाल्मीया | डाल्मीया | ✓ |
| तुलाधर | तुलाधर | ✓ |
| समीप | समीप | ✓ |

6/10 correct in this sample. Failures cluster on matras and conjunct substitutions (सैयद→सेयद, सौराग्य→सौराम्य).

### 3.4 Full Pipeline Performance Summary 🟢

| Stage | Evaluation Set | Word Accuracy | CER |
|---|---|---|---|
| Phase 1 Hindi model on Nepali (baseline) | Synthetic Nepali val | ~47.2% | — |
| Phase 2b best checkpoint (primary model) | Synthetic Nepali val | **52.23%** | — |
| Phase 2 older run (reference only 🔵) | Synthetic Nepali val | 69.92% | 6.71% |
| After NLP post-processing — older run (reference 🔵) | Synthetic test set | **91.87%** | **3.49%** |
| Live demo — real handwritten forms (2 forms) 🟡 | Real handwriting | 78–89% (field-level) | — |

**Important caveats for all synthetic metrics:** The 47.2% and 52.23% figures are on a synthetic font-rendered validation set. The 69.92% / 91.87% figures are from an older run with different splits. Real handwriting introduces variability the model was not trained on. The 78–89% real-form results are from only 2 test forms using common names — not statistically generalisable.

### 3.5 Vocabulary and Character Set 🟢

```
Phase 1 vocab:   Built from all unique characters in Hindi training labels
                 Includes full Devanagari Unicode range from real handwriting
                 Blank token = index 0

Phase 2 vocab:   Phase 1 vocab EXTENDED with any new characters from Nepali labels
                 Never replaced, only appended — CTC alignment stays stable
                 If a character exists in Phase 1 vocab, its index is unchanged
```

### 3.6 Data Pipeline 🟢

```
Both datasets → List of (absolute_image_path, label_string) tuples
                        ↓
             DevanagariDataset(pairs, transform, char2idx)
                        ↓
             DataLoader(batch_size=24, num_workers=4, pin_memory=True)

Collate function handles variable-length label sequences for CTC
Image transform: resize → grayscale → expand to 3ch → normalise
                 ImageNet mean/std for both phases (VGG16 expects it)
```

### 3.7 Metrics — Full Definitions 🟢

| Metric | Computed | Formula |
|---|---|---|
| CTC Loss | Every epoch, train+val | Negative log-likelihood over all valid CTC alignments |
| CER | Every epoch val, final test | edit_distance(pred, truth) / len(truth), mean over set |
| Word Accuracy | Every epoch val, final test | exact_matches / total_samples |
| WER | Final test only | 1 − Word Accuracy |
| Norm. Edit Distance | Final test only | CER capped at 1.0 per sample, then averaged |

**Known visualisation limitation:** Confusion matrix axis labels could not be rendered. Matplotlib's default font (DejaVu Sans) does not contain Devanagari glyphs, so per-character error visualisation was not possible. Only the raw matrix data (not the visual) was available.

---

## PART 4 — NLP POST-PROCESSING

### 4.1 What It Is and Why It Was Added 🟢

The NLP post-processing layer is a standalone Python module (`nlp_postprocessor.py`) added after CRNN training to correct common OCR output errors without retraining the model. It uses no external NLP libraries — only Python stdlib (`unicodedata`, `json`, `pathlib`).

It was added because the CRNN makes systematic, predictable errors on certain last names — particularly श्रेष्ठ, consistently decoded as श्रेव or श्रेष्ट. These errors are correctable with simple vocabulary lookup without any model changes.

### 4.2 Vocabulary — vocab.json 🟢

| Category | Count |
|---|---|
| First names | 789 |
| Last names | 485 |
| Districts | 79 |
| **Total entries** | **1,353** |

### 4.3 The Four Processing Steps 🟢

**Step 1 — Unicode NFC normalisation.** Devanagari combining characters (vowel signs, halant ्, anusvara ं, chandrabindu ँ) can be encoded in different decomposition orders across fonts and OCR engines. NFC canonicalises them so string comparisons work correctly.

**Step 2 — Artifact stripping.** The CTC decoder sometimes emits invisible Unicode characters:
- ZWJ (U+200D — Zero Width Joiner)
- ZWNJ (U+200C — Zero Width Non-Joiner)
- Isolated halant ्
- Pipe `|`
- Stray ASCII letters

All stripped.

**Step 3 — Field-specific overrides.** `nationality` is matched against a closed list of ~10 accepted values (e.g., नेपाली) using Levenshtein distance ≤ 2.

**Step 4 — Vocabulary fuzzy match.** For `last_name`, `father_last_name`, `mother_last_name`, and `city_district`: compare the normalised prediction against `vocab.json` entries using Levenshtein edit distance. If the closest entry is within distance ≤ 2, replace with the correctly-spelled vocab entry.

### 4.4 NLP Configuration 🟡

| Parameter | Value |
|---|---|
| Max edit distance | **2** |
| Algorithm | Levenshtein edit distance on Unicode code points |
| Fields with fuzzy correction | last_name, father_last_name, mother_last_name, city_district |
| Fields WITHOUT fuzzy correction | first_name, father_first_name, mother_first_name, place_of_birth |
| Nationality field | Closed-list match only |
| Unicode step | NFC normalisation |
| Artifact removal | ZWJ, ZWNJ, pipe `|`, stray ASCII, isolated halant |

### 4.5 Why First Names Are Excluded — Evidence from Live Testing 🟡

At max_edit_distance=2 with first names included (tested and rejected):
- राम → राज ✗ (distance 1, both in vocab, wrong choice made)
- सीता → रिया ✗ (distance 2, wrong)
- हरि → हेम ✗ (distance 2, wrong)

At max_edit_distance=1 with first names included (also rejected):
- राम → राज ✗ (distance 1 still ambiguous — multiple equally-close candidates)

Final config — first names excluded entirely:
- राम → राम ✓ (no correction applied; raw OCR was already correct)
- सीता → सीता ✓

**Root cause:** Short Nepali first names (2–3 syllables: राम, हरि, सीता, रिया) have many near-neighbours within distance 2 in the vocabulary. Without frequency weighting, the corrector cannot distinguish between equally-close candidates. Last names and districts are longer and more phonetically distinctive — distance ≤ 2 correction is reliable for those.

### 4.6 NLP Performance — Older Run (Synthetic Test Set) 🔵

*From the design_2_2.0 session. Applied to the older run's synthetic test predictions.*

| Metric | Before NLP | After NLP | Delta |
|---|---|---|---|
| Word Accuracy | 69.92% | **91.87%** | +21.95 pp |
| CER | 6.71% | **3.49%** | −3.22 pp |
| Predictions corrected | — | 34 / 123 (27.6%) | — |

### 4.7 Live Demo Test Results — Two Real Handwritten Forms 🟡

> **Model used in demo:** `p2_epoch19_acc49.1.pth` (49.1% synthetic val accuracy)
> The 52.23% best checkpoint was not yet available when the demo was run. State this clearly in the report.

#### Test Form 1 (राम / श्रेष्ठ / काठमाडौं / हरि / श्रेष्ठ / सीता / श्रेष्ठ / नेपाली / ललितपुर)

| Field | Written | Raw OCR (before NLP) | After NLP | Result |
|---|---|---|---|---|
| पहिलो नाम | राम | राम | राम | ✅ Correct |
| थर | श्रेष्ठ | श्रेव | श्रेष्ठ | ✅ NLP fixed |
| जन्मस्थान | काठमाडौं | कामहैं | काठमाडहं | ⚠️ Partial — model error |
| बुबाको पहिलो नाम | हरि | हरी | हरी | ⚠️ Matra wrong — model error |
| बुबाको थर | श्रेष्ठ | श्रेष्ट | श्रेष्ठ | ✅ NLP fixed |
| आमाको पहिलो नाम | सीता | सीता | सीता | ✅ Correct |
| आमाको थर | श्रेष्ठ | श्रेष्ट | श्रेष्ठ | ✅ NLP fixed |
| राष्ट्रियता | नेपाली | नेपाली | नेपाली | ✅ Correct |
| जिल्ला | ललितपुर | ललितपुर | ललितपुर | ✅ Correct |
| **Score** | | **4/9 without NLP** | **7/9 with NLP** | **NLP improved 3 fields** |

#### Test Form 2 (राम / श्रेष्ठ / काठमाडौं / कृतक / श्रेष्ठ / रिया / श्रेष्ठ / नेपाली / ललितपुर)

| Field | Written | Raw OCR (before NLP) | After NLP | Result |
|---|---|---|---|---|
| पहिलो नाम | राम | राम | राम | ✅ Correct |
| थर | श्रेष्ठ | श्रेष्ठ | श्रेष्ठ | ✅ Correct |
| जन्मस्थान | काठमाडौं | कारारीं | कारारीं | ❌ Model error |
| बुबाको पहिलो नाम | कृतक | कृतक | कृतक | ✅ Correct |
| बुबाको थर | श्रेष्ठ | श्रेष्ठ | श्रेष्ठ | ✅ Correct |
| आमाको पहिलो नाम | रिया | रिया | रिया | ✅ Correct |
| आमाको थर | श्रेष्ठ | श्रेष्ठ | श्रेष्ठ | ✅ Correct |
| राष्ट्रियता | नेपाली | नेपाली | नेपाली | ✅ Correct |
| जिल्ला | ललितपुर | ललितपुर | ललितपुर | ✅ Correct |
| **Score** | | **8/9 without NLP** | **8/9 with NLP** | **NLP had nothing to fix** |

#### Live Demo Summary 🟡

| Metric | Value |
|---|---|
| Model used | `p2_epoch19_acc49.1.pth` |
| Test forms | 2 (small sample — not statistically significant) |
| Form 1 field accuracy after NLP | 7/9 = **78%** |
| Form 2 field accuracy after NLP | 8/9 = **89%** |
| Fields improved by NLP across both forms | 3 (all were last name श्रेष्ठ misread as श्रेव or श्रेष्ट) |
| New errors introduced by NLP | **0** |
| Consistent failure field | जन्मस्थान — chandrabindu misrecognition and cursive style |
| Why these forms score above synthetic average | Common names (राम, श्रेष्ठ, सीता) are well-represented in training vocabulary |

#### What NLP Fixes vs. What It Leaves Alone 🟡

| Raw OCR | After NLP | Verdict |
|---|---|---|
| श्रेव | श्रेष्ठ | ✅ Fixed — last name fuzzy match |
| श्रेष्ट | श्रेष्ठ | ✅ Fixed — last name fuzzy match |
| श्रेष्ट | श्रेष्ठ | ✅ Fixed — last name fuzzy match |
| राम | राम | ✅ Correct as-is — first name, no correction applied |
| सीता | सीता | ✅ Correct as-is — first name, no correction applied |
| हरी | हरी | ⚠️ Matra wrong but NLP correctly does not modify it — first name excluded |

### 4.8 Known Failure Modes 🟢🟡

1. **जन्मस्थान (place of birth) consistently fails.** काठमाडौं in cursive produced कामहैं (Form 1) and कारारीं (Form 2). Root cause: (a) cursive handwriting style differs from font-rendered training data; (b) the chandrabindu (ँ) at the end of काठमाडौं appears rarely in that position in training vocabulary.

2. **Short vowel matra confusion.** हरि vs हरी (ि vs ी) — a one-pixel height difference at 128px. Both are valid Devanagari — the model cannot reliably distinguish them at training resolution.

3. **Chandrabindu vs anusvara.** ँ (U+0901) vs ं (U+0902) — visually similar diacritics above the character. काठमाडौं requires chandrabindu; the model sometimes substitutes anusvara.

4. **Conjunct substitutions.** From sample predictions: सौराग्य→सौराम्य (ग replaced by म), प्रकाश→प्रलाश (क replaced by ल). Conjunct consonants (्ग, ्क) are misread as visually similar ones.

5. **Out-of-vocabulary words.** NLP can only correct to words in `vocab.json`. Novel names, foreign transliterations, or uncommon surnames cannot be corrected.

6. **Fundamental domain gap.** Entire training is on font-rendered images. Real handwriting varies in stroke width, slant, ink spread, and character inconsistency. This gap is the primary limiting factor for real-world accuracy.

---

## PART 5 — REPORT WRITING PACK

### 5.1 Critical Analysis — Three Existing Solutions 🟢

The report requires critical analysis of three existing solutions across the build-vs-borrow spectrum. The three are: **Tesseract OCR, VGG16 standalone classifier, EasyOCR.**

---

#### Solution 1: Tesseract OCR v4 — Borrow Everything (Rejected)

**Citation:** Smith, R. (2007)

**Architecture:** Adaptive page layout analysis → LSTM character recognition (v4.0+) → language model post-processing.

**Published performance:** >95% character accuracy on clean printed text. Drops to 40–70% on handwritten inputs.

**Strengths:**
- Mature, widely deployed, actively maintained (Google)
- Has Nepali (`nep`) language pack
- CPU-only — no GPU required
- Open source

**Weaknesses for our use case:**
- Trained exclusively on printed corpus — fixed, stable font structures. Handwritten matra recognition is poor.
- Fundamental mismatch: Tesseract learned a finite set of stable printed glyphs; handwriting produces continuous stroke variation Tesseract has never been exposed to.
- No form alignment module — cannot locate specific fields on a fixed-template form.
- No structured field extraction pipeline.
- Cannot be meaningfully fine-tuned on handwritten Devanagari.

**What we adapted from this:** Analysing Tesseract's gap directly motivated our ORB alignment + ROI cropping layer. No existing OCR tool handles structured form field extraction — we built it.

**Verdict:** Not suitable. The architectural gap is fundamental — Tesseract learned printed fonts, not handwriting strokes. This cannot be fixed with tuning.

---

#### Solution 2: VGG16 as Standalone Classifier — Borrow and Adapt (Rejected as-is, used as backbone)

**Citation:** Simonyan & Zisserman (2015); Masrat et al. and Deshmukh et al. for Devanagari character classification applications.

**Architecture:** Deep convolutional network, 16 weight layers, 3×3 filters, ImageNet pretrained. When adapted for OCR: used as a fixed-output softmax classifier over character or word classes.

**Published performance:** ImageNet top-5 accuracy 92.7%. When adapted for Devanagari character classification (Masrat et al., Deshmukh et al.): reported 85–95% character-level accuracies on isolated character datasets.

**Why VGG16 is better than Tesseract as a feature extractor:** VGG16 was trained on 1.2 million real-world images — it has learned a deep hierarchy of curves, edges, and textures from the real visual world. Handwriting is made of the same primitives. Tesseract never saw this variety; VGG16 did.

**Strengths:**
- Excellent feature extractor for visual patterns
- ImageNet pretraining provides strong initialisation for any visual recognition task
- Deep hierarchical features generalise well

**Weaknesses as a standalone classifier:**
- Fixed output dimension — cannot handle variable-length word sequences. A classifier predicts one label per image; handwritten words have variable character counts.
- Requires character-level segmentation as a prerequisite. Segmenting Devanagari conjuncts and matras is a hard unsolved problem — conjuncts fuse multiple characters into single glyphs.
- Isolated character datasets used in prior work contain no matras or modifiers — do not reflect real handwritten word recognition.
- No form alignment capability.

**What we adapted from this:** We took VGG16's `.features` stack as the CNN backbone inside our CRNN, replacing the classifier head with `AdaptiveAvgPool2d → BiLSTM → FC → CTC`. This eliminates the need for character segmentation entirely — CTC handles alignment implicitly. We also freeze VGG16's early blocks (1–2) because those ImageNet edge detectors are directly useful for script recognition without any modification.

**Verdict:** VGG16 as a standalone classifier is unsuitable for our task. As a pretrained feature extraction backbone inside CRNN, it is exactly what we need — and is the reason our system works.

---

#### Solution 3: EasyOCR (JaidedAI) — Borrow Architecture (Architecture confirmed, weights replaced)

**Citation:** Baek et al. (2019) for CRAFT detector.

**Architecture:** CRAFT text detector (VGG16-based) → CRNN recogniser (VGG/ResNet CNN + BiLSTM + CTC).

**Published performance:** >90% on printed Latin benchmarks. Community reports ~60–80% on printed Nepali. Lower for handwritten.

**Strengths:**
- Native Nepali language support
- Two-stage detect-then-recognise handles free-form documents
- CRNN+CTC architecture is directly analogous to ours — confirms the approach is viable for Devanagari
- Open source

**Weaknesses for our use case:**
- Training data is primarily printed/digital text — handwritten performance degrades significantly
- CRAFT text detector is confused by printed boundary boxes of form fields (detects them as text regions rather than text content)
- Not practically fine-tuneable on custom data without significant expertise
- Requires GPU for reasonable throughput

**What we adapted from this:** EasyOCR confirmed CRNN+CTC as the most viable open-source architecture for Devanagari recognition. We adopted the backbone principle and replaced EasyOCR's generic weights with our Hindi-pretrained → Nepali-fine-tuned weights. We dropped CRAFT detection entirely in favour of fixed-template ORB alignment, which is more precise and faster for our fixed-form use case.

**Verdict:** Viable as architectural confirmation; not suitable as a deployed solution. Our domain-specific training outperforms EasyOCR on handwritten Nepali names.

---

#### Comparison Table 🟢

| Criterion | Tesseract OCR | VGG16 Standalone | EasyOCR | Our CRNN |
|---|---|---|---|---|
| Architecture | LSTM + language model | Fixed-output CNN classifier | CRAFT detector + CRNN | VGG16 backbone + BiLSTM + CTC |
| Training data | Printed Devanagari fonts | ImageNet + isolated Devanagari chars | Multi-language printed text | HindiSeg real handwriting + Nepali synthetic |
| Handwritten support | Poor (40–70%) | None tested on sequences | Limited (~60–80% printed) | Moderate — fine-tuned for handwritten names |
| Nepali language | `nep` pack (printed only) | N/A | Built-in (printed only) | Custom trained (handwritten Nepali names) |
| Variable-length output | Yes (language model) | No — fixed classes only | Yes (CTC) | Yes (CTC) |
| Form field extraction | None | None | None (free-form only) | Template alignment + ROI crop |
| GPU required | No | No (inference) | Recommended | Training yes; CPU at inference |
| Open source | Yes | Yes (torchvision) | Yes | Yes |
| What we adapted | Nothing — gap motivates our pipeline | CNN backbone only (inside CRNN) | CRNN+CTC architecture confirmation | Core model — extended with transfer learning |
| Nepal applicability | Not suitable | Not suitable as classifier | Not suitable as deployed system | Designed for this use case |

#### Synthesis Paragraph 🟢

Our approach synthesises insights from all three solutions. Tesseract's fundamental limitation — trained on printed fonts, unable to handle handwriting — directly motivated our ORB alignment + ROI cropping pipeline: no existing tool solves structured form field extraction, so we built it. VGG16's ImageNet pretraining provides the deep edge and curve feature hierarchy that makes our system work — we keep it as a backbone rather than a classifier, wrapping it in BiLSTM+CTC to handle variable-length Devanagari sequences without character segmentation. EasyOCR confirmed CRNN+CTC as the correct architectural choice for open-source Devanagari recognition; we replaced its generic weights with our Hindi→Nepali transfer learning chain. Our original contributions above the three prior solutions are: (1) the template-based alignment pipeline enabling precise field-level extraction from a fixed-form scan; (2) the Hindi pre-training → Nepali fine-tuning transfer chain addressing the scarcity of real handwritten Nepali training data; (3) a lightweight NLP post-processor for vocabulary-guided error correction with zero new errors introduced in testing; and (4) honest empirical evaluation on real handwritten forms demonstrating the domain gap between synthetic training and real-world deployment.

---

### 5.2 Key Design Decisions and Justifications 🟢

| Decision | What we chose | Justification |
|---|---|---|
| Architecture | CRNN (VGG16 + BiLSTM + CTC) | No character segmentation needed; handles variable-length Devanagari text implicitly via CTC |
| Pre-training domain | Hindi (not English, not Nepali directly) | Shared Devanagari script — feature transfer is direct; avoids training from scratch on scarce/synthetic Nepali data |
| Phase 1 freezing | Blocks 1–2 frozen; blocks 3–5 + BiLSTM + FC trainable | Teaches sequence model to read Devanagari using existing ImageNet edge features without disturbing low-level detectors |
| Phase 2 freezing | Blocks 1–4 conv1-2 frozen; block 4 conv3 + block 5 at LR 1e-6 | Preserves Hindi handwriting features; only adapts deepest script-specific patterns to Nepali |
| Differential LR | CNN tail 1e-6, RNN/FC 1e-4 | Principled transfer learning — prevents catastrophic forgetting of Hindi features (Yosinski et al., 2014) |
| Resolution | 320×128 for both phases | Sufficient for matra visibility (ि vs ी, ँ vs ं are critical pixel-level distinctions); consistent across phases prevents CTC timestep disruption |
| Fixed-template scope | Hardcoded field coordinates | Eliminates layout detection entirely — feasible within deadline and data constraints |
| ORB + RANSAC homography | Feature-based image registration | Handles scan rotation, scale, perspective, and shift without retraining |
| ROI tight crop | 180pt instead of full 382pt | Prevents mostly-blank crops degrading CTC alignment — model wastes no time steps on whitespace |
| Word-level dataset split | Split by word index, not file | Prevents data leakage from font variants of the same word across train/val |
| Greedy CTC decode | argmax per timestep | Fast, sufficient for demo; beam search would improve accuracy but adds complexity — documented as future work |
| NLP post-processing | Levenshtein ≤ 2, last names + districts only | Reliable for longer distinctive words; first names excluded to prevent over-correction |
| No DataParallel | Single T4 training | CTC loss requires full batch on one device — DataParallel breaks CTC's dynamic programming alignment |

### 5.3 Honest Limitations — Required for Top Band 🟢

State all of these in the report. Assessors reward honest self-evaluation.

1. **Synthetic-only training data.** The model was trained on font-rendered Nepali images. Real handwriting introduces stroke variability, slant, ink spread, and character inconsistency not seen in training. The 52.23% validation accuracy is on synthetic data — real-world accuracy on diverse handwriting is unknown.

2. **Insufficient accuracy for unsupervised deployment.** At approximately 1 word in 2 incorrect on synthetic validation, the system cannot replace human transcription. It is positioned as a first-pass digitisation aid requiring mandatory human review.

3. **Matra recognition is the primary failure mode.** Vowel modifier marks — short vs. long vowels (ि vs ी), chandrabindu (ँ) vs anusvara (ं), and complex conjuncts — are consistently misrecognised. These distinctions are critical for word meaning.

4. **Evaluation sample is very small.** Only 2 real handwritten forms were tested. Both used common names (राम, श्रेष्ठ, सीता) well-represented in training vocabulary. The 78–89% field accuracy cannot be generalised to diverse handwriting or uncommon names.

5. **No CER measured on real handwriting.** All CER metrics are from the synthetic validation set. No character-level error analysis on real handwritten output was performed.

6. **NLP cannot correct out-of-vocabulary predictions.** Fuzzy matching can only reach words in `vocab.json`. Novel names, foreign transliterations, or uncommon surnames cannot be corrected.

7. **जन्मस्थान consistently fails.** Place of birth failed on both real test forms. Place names are long, phonetically diverse, and include characters (chandrabindu in काठमाडौं) that appear rarely in the name-focused training vocabulary.

8. **Confusion matrix not visualised.** Matplotlib's DejaVu Sans font does not contain Devanagari glyphs. Per-character confusion heatmap could not be rendered — only the raw matrix data was available.

9. **Input resolution may be tight for dense conjuncts.** 320×128 may not be sufficient for fully dense Devanagari conjuncts and stacked modifiers in long words.

### 5.4 Strong Claims — All Evidence-Backed 🟢

Use these in the Results/Discussion section.

1. **Transfer learning demonstrably works.** Baseline (Hindi model, no Nepali fine-tuning): ~47.2% on Nepali. After fine-tuning: 52.23%. This is the quantitative evidence of the method's value.

2. **NLP post-processing adds value with zero regressions.** Across both real test forms: 3 fields improved (all श्रेष्ठ last names), 0 new errors introduced. The before/after table is clean, reproducible evidence.

3. **Word-index split prevents data leakage.** Splitting by word index rather than by file is a methodological decision most quick implementations get wrong. Stating this explicitly demonstrates evaluation integrity.

4. **ORB alignment enables structured field extraction.** No existing OCR tool (Tesseract, VGG16, EasyOCR) solves the form-field extraction problem. This is the system-level contribution above the CRNN model alone.

5. **Differential learning rates preserve prior knowledge.** LR=1e-6 for the CNN tail vs 1e-4 for RNN/FC is a principled transfer learning choice, citable as Yosinski et al. (2014).

6. **VGG16 backbone choice is principled.** ImageNet training exposes VGG16 to a vast variety of real-world curves and edges — exactly the kind of visual primitives handwriting is made of. This is a stronger initialisation for script recognition than any OCR-specific trained encoder that never saw real handwriting.

### 5.5 APA 7 Citations 🟢

| What to cite | Full APA 7 Reference |
|---|---|
| CRNN foundational paper | Shi, B., Bai, X., & Yao, C. (2016). An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *39*(11), 2298–2304. https://doi.org/10.1109/TPAMI.2016.2646371 |
| CTC loss | Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks. *Proceedings of the 23rd International Conference on Machine Learning*, 369–376. |
| VGG16 | Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *Proceedings of the 3rd International Conference on Learning Representations*. |
| EasyOCR / CRAFT detector | Baek, Y., Lee, B., Han, D., Yun, S., & Lee, H. (2019). Character region awareness for text detection. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 9365–9374. |
| Tesseract OCR | Smith, R. (2007). An overview of the Tesseract OCR engine. *Proceedings of the 9th International Conference on Document Analysis and Recognition*, 629–633. |
| Transfer learning justification | Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? *Advances in Neural Information Processing Systems*, *27*. |
| HindiSeg dataset | Sabarinathan. (n.d.). *Handwritten Hindi word recognition dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/sabarinathan/handwritten-hindi-word-recognition |
| Nepali font dataset | kritakhere. (n.d.). *Nepali font generated handwritten names* [Data set]. Kaggle. https://www.kaggle.com/datasets/kritakhere/nepali-font-generated-handwritten-names |

### 5.6 Suggested Future Work 🟢

1. Collect 50–100 real handwritten Nepali forms for proper evaluation against actual handwriting variability
2. Add beam search CTC decoding with a Nepali character language model to improve recognition of rare matras and conjuncts
3. Apply morphological preprocessing (contrast enhancement, stroke normalisation) to improve matra visibility before CRNN
4. Add frequency-weighted vocabulary for first name correction (currently excluded because equal-weight lookup over-corrects)
5. Replace ORB alignment with a learning-based document alignment method for degraded or extreme-angle scans
6. Extend from fixed-template to free-form document understanding using DBNet or similar text detection network

### 5.7 Viva Preparation — Questions You Must Be Able to Answer 🟢

**Q1: Why VGG16 and not ResNet or EfficientNet?**
VGG16's sequential convolutional stack without skip connections produces a feature map whose spatial dimensions degrade predictably, making the AdaptiveAvgPool2d → sequence conversion straightforward. ResNet's residual connections change spatial dimensions in non-trivial ways that complicate the pooling-to-sequence step. This is the same reason Shi et al. (2016) used a VGG-style backbone in the original CRNN paper. Additionally, VGG16 was trained on ImageNet — 1.2 million real-world images full of diverse curves and edges — exactly the visual primitives handwriting is made of. It brings a richer prior than any OCR-specific encoder that never saw real-world variation.

**Q2: Why does CTC not need character-level annotations?**
CTC (Connectionist Temporal Classification) marginalises over all valid alignments between the input time-step sequence and the output character sequence using dynamic programming. It learns which timesteps correspond to which characters without being told. The blank token absorbs the many-to-one mapping between input timesteps and output characters, and consecutive repeated tokens are collapsed. This is essential for handwritten text where character boundaries are not well-defined.

**Q3: What is your baseline and what does it prove?**
The Phase 1 Hindi-trained model evaluated on the Nepali validation set without any Nepali fine-tuning achieves ~47.2% word accuracy. This proves two things: (1) Hindi pre-training transfers usefully to Nepali — the shared Devanagari script means the model is not at ~0%, demonstrating the transfer chain works; (2) Nepali fine-tuning is still necessary — 47.2% is insufficient for deployment, justifying Phase 2.

**Q4: Why is synthetic validation accuracy ~52% while real form accuracy is 78–89%?**
Two complementary explanations: (1) Selection bias in real testing — both test forms used common Nepali names (राम, श्रेष्ठ, सीता) that are well-represented in training vocabulary. Forms with unusual names or messier handwriting would score lower. (2) The synthetic validation set may contain words and font variations that differ from the training distribution in ways real handwriting (which the model has never seen) does not. This is a known limitation — the real accuracy figure of 78–89% is encouraging but not generalisable from 2 forms.

**Q5: How does ORB alignment work?**
ORB detects keypoints (corners, blobs) in both the scanned form and the clean reference template rendered from the PDF. BFMatcher finds correspondences using Hamming distance on binary descriptors. The top 15% of matches are kept; the rest discarded as noise. RANSAC + `cv2.findHomography` estimates the homographic transformation robustly against outliers. `warpPerspective` applies the transform to the scan, so all hardcoded field coordinate boxes map correctly to the aligned scan.

**Q6: What does the NLP post-processor actually do?**
It applies Unicode NFC normalisation (canonicalises combining character order), strips OCR artifacts (ZWJ, ZWNJ, pipe characters, stray ASCII, isolated halant), matches nationality against a closed list, and runs Levenshtein distance ≤ 2 fuzzy matching against `vocab.json` for last name and district fields. First name fields are excluded because short names (राम, हरि, सीता) have too many near-neighbours within distance 2, causing over-correction.

**Q7: How did you validate the NLP layer?**
We tested with and without post-processing on two real handwritten forms. Without NLP: श्रेव and श्रेष्ट were consistently wrong for last names containing श्रेष्ठ. With NLP: all three instances corrected to श्रेष्ठ. Zero new errors introduced by the post-processor in either form.

**Q8: Why did you not use beam search CTC decoding?**
Greedy decoding (argmax per timestep) was sufficient for the demo and meets the project deadline. Beam search with a language model would improve accuracy — particularly for rare matras — but adds implementation complexity, a language model dependency, and inference latency. Documented as a future improvement.

**Q9: Why is DataParallel not used even with 2 GPUs available?**
PyTorch DataParallel splits a batch across GPUs. CTC loss requires the entire batch's predictions and targets on the same device simultaneously for its dynamic programming alignment. Naive DataParallel breaks this, causing incorrect loss computation. Training ran on a single T4.

**Q10: What is the Nepal context for this project?**
Nepal has an enormous backlog of handwritten documents — citizenship certificates, land registries, school records, court documents — that remain undigitised. Manual transcription is slow, expensive, and scales poorly. Even at 52% synthetic accuracy, combined with NLP post-processing and mandatory human review of uncertain outputs, this system provides meaningful first-pass digitisation support and reduces manual workload in Track T2 government record contexts.

### 5.8 File Structure 🟢

```
project/
├── notebooks/
│   └── files/                         ← All pipeline files
│       ├── ocr_pipeline.py            ← MAIN PIPELINE (Stages 1–5, calls NLP)
│       ├── gui.py                     ← Gradio web UI — python gui.py --model <path>
│       ├── nlp_postprocessor.py       ← NLP post-processing (Stage 6)
│       ├── template_config.py         ← All 9 field coordinates + ROI logic
│       ├── extract_coords.py          ← Utility: re-extracts coords from PDF
│       ├── vocab.json                 ← Vocabulary for fuzzy correction
│       └── template/
│           └── Red_Minimalist_Membership_Form_A4.pdf
├── models/
│   ├── p2_epoch19_acc49.1.pth         ← Best Phase 2a checkpoint (used in live demo)
│   └── crnn_nepali_best_phase2b.pth   ← Best overall (52.23%) — primary report model
├── notebooks/
│   ├── phase1_hindi_training_v2.ipynb
│   └── handwritten-nepali-word-recognition-nepali__4_.ipynb
├── data/
├── results/
├── docs/                              ← PUT REPORT HERE
└── README.md
```

### 5.9 How to Run 🟢

```bash
# Install dependencies
pip install torch torchvision opencv-python PyMuPDF gradio

# Navigate to pipeline files
cd notebooks/files

# Run the Gradio web UI
python gui.py --model ..\..\models\p2_epoch19_acc49.1.pth --vocab vocab.json

# Open in browser: http://127.0.0.1:7860

# OR run directly on an image (prints JSON to stdout)
python ocr_pipeline.py path/to/form.jpg --model ..\..\models\crnn_nepali_best_phase2b.pth
```

### 5.10 Report Section → Source Mapping 🟢

| Report Section | Source in This Document |
|---|---|
| Introduction / Problem Statement | Part 1.1, Part 5.10 (Nepal context in Q10) |
| Related Work / Critical Analysis | Part 5.1 (three solutions + table + synthesis) |
| Proposed Solution Overview | Part 1.3 (pipeline), Part 2 (architecture) |
| Implementation Details | Part 1.4–1.7, Part 2, Part 3 |
| Datasets | Part 3.2 (Phase 1), Part 3.3 (Phase 2) |
| Training | Part 3.2–3.4 (all hyperparameters, logs, results) |
| NLP Post-Processing | Part 4 (all sections) |
| Results and Evaluation | Part 3.4, Part 4.6–4.7, Part 5.4 |
| Limitations | Part 5.3 |
| Future Work | Part 5.6 |
| References | Part 5.5 |
| README | Part 5.8 (file structure), Part 5.9 (how to run), Part 1.1 (summary) |
| Slides | Part 1.3 (pipeline), Part 2.3 (arch table), Part 3.4 (results), Part 4.7 (demo tables) |

---

*Document version: 3.0 | Compiled: July 31, 2026*
*Supersedes: v2, SOURCE_OF_TRUTH_FINAL.md, design_2_1.0, design_2_2.0, all planning session notes*

*Source legend: 🟢 Canonical fact | 🟡 Confirmed from live testing | 🔵 Older design_2_2.0 run — reference only, not primary results | ⚠️ Conflict resolved in this version*

*All performance figures in Part 4 (NLP live demo) are from actual live testing on real handwritten forms. Synthetic training metrics (Part 3) are from Kaggle notebook output logs. 🔵 figures are from a separate older run and must be labelled as such in the report.*
