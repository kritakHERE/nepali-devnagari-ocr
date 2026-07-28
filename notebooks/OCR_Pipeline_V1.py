import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import json
from pathlib import Path
import torchvision.models as models

# ---------------------- 1. CRNN Architecture (must match training) ----------------------
class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.cnn = vgg.features
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        f = self.cnn(x)
        f = self.pool(f)
        b, c, h, w = f.size()
        f = f.squeeze(2)
        f = f.permute(2, 0, 1)
        r, _ = self.rnn(f)
        out = self.fc(r)
        return torch.nn.functional.log_softmax(out, dim=2)


# ---------------------- 2. CTC Decoder ----------------------
def ctc_decode(output, idx2char):
    preds = output.argmax(dim=2).permute(1, 0)
    decoded = []
    for seq in preds:
        chars, prev = [], None
        for idx in seq.tolist():
            if idx != prev and idx != 0:
                chars.append(idx2char.get(idx, ''))
            prev = idx
        decoded.append(''.join(chars))
    return decoded


# ---------------------- 3. Load Model (FORCED CPU) ----------------------
def load_model(checkpoint_path, vocab_path):
    device = torch.device('cpu')  # <-- FORCE CPU! No overheating.
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.rstrip('\n') for line in f]
    idx2char = {i: c for i, c in enumerate(vocab)}
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    num_classes = ckpt['num_classes']
    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, idx2char, device


# ---------------------- 4. Recognize a single word (supports Devanagari) ----------------------
def predict_word(model, idx2char, word_img_np, device):
    # word_img_np is a grayscale numpy array (0-255)
    pil_img = Image.fromarray(word_img_np).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        pred = ctc_decode(out, idx2char)[0]
    return pred


# ---------------------- 5. Segment words from a cropped field ----------------------
def segment_words(region_binary):
    # region_binary: white text on black background (thresholded)
    contours, _ = cv2.findContours(region_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    word_imgs = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:  # filter noise
            word_img = region_binary[y:y+h, x:x+w]
            word_imgs.append((x, word_img))
    word_imgs.sort(key=lambda t: t[0])  # left-to-right order
    return [img for _, img in word_imgs]


# ---------------------- 6. Main Form OCR Pipeline ----------------------
def process_form(image_path, model, idx2char, device):
    # --- Define fixed field bounding boxes (ADJUST THESE TO YOUR FORM) ---
    # These coordinates assume the form is scanned at a standard resolution.
    # You can also use label-detection (Tesseract) instead of fixed boxes.
    FIELDS = {
        'name':        (100, 150, 400, 200),   # (x1, y1, x2, y2)
        'surname':     (100, 250, 400, 300),
        'district':    (100, 350, 400, 400),
        'father_name': (100, 450, 400, 500),
        'mother_name': (100, 550, 400, 600),
    }
    
    # Read and preprocess image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Binary inverse (white text on black background for contour detection)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    result = {}
    for field_name, (x1, y1, x2, y2) in FIELDS.items():
        roi_binary = thresh[y1:y2, x1:x2]
        words = segment_words(roi_binary)
        if words:
            preds = [predict_word(model, idx2char, w, device) for w in words]
            text = ' '.join(preds)
        else:
            text = ''
        result[field_name] = text
    
    return result


# ---------------------- 7. Run it ----------------------
if __name__ == '__main__':
    # Set your file paths
    MODEL_PATH = 'crnn_nepali_best.pth'   # Path to your .pth file
    VOCAB_PATH = 'vocab_phase2.txt'       # Path to your vocab file
    IMAGE_PATH = 'filled_form.jpg'        # Path to your scanned form

    # Load model (CPU only)
    model, idx2char, device = load_model(MODEL_PATH, VOCAB_PATH)
    print("Model loaded on CPU. Ready for inference.")

    # Process the form
    output_data = process_form(IMAGE_PATH, model, idx2char, device)
    
    # Save as JSON
    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\n--- Output JSON ---")
    print(json.dumps(output_data, ensure_ascii=False, indent=2))