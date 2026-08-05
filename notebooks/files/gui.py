"""
Gradio GUI for the Nepali OCR pipeline.

Wraps OCRPipeline so you can drag-and-drop a scanned/photographed filled
form and see all 9 extracted fields directly in a browser tab, instead of
reading terminal JSON output.

USAGE
-----
    python gui.py --model ..\\..\\models\\p2_epoch19_acc49.1.pth

Then open the printed local URL (usually http://127.0.0.1:7860) in your
browser. Upload a photo/scan of a FILLED form, click Submit, and the
recognized text for each field will appear.

NOTE: the underlying model+pipeline load ONCE at startup (can take a few
seconds), not on every image you submit -- so the first request after
starting the server may feel slow while PyTorch/VGG16 initialize, but
subsequent submissions should be fast.
"""

from __future__ import annotations
import argparse
import tempfile
from pathlib import Path

import cv2

try:
    import gradio as gr
except ImportError:
    raise SystemExit(
        "Gradio is not installed. Run: pip install gradio\n"
        "(No --break-system-packages needed on Windows.)"
    )

from ocr_pipeline import OCRPipeline

# Human-readable Nepali labels for display, matching template_config.py field names.
FIELD_LABELS = {
    "first_name": "पहिलो नाम (First Name)",
    "last_name": "थर (Last Name)",
    "place_of_birth": "जन्मस्थान (Place of Birth)",
    "father_first_name": "बुबाको पहिलो नाम (Father's First Name)",
    "father_last_name": "बुबाको थर (Father's Last Name)",
    "mother_first_name": "आमाको पहिलो नाम (Mother's First Name)",
    "mother_last_name": "आमाको थर (Mother's Last Name)",
    "nationality": "राष्ट्रियता (Nationality)",
    "city_district": "जिल्ला (City/District)",
}


def build_app(pipeline: OCRPipeline) -> "gr.Blocks":
    def run_pipeline(image_path: str):
        """Gradio passes an uploaded image as a filepath (via type='filepath').
        Returns a dict for the results table plus the aligned image so you
        can visually sanity-check alignment/cropping went as expected.
        """
        if image_path is None:
            return {}, None, "Upload an image first."

        try:
            result = pipeline.run(image_path)
        except Exception as e:  # noqa: BLE001 -- surface any pipeline error to the UI, don't crash the app
            return {}, None, f"Pipeline error: {e}"

        # Build a display-friendly table: Nepali field label -> recognized text
        display_rows = [
            [FIELD_LABELS.get(name, name), text if text else "(empty / not recognized)"]
            for name, text in result.items()
        ]

        return display_rows, image_path, "Done."

    with gr.Blocks(title="Nepali OCR Pipeline") as demo:
        gr.Markdown(
            "# नेपाली OCR — Form Field Extraction\n"
            "Upload a photo or scan of a **filled-in** membership form. "
            "The pipeline aligns it to the reference template, crops each "
            "field's zone of interest, and runs the CRNN model on each one."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="filepath",
                    label="Filled form (photo or scan)",
                )
                submit_btn = gr.Button("Run OCR", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)

            with gr.Column():
                results_table = gr.Dataframe(
                    headers=["Field", "Recognized Text"],
                    label="Extracted Fields",
                    interactive=False,
                )
                preview_image = gr.Image(label="Uploaded image (for reference)")

        submit_btn.click(
            fn=run_pipeline,
            inputs=[image_input],
            outputs=[results_table, preview_image, status],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the Nepali OCR Gradio GUI.")
    parser.add_argument("--model", required=True, help="Path to a p2_epoch*.pth checkpoint")
    parser.add_argument("--template-pdf", default=None, help="Path to the clean template PDF")
    parser.add_argument("--vocab", default="vocab.json", help="Path to vocab.json for NLP post-processing")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link (valid ~72h)")
    args = parser.parse_args()

    print("Loading pipeline (this may take a few seconds)...")
    pipeline = OCRPipeline(model_path=args.model, template_pdf_path=args.template_pdf, vocab_path=args.vocab)
    print("Pipeline loaded. Starting GUI...")

    app = build_app(pipeline)
    app.launch(share=args.share)
