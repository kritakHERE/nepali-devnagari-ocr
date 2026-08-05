"""Gradio GUI for the Nepali OCR v2 pipeline.

This launcher is paired with `ocr_pipeline_v2.py` and is intended for the
phase-2 checkpoints that use the 128x320 input resolution and external
`vocab.json` file.

USAGE
-----
    python gui2.0.py --model ..\..\models\best_models\phase2_BEST.pth

Then open the printed local URL (usually http://127.0.0.1:7860) in your
browser. Upload a photo/scan of a FILLED form, click Submit, and the
recognized text for each field will appear.
"""

from __future__ import annotations

import argparse

try:
    import gradio as gr
except ImportError:
    raise SystemExit(
        "Gradio is not installed. Run: pip install gradio\n"
        "(No --break-system-packages needed on Windows.)"
    )

from ocr_pipeline_v2 import OCRPipeline

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
        Returns a dict for the results table plus the uploaded image so the
        user can visually sanity-check the input.
        """
        if image_path is None:
            return {}, None, "Upload an image first."

        try:
            result = pipeline.run(image_path)
        except Exception as e:  # noqa: BLE001 -- surface any pipeline error to the UI, don't crash the app
            return {}, None, f"Pipeline error: {e}"

        display_rows = [
            [FIELD_LABELS.get(name, name), text if text else "(empty / not recognized)"]
            for name, text in result.items()
        ]

        return display_rows, image_path, "Done."

    with gr.Blocks(title="Nepali OCR Pipeline v2") as demo:
        gr.Markdown(
            "# नेपाली OCR v2 — Form Field Extraction\n"
            "Upload a photo or scan of a **filled-in** membership form. "
            "This launcher uses the v2 model path with 128×320 input "
            "resolution and `vocab.json` loaded externally."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="filepath",
                    label="Filled form (photo or scan)",
                )
                submit_btn = gr.Button("Run OCR v2", variant="primary")
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
    parser = argparse.ArgumentParser(description="Launch the Nepali OCR v2 Gradio GUI.")
    parser.add_argument("--model", required=True, help="Path to a phase2_BEST.pth checkpoint")
    parser.add_argument("--template-pdf", default=None, help="Path to the clean template PDF")
    parser.add_argument("--vocab", default="vocab.json", help="Path to vocab.json for NLP post-processing")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link (valid ~72h)")
    args = parser.parse_args()

    print("Loading v2 pipeline (this may take a few seconds)...")
    pipeline = OCRPipeline(model_path=args.model, template_pdf_path=args.template_pdf, vocab_path=args.vocab)
    print("Pipeline loaded. Starting GUI v2...")

    app = build_app(pipeline)
    app.launch(share=args.share)