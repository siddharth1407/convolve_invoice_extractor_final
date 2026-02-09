import time
import os
import json
import torch
import shutil
from pathlib import Path
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text

# Import your actual logic
from executable import load_model, process_single_image
# Import your new Vision Engine
from utils.vision_utils import VisionEngine
# Import PDF processing
from utils.pdf_utils import get_pdf_processor

# Initialize Console
console = Console()

# Global model cache
_model_cache = {}
_vision_engine_cache = None


def get_cached_model():
    """Load model once and reuse it."""
    global _model_cache
    if "model" not in _model_cache:
        console.print(
            "[bold yellow]⏳ Loading Qwen2-VL (Brain)...[/bold yellow]")
        _model_cache["model"] = load_model()
    return _model_cache["model"]


def get_cached_vision_engine():
    """Load vision engine once and reuse it."""
    global _vision_engine_cache
    if _vision_engine_cache is None:
        console.print("[bold yellow]⏳ Loading YOLOv8 (Eye)...[/bold yellow]")
        _vision_engine_cache = VisionEngine()
    return _vision_engine_cache


def make_layout():
    """Defines the grid structure of the terminal UI."""
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )
    layout["main"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )
    return layout


def create_header():
    return Panel(
        Text("NEURO-SYMBOLIC INVOICE EXTRACTOR v1.0",
             justify="center", style="bold white on blue"),
        style="blue"
    )


def create_log_table(logs):
    table = Table(title="System Event Log", expand=True, border_style="green")
    table.add_column("Time", style="cyan", no_wrap=True)
    table.add_column("Module", style="magenta")
    table.add_column("Message", style="white")

    for log in logs[-12:]:  # Show last 12 events
        table.add_row(log[0], log[1], log[2])

    return Panel(table, title="Backend Logic Layer", border_style="green")


def create_json_view(data):
    if not data:
        return Panel(Text("Waiting for inference...", justify="center"), title="Live JSON Output")

    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
    return Panel(syntax, title="Structured Extraction (Qwen2-VL + YOLO)", border_style="yellow")


def run_demo(input_dir):
    # 1. Initialize Engines (cached)
    model, processor, device = get_cached_model()
    vision = get_cached_vision_engine()
    pdf_processor = get_pdf_processor(dpi=300)  # 300 DPI for good OCR quality

    # Create working directory for PDF-derived images
    working_dir = os.path.join(input_dir, ".pdf_working")
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir, exist_ok=True)

    # Get both images and PDFs
    image_extensions = ('.png', '.jpg', '.jpeg')
    pdf_extensions = ('.pdf',)

    all_files = []
    for f in os.listdir(input_dir):
        if f.lower().endswith(image_extensions):
            all_files.append((f, 'image', os.path.join(input_dir, f)))
        elif f.lower().endswith(pdf_extensions):
            all_files.append((f, 'pdf', os.path.join(input_dir, f)))

    all_files.sort()

    if not all_files:
        console.print(
            f"[bold red]❌ No images or PDFs found in {input_dir}![/bold red]")
        return

    # Create output directory
    output_dir = "sample_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    annotated_dir = os.path.join(output_dir, "annotated_images")
    if not os.path.exists(annotated_dir):
        os.makedirs(annotated_dir)

    layout = make_layout()
    layout["header"].update(create_header())

    logs = []
    all_results = []
    processed_count = 0
    total_files = len(all_files)
    pdf_image_map = {}

    with Live(layout, refresh_per_second=4, screen=True):
        for file_idx, (filename, file_type, file_path) in enumerate(all_files, 1):

            # Handle PDFs - convert to images
            if file_type == 'pdf':
                console.print(
                    f"\n[bold cyan]📄 Converting PDF to images: {filename}[/bold cyan]")
                try:
                    image_paths = pdf_processor.pdf_to_images(file_path)
                    console.print(
                        f"[green]✅ PDF converted to {len(image_paths)} page(s)[/green]")
                    # Process each page
                    for page_num, image_path in enumerate(image_paths, 1):
                        processed_count += 1
                        page_filename = f"{Path(filename).stem}_page_{page_num:03d}.png"

                        # Copy image to working directory (persistent location for visualization)
                        working_image_path = os.path.join(
                            working_dir, page_filename)
                        shutil.copy2(image_path, working_image_path)
                        pdf_image_map[page_filename] = working_image_path

                        print(
                            f"\n[PDF Page {page_num}/{len(image_paths)}] Processing: {page_filename}")
                        _process_image(
                            model, processor, device, vision,
                            image_path, page_filename, processed_count, total_files,
                            logs, layout, all_results, output_dir, annotated_dir
                        )
                except Exception as e:
                    console.print(
                        f"[bold red]❌ Failed to process PDF {filename}: {e}[/bold red]")
                    continue
            else:
                # Regular image
                processed_count += 1
                print(
                    f"\n[Image {file_idx}/{total_files}] Processing: {filename}")
                print(
                    f"Model state: _loaded_model={load_model.__globals__.get('_loaded_model', None) is not None}")
                _process_image(
                    model, processor, device, vision,
                    file_path, filename, processed_count, total_files,
                    logs, layout, all_results, output_dir, annotated_dir
                )

    # Cleanup working directory
    if working_dir and os.path.exists(working_dir):
        shutil.rmtree(working_dir)
        console.print("[cyan]✓ Cleaned up temporary files[/cyan]")


def _process_image(model, processor, device, vision, image_path, filename,
                   current, total, logs, layout, all_results, output_dir, annotated_dir):
    """Helper function to process a single image."""

    # --- PHASE 1: VISION SCAN (YOLO) ---
    t = time.strftime("%H:%M:%S")
    logs.append((t, "SYSTEM", f"Processing: {filename}"))
    layout["left"].update(create_log_table(logs))
    layout["footer"].update(
        Panel(f"Scanning: {filename}...", style="bold yellow"))
    time.sleep(0.5)

    # Run YOLO
    yolo_stamp, yolo_sig = vision.detect_objects(image_path)

    if yolo_stamp['present']:
        logs.append((t, "EYE (YOLO)", "✅ Stamp Detected"))
    if yolo_sig['present']:
        logs.append((t, "EYE (YOLO)", "✅ Signature Detected"))

    # --- PHASE 2: TEXT READ (QWEN) ---
    time.sleep(0.5)
    t = time.strftime("%H:%M:%S")
    logs.append((t, "BRAIN (QWEN)", "Transcribing Vernacular Text..."))
    layout["left"].update(create_log_table(logs))

    # Run Qwen
    result = process_single_image(
        model, processor, device, image_path, filename)
    fields = result.get("fields", {})

    # --- PHASE 3: MERGE & SAVE ---
    # Inject YOLO results into the final JSON
    if yolo_stamp['present']:
        fields['stamp'] = yolo_stamp
    if yolo_sig['present']:
        fields['signature'] = yolo_sig

    conf = result.get("confidence", 0.0)

    # Store for final saving
    all_results.append({
        "doc_id": filename,
        "fields": fields,
        "confidence": conf
    })

    t = time.strftime("%H:%M:%S")
    if conf > 0.9:
        logs.append((t, "LOGIC", f"Confidence High ({conf}). Valid."))
    else:
        logs.append(
            (t, "LOGIC", f"⚠️ Low Confidence ({conf}). Flagged."))

    layout["left"].update(create_log_table(logs))
    layout["right"].update(create_json_view(fields))
    layout["footer"].update(
        Panel(f"DONE: {filename}", style="bold green"))

    # --- PHASE 4: SAVE AFTER EACH IMAGE ---
    # Save JSON
    json_path = os.path.join(output_dir, "result.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    # Draw and save annotated image
    import cv2
    img = cv2.imread(image_path)
    if img is not None:
        h, w, _ = img.shape

        # Draw stamp
        stamp = fields.get('stamp', {})
        if stamp.get('present'):
            bbox = stamp.get('bbox', [])
            if len(bbox) == 4 and any(x > 0 for x in bbox):
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(img, "STAMP", (x1, max(y1-10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Draw signature
        sig = fields.get('signature', {})
        if sig.get('present'):
            bbox = sig.get('bbox', [])
            if len(bbox) == 4 and any(x > 0 for x in bbox):
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(img, "SIGNATURE", (x1, max(y1-10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Save annotated image
        out_path = os.path.join(annotated_dir, f"annotated_{filename}")
        cv2.imwrite(out_path, img)
        logs.append((t, "SAVE", f"💾 Saved {os.path.basename(out_path)}"))

    # Pause so you can explain it to the judges
    time.sleep(3)


if __name__ == "__main__":
    if not os.path.exists("input_images"):
        os.makedirs("input_images")
    run_demo("input_images")
