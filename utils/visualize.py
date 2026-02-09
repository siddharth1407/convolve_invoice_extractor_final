import os
import json
import cv2
import numpy as np


def draw_boxes(input_dir, json_path, output_dir, working_dir=None):
    """
    Draws bounding boxes on images based on the extraction results.
    - Stamp: Red Box
    - Signature: Blue Box

    Args:
        input_dir: Directory containing original images
        json_path: Path to JSON results file
        output_dir: Directory to save annotated images
        working_dir: Optional directory containing PDF-derived images
    """
    # 1. Load the JSON results
    if not os.path.exists(json_path):
        print(f"⚠️ Visualization skipped. JSON not found at {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 2. Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"🎨 Generating annotated images in {output_dir}...")

    # 3. Process each document
    for entry in results:
        filename = entry.get('doc_id')
        fields = entry.get('fields', {})

        # Check if image is in input_dir (original images)
        img_path = os.path.join(input_dir, filename)

        # If not found and working_dir provided, check there (PDF-derived images)
        if not os.path.exists(img_path) and working_dir:
            img_path = os.path.join(working_dir, filename)

        # Skip if image missing
        if not os.path.exists(img_path):
            print(f"⚠️  Image not found: {filename}")
            continue

        # Read Image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  Failed to read image: {filename}")
            continue

        h, w, _ = img.shape

        # --- DRAW STAMP (Red) ---
        stamp = fields.get('stamp', {})
        if stamp.get('present'):
            bbox = stamp.get('bbox', [])  # Expecting [x1, y1, x2, y2]
            if len(bbox) == 4 and any(x > 0 for x in bbox):
                x1, y1, x2, y2 = map(int, bbox)
                # Draw Rectangle (BGR: 0, 0, 255 = Red)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                # Draw Label
                cv2.putText(img, "STAMP", (x1, max(y1-10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- DRAW SIGNATURE (Blue) ---
        sig = fields.get('signature', {})
        if sig.get('present'):
            bbox = sig.get('bbox', [])
            if len(bbox) == 4 and any(x > 0 for x in bbox):
                x1, y1, x2, y2 = map(int, bbox)
                # Draw Rectangle (BGR: 255, 0, 0 = Blue)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                # Draw Label
                cv2.putText(img, "SIGNATURE", (x1, max(y1-10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 4. Save the Annotated Image
        out_path = os.path.join(output_dir, f"annotated_{filename}")
        cv2.imwrite(out_path, img)

    print("✅ Visualization Complete.")
