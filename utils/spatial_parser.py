import numpy as np


def generate_spatial_format(ocr_result, image_width, image_height):
    """
    Converts raw OCR results into a spatially preserved text block.
    Sorts text top-to-bottom, left-to-right.
    Returns (spatial_text, raw_text) for downstream use.
    """
    if not ocr_result or not ocr_result[0]:
        return "", ""

    # Flatten the list structure from PaddleOCR
    # Format: [[box, (text, confidence)], ...]
    boxes = []
    for line in ocr_result:
        if not line:
            continue
        for word_info in line:
            try:
                box = word_info[0]
                text = word_info[1][0]
                conf = word_info[1][1]

                # Lower threshold to capture more text (was 0.6)
                # Handwritten text often has lower confidence
                if conf < 0.4:
                    continue

                # Skip very short fragments
                if len(text.strip()) < 1:
                    continue

                # Calculate centroid Y (vertical position) and X (horizontal)
                y_center = sum([p[1] for p in box]) / 4
                x_left = box[0][0]

                boxes.append({
                    "text": text,
                    "y": y_center,
                    "x": x_left,
                    "conf": conf
                })
            except (IndexError, TypeError):
                continue

    if not boxes:
        return "", ""

    # Sort by Y (rows) with a tolerance
    boxes.sort(key=lambda b: b["y"])

    lines = []
    current_line = []
    last_y = -100

    # Dynamic line threshold based on image height
    line_threshold = max(15, image_height * 0.015)  # 1.5% of image height

    for box in boxes:
        # If this box is significantly lower than the last one, start a new line
        if box["y"] - last_y > line_threshold:
            if current_line:
                # Sort the previous line by X (left to right)
                current_line.sort(key=lambda b: b["x"])
                lines.append(" ".join([b["text"] for b in current_line]))
            current_line = []
            last_y = box["y"]

        current_line.append(box)

    # Append the last line
    if current_line:
        current_line.sort(key=lambda b: b["x"])
        lines.append(" ".join([b["text"] for b in current_line]))

    spatial_text = "\n".join(lines)

    # Build raw concatenated text (no layout, just all text)
    raw_text = " ".join(lines)

    return spatial_text, raw_text


def extract_table_rows(ocr_result, image_width, image_height):
    """
    Extract text organized as table rows.
    Useful for finding item-price pairs.
    """
    if not ocr_result or not ocr_result[0]:
        return []

    boxes = []
    for line in ocr_result:
        if not line:
            continue
        for word_info in line:
            try:
                box = word_info[0]
                text = word_info[1][0]
                conf = word_info[1][1]

                if conf < 0.4 or len(text.strip()) < 1:
                    continue

                y_center = sum([p[1] for p in box]) / 4
                x_left = box[0][0]
                x_right = box[2][0]

                boxes.append({
                    "text": text,
                    "y": y_center,
                    "x_left": x_left,
                    "x_right": x_right,
                })
            except (IndexError, TypeError):
                continue

    if not boxes:
        return []

    # Group into rows
    boxes.sort(key=lambda b: b["y"])

    rows = []
    current_row = []
    last_y = -100
    line_threshold = max(15, image_height * 0.015)

    for box in boxes:
        if box["y"] - last_y > line_threshold:
            if current_row:
                current_row.sort(key=lambda b: b["x_left"])
                rows.append(current_row)
            current_row = []
            last_y = box["y"]
        current_row.append(box)

    if current_row:
        current_row.sort(key=lambda b: b["x_left"])
        rows.append(current_row)

    return rows
