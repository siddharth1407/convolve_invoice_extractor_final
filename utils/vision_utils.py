import os
import torch
from ultralytics import YOLO

# This points to your trained model
YOLO_MODEL_PATH = "model_weights/best.pt"


def _select_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class VisionEngine:
    def __init__(self):
        self.model = None
        self.device = _select_device()

        # Check if model exists
        if os.path.exists(YOLO_MODEL_PATH):
            print(
                f"👁️ Loading YOLO Vision Model from {YOLO_MODEL_PATH} on {self.device}...")
            try:
                # Load the model
                self.model = YOLO(YOLO_MODEL_PATH)
                # Attempt to move model to device for faster inference when supported
                try:
                    self.model.to(self.device)
                except Exception:
                    # Some ultralytics versions handle device per-predict call
                    pass
            except Exception as e:
                print(f"⚠️ Failed to initialize YOLO model: {e}")
                self.model = None
        else:
            print(
                f"⚠️ YOLO model not found at {YOLO_MODEL_PATH}. Skipping object detection.")

    def detect_objects(self, image_path):
        """
        Runs YOLO inference. Uses GPU/MPS when available for speed.
        """
        # Default empty result
        stamp_res = {"present": False, "bbox": [0, 0, 0, 0]}
        sig_res = {"present": False, "bbox": [0, 0, 0, 0]}

        if not self.model:
            return stamp_res, sig_res

        try:
            results = self.model.predict(
                image_path, conf=0.25, verbose=False, device=self.device)

            # Process results
            for result in results:
                for box in result.boxes:
                    coords = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
                    cls_id = int(box.cls[0].item())
                    label = self.model.names[cls_id]

                    # Convert to integers - keep as [xmin, ymin, xmax, ymax]
                    x1, y1, x2, y2 = map(int, coords)

                    # Store in standard [x1, y1, x2, y2] format
                    bbox_formatted = [x1, y1, x2, y2]

                    # Store results based on label
                    if label == 'stamp':
                        stamp_res = {"present": True, "bbox": bbox_formatted}
                    elif 'signature' in label:
                        sig_res = {"present": True, "bbox": bbox_formatted}

        except Exception as e:
            print(f"❌ YOLO Inference Error: {e}")

        return stamp_res, sig_res
