from ultralytics import YOLO


class Detector:
    """YOLO detector configured to improve detection of small or difficult objects."""

    def __init__(self, model_name="yolo11n.pt"):
        self.model_name = model_name
        self.model = YOLO(model_name)

    def detect(self, frame, confidence=0.20):
        result = self.model.predict(
            source=frame,
            conf=float(confidence),
            iou=0.50,
            imgsz=960,
            augment=True,
            verbose=False,
        )[0]

        detections = []
        names = result.names

        if result.boxes is None:
            return detections

        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())

            if isinstance(names, dict):
                class_name = names.get(cls_id, str(cls_id))
            else:
                class_name = names[cls_id]

            detections.append({
                "xyxy": [float(v) for v in xyxy],
                "class_id": cls_id,
                "class_name": str(class_name).lower().strip(),
                "confidence": score,
            })

        return detections
