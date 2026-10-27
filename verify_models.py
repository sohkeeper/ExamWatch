"""Optional pretrained-model smoke test; does not measure exam-event accuracy.

Usage: python verify_models.py /path/to/a/person_photo.jpg
"""
import sys
import cv2
from monitor.detector import YoloDetector

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python verify_models.py person_photo.jpg")
    image = cv2.imread(sys.argv[1])
    if image is None:
        raise SystemExit("Could not decode the image.")
    detections = YoloDetector().detect(image)
    print(f"Model inference completed. People detected: {len(detections)}")
    print(f"Usable head-pose proxies: {sum(d.head_offset is not None for d in detections)}")
    print(f"Associated phone detections: {sum(d.phone_confidence > 0 for d in detections)}")
    print("This is an integration smoke test, not an examination accuracy benchmark.")
