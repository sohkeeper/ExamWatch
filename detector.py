"""YOLO11 perception adapter. Heavy dependencies/models load only for real video."""
import math
from .types import Detection


class YoloDetector:
    def __init__(self, person_conf=0.35, phone_conf=0.30, device="cpu"):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install the full requirements.txt to enable AI video analysis.") from exc
        self.pose = YOLO("yolo11n-pose.pt")
        self.objects = YOLO("yolo11n.pt")
        self.person_conf, self.phone_conf, self.device = person_conf, phone_conf, device
        self.phone_classes = [key for key, name in self.objects.names.items() if name == "cell phone"]
        if not self.phone_classes:
            raise RuntimeError("The selected object model has no cell phone class.")

    def detect(self, frame):
        pose = self.pose.predict(frame, conf=self.person_conf, imgsz=640,
                                 device=self.device, verbose=False)[0]
        people = []
        if pose.boxes is not None:
            boxes = pose.boxes.xyxy.cpu().numpy()
            confs = pose.boxes.conf.cpu().numpy()
            keypoints = pose.keypoints.data.cpu().numpy() if pose.keypoints is not None else []
            for index, (box, conf) in enumerate(zip(boxes, confs)):
                offset, head_conf = None, 0.0
                if index < len(keypoints):
                    kp = keypoints[index]
                    # COCO pose: 0 nose, 3 left ear, 4 right ear.
                    if len(kp) >= 5 and kp.shape[1] >= 3 and min(kp[0, 2], kp[3, 2], kp[4, 2]) >= 0.5:
                        ear_width = abs(float(kp[3, 0] - kp[4, 0]))
                        if ear_width >= 6:
                            offset = float((kp[0, 0] - (kp[3, 0] + kp[4, 0]) / 2) / ear_width)
                            head_conf = float(min(kp[0, 2], kp[3, 2], kp[4, 2]))
                people.append(Detection(tuple(float(x) for x in box), float(conf), offset, head_conf))
        phone_result = self.objects.predict(frame, classes=self.phone_classes, conf=self.phone_conf,
                                             imgsz=640, device=self.device, verbose=False)[0]
        if phone_result.boxes is not None:
            for box, conf in zip(phone_result.boxes.xyxy.cpu().numpy(), phone_result.boxes.conf.cpu().numpy()):
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                candidates = []
                for index, person in enumerate(people):
                    x1, y1, x2, y2 = person.box
                    pad_x, pad_y = (x2 - x1) * 0.15, (y2 - y1) * 0.15
                    if x1 - pad_x <= center[0] <= x2 + pad_x and y1 - pad_y <= center[1] <= y2 + pad_y:
                        candidates.append((math.dist(center, person.center), index))
                if candidates:
                    # One phone detection is attributed to at most one track.
                    index = min(candidates)[1]
                    people[index].phone_confidence = max(people[index].phone_confidence,
                                                          min(float(conf), people[index].confidence))
        return people
