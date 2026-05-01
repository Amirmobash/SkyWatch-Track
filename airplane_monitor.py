import argparse
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from filterpy.kalman import KalmanFilter
    HAS_FILTERPY = True
except ImportError:
    HAS_FILTERPY = False


class TrackState(Enum):
    NEW = "new"
    TRACKING = "tracking"
    LOST = "lost"


@dataclass
class Detection:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    confidence: float
    class_id: int
    name: str

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self.xmin, self.ymin, self.xmax, self.ymax

    @property
    def center(self) -> Tuple[int, int]:
        return (self.xmin + self.xmax) // 2, (self.ymin + self.ymax) // 2


@dataclass
class PerformanceMonitor:
    frame_times: deque = field(default_factory=lambda: deque(maxlen=60))
    inference_times: deque = field(default_factory=lambda: deque(maxlen=60))

    def add_frame_time(self, value: float) -> None:
        self.frame_times.append(value)

    def add_inference_time(self, value: float) -> None:
        self.inference_times.append(value)

    @property
    def fps(self) -> float:
        if not self.frame_times:
            return 0.0

        average = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / average if average > 0 else 0.0

    @property
    def inference_ms(self) -> float:
        if not self.inference_times:
            return 0.0

        return (sum(self.inference_times) / len(self.inference_times)) * 1000.0


class TrackedObject:
    def __init__(
        self,
        track_id: int,
        detection: Detection,
        use_kalman: bool = True,
        max_history: int = 80
    ):
        self.id = track_id
        self.bbox = detection.bbox
        self.name = detection.name
        self.confidence = detection.confidence
        self.centers = deque(maxlen=max_history)
        self.centers.append(detection.center)
        self.missing_frames = 0
        self.state = TrackState.NEW
        self.first_seen = time.time()
        self.last_seen = self.first_seen
        self.use_kalman = use_kalman and HAS_FILTERPY
        self.kalman = self._create_kalman_filter(detection.center) if self.use_kalman else None

    def _create_kalman_filter(self, center: Tuple[int, int]) -> KalmanFilter:
        kf = KalmanFilter(dim_x=4, dim_z=2)

        kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        kf.x = np.array([center[0], center[1], 0, 0], dtype=float)
        kf.P *= 100.0
        kf.R *= 8.0
        kf.Q *= 0.05

        return kf

    def update(self, detection: Detection) -> None:
        center = detection.center

        if self.kalman is not None:
            self.kalman.predict()
            self.kalman.update(np.array(center, dtype=float))
            center = int(self.kalman.x[0]), int(self.kalman.x[1])

        self.bbox = detection.bbox
        self.name = detection.name
        self.confidence = detection.confidence
        self.centers.append(center)
        self.missing_frames = 0
        self.last_seen = time.time()

        if len(self.centers) >= 3:
            self.state = TrackState.TRACKING

    def mark_missing(self) -> None:
        self.missing_frames += 1

        if self.kalman is not None:
            self.kalman.predict()
            self.centers.append((int(self.kalman.x[0]), int(self.kalman.x[1])))

        if self.missing_frames > 3:
            self.state = TrackState.LOST

    def center(self) -> Tuple[int, int]:
        return self.centers[-1]

    def speed(self, fps: float) -> float:
        if len(self.centers) < 2:
            return 0.0

        points = list(self.centers)[-10:]

        if len(points) < 2:
            return 0.0

        distance = 0.0

        for index in range(1, len(points)):
            x1, y1 = points[index - 1]
            x2, y2 = points[index]
            distance += math.hypot(x2 - x1, y2 - y1)

        return distance / max(1, len(points) - 1) * fps

    def direction(self) -> str:
        if len(self.centers) < 2:
            return "steady"

        points = list(self.centers)[-12:]
        start = points[0]
        end = points[-1]

        dx = end[0] - start[0]
        dy = end[1] - start[1]

        if math.hypot(dx, dy) < 6:
            return "steady"

        horizontal = ""
        vertical = ""

        if abs(dx) > 4:
            horizontal = "right" if dx > 0 else "left"

        if abs(dy) > 4:
            vertical = "down" if dy > 0 else "up"

        if horizontal and vertical:
            return f"{vertical}-{horizontal}"

        return horizontal or vertical or "steady"


class ObjectTracker:
    def __init__(
        self,
        max_missing: int = 30,
        max_distance: float = 120.0,
        use_kalman: bool = True
    ):
        self.max_missing = max_missing
        self.max_distance = max_distance
        self.use_kalman = use_kalman
        self.next_id = 1
        self.tracks: Dict[int, TrackedObject] = {}

    def update(self, detections: List[Detection]) -> Dict[int, TrackedObject]:
        if not detections:
            self._mark_all_missing()
            return self.tracks

        if not self.tracks:
            for detection in detections:
                self._add_track(detection)
            return self.tracks

        track_ids = list(self.tracks.keys())
        track_centers = np.array([self.tracks[track_id].center() for track_id in track_ids], dtype=float)
        detection_centers = np.array([detection.center for detection in detections], dtype=float)

        distances = self._distance_matrix(track_centers, detection_centers)
        matches = self._match(distances)

        used_tracks = set()
        used_detections = set()

        for track_index, detection_index in matches:
            if distances[track_index, detection_index] > self.max_distance:
                continue

            track_id = track_ids[track_index]
            self.tracks[track_id].update(detections[detection_index])
            used_tracks.add(track_index)
            used_detections.add(detection_index)

        for index, track_id in enumerate(track_ids):
            if index not in used_tracks:
                self.tracks[track_id].mark_missing()

        for index, detection in enumerate(detections):
            if index not in used_detections:
                self._add_track(detection)

        self._remove_old_tracks()
        return self.tracks

    def _add_track(self, detection: Detection) -> None:
        self.tracks[self.next_id] = TrackedObject(
            track_id=self.next_id,
            detection=detection,
            use_kalman=self.use_kalman
        )
        self.next_id += 1

    def _mark_all_missing(self) -> None:
        for track in self.tracks.values():
            track.mark_missing()

        self._remove_old_tracks()

    def _remove_old_tracks(self) -> None:
        dead_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missing_frames > self.max_missing
        ]

        for track_id in dead_ids:
            del self.tracks[track_id]

    @staticmethod
    def _distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = a[:, None, :] - b[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2))

    @staticmethod
    def _match(distances: np.ndarray) -> List[Tuple[int, int]]:
        if HAS_SCIPY:
            rows, cols = linear_sum_assignment(distances)
            return list(zip(rows, cols))

        pairs = []

        for row in range(distances.shape[0]):
            for col in range(distances.shape[1]):
                pairs.append((distances[row, col], row, col))

        pairs.sort(key=lambda item: item[0])

        used_rows = set()
        used_cols = set()
        result = []

        for _, row, col in pairs:
            if row in used_rows or col in used_cols:
                continue

            result.append((row, col))
            used_rows.add(row)
            used_cols.add(col)

        return result


class YoloDetector:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.35,
        iou: float = 0.45,
        target_names: Optional[List[str]] = None,
        device: str = "auto",
        max_detections: int = 100
    ):
        if not HAS_ULTRALYTICS:
            raise RuntimeError("Ultralytics is not installed. Install it with: pip install ultralytics")

        self.model_name = model_name
        self.confidence = confidence
        self.iou = iou
        self.target_names = [name.lower() for name in (target_names or ["airplane"])]
        self.device = self._choose_device(device)
        self.max_detections = max_detections
        self.model = YOLO(model_name)
        self.names = getattr(self.model, "names", {})

    def _choose_device(self, device: str) -> str:
        if device != "auto":
            return device

        if HAS_TORCH and torch.cuda.is_available():
            return "cuda"

        if HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            max_det=self.max_detections,
            device=self.device,
            verbose=False
        )

        result = results[0]
        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls.detach().cpu().numpy()[0])
            confidence = float(box.conf.detach().cpu().numpy()[0])
            name = str(self.names.get(class_id, class_id))

            if not self._is_target(name):
                continue

            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()
            detections.append(
                Detection(
                    xmin=int(x1),
                    ymin=int(y1),
                    xmax=int(x2),
                    ymax=int(y2),
                    confidence=confidence,
                    class_id=class_id,
                    name=name
                )
            )

        return detections

    def _is_target(self, name: str) -> bool:
        name = name.lower()
        return any(target in name for target in self.target_names)


def parse_source(value: str) -> Union[int, str]:
    value = value.strip()

    if value.isdigit():
        return int(value)

    return value


def draw_detections(frame: np.ndarray, detections: List[Detection]) -> None:
    for detection in detections:
        cv2.rectangle(
            frame,
            (detection.xmin, detection.ymin),
            (detection.xmax, detection.ymax),
            (0, 180, 0),
            2
        )

        label = f"{detection.name} {detection.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (detection.xmin, max(20, detection.ymin - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2
        )


def draw_tracks(frame: np.ndarray, tracks: Dict[int, TrackedObject], fps: float, show_trails: bool) -> None:
    for track_id, track in tracks.items():
        if track.state == TrackState.LOST:
            continue

        cx, cy = track.center()
        speed = track.speed(fps)
        direction = track.direction()

        cv2.circle(frame, (cx, cy), 5, (0, 128, 255), -1)

        text = f"ID {track_id} | {direction} | {speed:.0f}px/s"
        cv2.putText(
            frame,
            text,
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2
        )

        if show_trails:
            points = list(track.centers)

            for index in range(1, len(points)):
                cv2.line(frame, points[index - 1], points[index], (0, 128, 255), 2)


def draw_status(frame: np.ndarray, monitor: PerformanceMonitor, detections: int, tracks: int) -> None:
    text = f"FPS: {monitor.fps:.1f} | Inference: {monitor.inference_ms:.1f} ms | Detections: {detections} | Tracks: {tracks}"

    cv2.rectangle(frame, (10, 10), (760, 44), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


def open_writer(path: str, capture: cv2.VideoCapture) -> cv2.VideoWriter:
    fps = capture.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 1:
        fps = 30.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def run(args: argparse.Namespace) -> None:
    source = parse_source(args.source)
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    detector = YoloDetector(
        model_name=args.model,
        confidence=args.confidence,
        iou=args.iou,
        target_names=args.target,
        device=args.device,
        max_detections=args.max_detections
    )

    tracker = ObjectTracker(
        max_missing=args.max_missing,
        max_distance=args.max_distance,
        use_kalman=not args.no_kalman
    )

    monitor = PerformanceMonitor()
    writer = open_writer(args.output, capture) if args.output else None

    try:
        while True:
            frame_start = time.perf_counter()

            ok, frame = capture.read()

            if not ok:
                break

            inference_start = time.perf_counter()
            detections = detector.detect(frame)
            monitor.add_inference_time(time.perf_counter() - inference_start)

            tracks = tracker.update(detections)

            output = frame.copy()
            draw_detections(output, detections)
            draw_tracks(output, tracks, monitor.fps or 30.0, args.trails)
            draw_status(output, monitor, len(detections), len(tracks))

            if writer is not None:
                writer.write(output)

            if args.display:
                cv2.imshow("Airplane Monitor", output)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            monitor.add_frame_time(time.perf_counter() - frame_start)

    finally:
        capture.release()

        if writer is not None:
            writer.release()

        if args.display:
            cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airplane_monitor",
        description="Detects airplanes, tracks them, and estimates movement direction."
    )

    parser.add_argument("--source", default="0", help="Camera index, video path, or stream URL")
    parser.add_argument("--output", help="Optional output video path")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path or model name")
    parser.add_argument("--confidence", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold")
    parser.add_argument("--target", nargs="+", default=["airplane"], help="Object names to detect")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--display", action="store_true", help="Show live preview")
    parser.add_argument("--trails", action="store_true", help="Draw movement trails")
    parser.add_argument("--no-kalman", action="store_true", help="Disable Kalman smoothing")
    parser.add_argument("--max-missing", type=int, default=30, help="Frames before removing a lost track")
    parser.add_argument("--max-distance", type=float, default=120.0, help="Maximum matching distance between frames")
    parser.add_argument("--max-detections", type=int, default=100, help="Maximum detections per frame")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        run(args)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
