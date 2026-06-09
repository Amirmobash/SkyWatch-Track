"""
plane_monitor.py

A readable, beginner-friendly plane detection and tracking script.

What it does:
1. Opens a camera, video file, or stream URL.
2. Uses YOLO to detect airplanes in each frame.
3. Tracks each detected plane across frames.
4. Optionally smooths movement with a Kalman filter.
5. Draws bounding boxes, track IDs, direction, speed, FPS, and detection time.
6. Can display the result live or save it as a processed video.

Install the main dependencies:
    pip install opencv-python numpy ultralytics scipy filterpy

Example usage:
    python plane_monitor_humanized.py --source video.mp4 --display --trails
    python plane_monitor_humanized.py --source 0 --display
    python plane_monitor_humanized.py --source video.mp4 --output result.mp4
"""

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Optional GPU support. If PyTorch is missing, the script still runs on CPU.
try:
    import torch
except ImportError:
    torch = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None


@dataclass
class Detection:
    """One YOLO detection converted into a simple, easy-to-use object."""
    left: int
    top: int
    right: int
    bottom: int
    score: float
    class_id: int
    label: str

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom

    @property
    def center(self) -> Tuple[int, int]:
        return (self.left + self.right) // 2, (self.top + self.bottom) // 2


@dataclass
class PerformanceStats:
    """Keeps a short moving average of FPS and detection speed."""
    frame_times: deque = field(default_factory=lambda: deque(maxlen=60))
    detection_times: deque = field(default_factory=lambda: deque(maxlen=60))

    def add_frame_time(self, seconds: float) -> None:
        self.frame_times.append(seconds)

    def add_detection_time(self, seconds: float) -> None:
        self.detection_times.append(seconds)

    @property
    def fps(self) -> float:
        if not self.frame_times:
            return 0.0

        average_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / average_time if average_time > 0 else 0.0

    @property
    def detection_ms(self) -> float:
        if not self.detection_times:
            return 0.0

        return sum(self.detection_times) / len(self.detection_times) * 1000.0


class PlaneTrack:
    """Represents one tracked airplane across multiple video frames."""
    def __init__(self, track_id: int, detection: Detection, smooth: bool = True):
        self.id = track_id
        self.box = detection.box
        self.label = detection.label
        self.score = detection.score
        self.points = deque(maxlen=80)
        self.points.append(detection.center)
        self.missed_frames = 0
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.kalman = self._build_kalman(detection.center) if smooth and KalmanFilter else None

    def _build_kalman(self, center: Tuple[int, int]) -> Any:
        x, y = center

        kalman = KalmanFilter(dim_x=4, dim_z=2)

        kalman.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

        kalman.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        kalman.x = np.array([[x], [y], [0], [0]], dtype=float)
        kalman.P *= 100.0
        kalman.R *= 8.0
        kalman.Q *= 0.05

        return kalman

    def update(self, detection: Detection) -> None:
        center = detection.center

        if self.kalman:
            self.kalman.predict()
            self.kalman.update(np.array([[center[0]], [center[1]]], dtype=float))
            center = int(self.kalman.x[0, 0]), int(self.kalman.x[1, 0])

        self.box = detection.box
        self.label = detection.label
        self.score = detection.score
        self.points.append(center)
        self.missed_frames = 0
        self.updated_at = time.time()

    def mark_missing(self) -> None:
        self.missed_frames += 1

        if self.kalman:
            self.kalman.predict()
            center = int(self.kalman.x[0, 0]), int(self.kalman.x[1, 0])
            self.points.append(center)

    def center(self) -> Tuple[int, int]:
        return self.points[-1]

    def speed(self, fps: float) -> float:
        if len(self.points) < 2:
            return 0.0

        recent = list(self.points)[-10:]
        distance = 0.0

        for previous, current in zip(recent, recent[1:]):
            distance += math.hypot(current[0] - previous[0], current[1] - previous[1])

        steps = max(1, len(recent) - 1)
        return distance / steps * fps

    def direction(self) -> str:
        if len(self.points) < 2:
            return "steady"

        recent = list(self.points)[-12:]
        start_x, start_y = recent[0]
        end_x, end_y = recent[-1]

        dx = end_x - start_x
        dy = end_y - start_y

        if math.hypot(dx, dy) < 6:
            return "steady"

        parts = []

        if abs(dy) > 4:
            parts.append("down" if dy > 0 else "up")

        if abs(dx) > 4:
            parts.append("right" if dx > 0 else "left")

        return "-".join(parts) if parts else "steady"


class PlaneTracker:
    """Matches new detections to existing tracks and removes old lost tracks."""
    def __init__(self, max_missing: int = 30, max_distance: float = 120.0, smooth: bool = True):
        self.max_missing = max_missing
        self.max_distance = max_distance
        self.smooth = smooth
        self.next_id = 1
        self.tracks: Dict[int, PlaneTrack] = {}

    def update(self, detections: List[Detection]) -> Dict[int, PlaneTrack]:
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

        self._remove_lost_tracks()
        return self.tracks

    def _add_track(self, detection: Detection) -> None:
        self.tracks[self.next_id] = PlaneTrack(self.next_id, detection, self.smooth)
        self.next_id += 1

    def _mark_all_missing(self) -> None:
        for track in self.tracks.values():
            track.mark_missing()

        self._remove_lost_tracks()

    def _remove_lost_tracks(self) -> None:
        lost_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed_frames > self.max_missing
        ]

        for track_id in lost_ids:
            del self.tracks[track_id]

    def _distance_matrix(self, tracks: np.ndarray, detections: np.ndarray) -> np.ndarray:
        diff = tracks[:, None, :] - detections[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2))

    def _match(self, distances: np.ndarray) -> List[Tuple[int, int]]:
        if linear_sum_assignment:
            rows, columns = linear_sum_assignment(distances)
            return list(zip(rows, columns))

        pairs = []

        for row in range(distances.shape[0]):
            for column in range(distances.shape[1]):
                pairs.append((distances[row, column], row, column))

        pairs.sort(key=lambda item: item[0])

        used_rows = set()
        used_columns = set()
        matches = []

        for _, row, column in pairs:
            if row in used_rows or column in used_columns:
                continue

            matches.append((row, column))
            used_rows.add(row)
            used_columns.add(column)

        return matches


class PlaneDetector:
    """Small wrapper around the Ultralytics YOLO model."""
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.35,
        iou: float = 0.45,
        targets: Optional[List[str]] = None,
        device: str = "auto",
        max_detections: int = 100
    ):
        if YOLO is None:
            raise RuntimeError("Ultralytics is not installed. Run: pip install ultralytics")

        self.model_name = model_name
        self.confidence = confidence
        self.iou = iou
        self.targets = [target.lower() for target in (targets or ["airplane"])]
        self.device = self._pick_device(device)
        self.max_detections = max_detections
        self.model = YOLO(model_name)
        self.names = getattr(self.model, "names", {})

    def _pick_device(self, device: str) -> str:
        if device != "auto":
            return device

        if torch and torch.cuda.is_available():
            return "cuda"

        if torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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
            score = float(box.conf.detach().cpu().numpy()[0])
            label = str(self.names.get(class_id, class_id)).lower()

            if not self._is_target(label):
                continue

            left, top, right, bottom = box.xyxy[0].detach().cpu().numpy()

            detections.append(
                Detection(
                    left=int(left),
                    top=int(top),
                    right=int(right),
                    bottom=int(bottom),
                    score=score,
                    class_id=class_id,
                    label=label
                )
            )

        return detections

    def _is_target(self, label: str) -> bool:
        return any(target in label for target in self.targets)


def read_source(value: str) -> Union[int, str]:
    value = value.strip()
    return int(value) if value.isdigit() else value


def draw_detections(frame: np.ndarray, detections: List[Detection]) -> None:
    for detection in detections:
        cv2.rectangle(
            frame,
            (detection.left, detection.top),
            (detection.right, detection.bottom),
            (0, 180, 0),
            2
        )

        text = f"{detection.label} {detection.score:.2f}"

        cv2.putText(
            frame,
            text,
            (detection.left, max(20, detection.top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2
        )


def draw_tracks(frame: np.ndarray, tracks: Dict[int, PlaneTrack], fps: float, show_trails: bool) -> None:
    for track_id, track in tracks.items():
        if track.missed_frames > 3:
            continue

        x, y = track.center()
        speed = track.speed(fps)
        direction = track.direction()

        cv2.circle(frame, (x, y), 5, (0, 128, 255), -1)

        text = f"Plane {track_id} | {direction} | {speed:.0f}px/s"

        cv2.putText(
            frame,
            text,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2
        )

        if show_trails:
            points = list(track.points)

            for previous, current in zip(points, points[1:]):
                cv2.line(frame, previous, current, (0, 128, 255), 2)


def draw_status(frame: np.ndarray, stats: PerformanceStats, detections: int, tracks: int) -> None:
    text = (
        f"FPS: {stats.fps:.1f} | "
        f"Detection: {stats.detection_ms:.1f} ms | "
        f"Planes: {detections} | "
        f"Tracks: {tracks}"
    )

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


def make_writer(path: str, video: cv2.VideoCapture) -> cv2.VideoWriter:
    fps = video.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 1:
        fps = 30.0

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def run(args: argparse.Namespace) -> None:
    """Main processing loop: read frame, detect planes, track them, draw results."""
    source = read_source(args.source)
    video = cv2.VideoCapture(source)

    if not video.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    detector = PlaneDetector(
        model_name=args.model,
        confidence=args.confidence,
        iou=args.iou,
        targets=args.target,
        device=args.device,
        max_detections=args.max_detections
    )

    tracker = PlaneTracker(
        max_missing=args.max_missing,
        max_distance=args.max_distance,
        smooth=not args.no_kalman
    )

    stats = PerformanceStats()
    writer = make_writer(args.output, video) if args.output else None

    try:
        while True:
            frame_start = time.perf_counter()

            ok, frame = video.read()

            if not ok:
                break

            detection_start = time.perf_counter()
            detections = detector.detect(frame)
            stats.add_detection_time(time.perf_counter() - detection_start)

            tracks = tracker.update(detections)
            output = frame.copy()

            draw_detections(output, detections)
            draw_tracks(output, tracks, stats.fps or 30.0, args.trails)
            draw_status(output, stats, len(detections), len(tracks))

            if writer:
                writer.write(output)

            if args.display:
                cv2.imshow("Plane Monitor", output)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            stats.add_frame_time(time.perf_counter() - frame_start)

    finally:
        video.release()

        if writer:
            writer.release()

        if args.display:
            cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    """Create all command-line options in one place."""
    parser = argparse.ArgumentParser(
        prog="plane_monitor",
        description="Detect and track planes in video."
    )

    parser.add_argument("--source", default="0", help="Camera number, video file, or stream URL")
    parser.add_argument("--output", help="Save the processed video")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model name or path")
    parser.add_argument("--confidence", type=float, default=0.35, help="Minimum detection confidence")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold")
    parser.add_argument("--target", nargs="+", default=["airplane"], help="Object names to detect")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--display", action="store_true", help="Show the video window")
    parser.add_argument("--trails", action="store_true", help="Show movement trails")
    parser.add_argument("--no-kalman", action="store_true", help="Disable smooth tracking")
    parser.add_argument("--max-missing", type=int, default=30, help="Frames to keep missing tracks")
    parser.add_argument("--max-distance", type=float, default=120.0, help="Maximum matching distance")
    parser.add_argument("--max-detections", type=int, default=100, help="Maximum detections per frame")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        run(args)
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
