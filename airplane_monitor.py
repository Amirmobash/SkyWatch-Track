import argparse
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

try:
    from ultralytics import YOLO
    has_yolo = True
except ImportError:
    has_yolo = False

try:
    from scipy.optimize import linear_sum_assignment
    has_scipy = True
except ImportError:
    has_scipy = False

try:
    from filterpy.kalman import KalmanFilter
    has_filterpy = True
except ImportError:
    has_filterpy = False


@dataclass
class Detection:
    left: int
    top: int
    right: int
    bottom: int
    score: float
    class_id: int
    name: str

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom

    @property
    def center(self) -> Tuple[int, int]:
        x = (self.left + self.right) // 2
        y = (self.top + self.bottom) // 2
        return x, y


@dataclass
class SpeedMeter:
    frame_times: deque = field(default_factory=lambda: deque(maxlen=60))
    detection_times: deque = field(default_factory=lambda: deque(maxlen=60))

    def add_frame(self, seconds: float) -> None:
        self.frame_times.append(seconds)

    def add_detection(self, seconds: float) -> None:
        self.detection_times.append(seconds)

    @property
    def fps(self) -> float:
        if not self.frame_times:
            return 0.0

        average = sum(self.frame_times) / len(self.frame_times)
        if average <= 0:
            return 0.0

        return 1.0 / average

    @property
    def detection_ms(self) -> float:
        if not self.detection_times:
            return 0.0

        average = sum(self.detection_times) / len(self.detection_times)
        return average * 1000.0


class PlaneTrack:
    def __init__(self, track_id: int, detection: Detection, use_kalman: bool = True):
        self.id = track_id
        self.box = detection.box
        self.name = detection.name
        self.score = detection.score
        self.points = deque(maxlen=80)
        self.points.append(detection.center)
        self.missed = 0
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.kalman = self.make_kalman(detection.center) if use_kalman and has_filterpy else None

    def make_kalman(self, center: Tuple[int, int]) -> Any:
        x, y = center

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

        kf.x = np.array([[x], [y], [0], [0]], dtype=float)
        kf.P *= 100.0
        kf.R *= 8.0
        kf.Q *= 0.05

        return kf

    def update(self, detection: Detection) -> None:
        center = detection.center

        if self.kalman is not None:
            self.kalman.predict()
            self.kalman.update(np.array([[center[0]], [center[1]]], dtype=float))
            center = int(self.kalman.x[0, 0]), int(self.kalman.x[1, 0])

        self.box = detection.box
        self.name = detection.name
        self.score = detection.score
        self.points.append(center)
        self.missed = 0
        self.updated_at = time.time()

    def mark_missing(self) -> None:
        self.missed += 1

        if self.kalman is not None:
            self.kalman.predict()
            center = int(self.kalman.x[0, 0]), int(self.kalman.x[1, 0])
            self.points.append(center)

    def center(self) -> Tuple[int, int]:
        return self.points[-1]

    def speed(self, fps: float) -> float:
        if len(self.points) < 2:
            return 0.0

        recent_points = list(self.points)[-10:]
        total_distance = 0.0

        for i in range(1, len(recent_points)):
            x1, y1 = recent_points[i - 1]
            x2, y2 = recent_points[i]
            total_distance += math.hypot(x2 - x1, y2 - y1)

        steps = max(1, len(recent_points) - 1)
        return total_distance / steps * fps

    def direction(self) -> str:
        if len(self.points) < 2:
            return "steady"

        recent_points = list(self.points)[-12:]
        start_x, start_y = recent_points[0]
        end_x, end_y = recent_points[-1]

        dx = end_x - start_x
        dy = end_y - start_y

        if math.hypot(dx, dy) < 6:
            return "steady"

        words = []

        if abs(dy) > 4:
            words.append("down" if dy > 0 else "up")

        if abs(dx) > 4:
            words.append("right" if dx > 0 else "left")

        return "-".join(words) if words else "steady"


class PlaneTracker:
    def __init__(self, max_missing: int = 30, max_distance: float = 120.0, use_kalman: bool = True):
        self.max_missing = max_missing
        self.max_distance = max_distance
        self.use_kalman = use_kalman
        self.next_id = 1
        self.tracks: Dict[int, PlaneTrack] = {}

    def update(self, detections: List[Detection]) -> Dict[int, PlaneTrack]:
        if not detections:
            self.mark_all_missing()
            return self.tracks

        if not self.tracks:
            for detection in detections:
                self.add_track(detection)

            return self.tracks

        track_ids = list(self.tracks.keys())
        track_centers = np.array([self.tracks[track_id].center() for track_id in track_ids], dtype=float)
        detection_centers = np.array([detection.center for detection in detections], dtype=float)

        distances = self.get_distances(track_centers, detection_centers)
        matches = self.match_items(distances)

        used_tracks = set()
        used_detections = set()

        for track_index, detection_index in matches:
            distance = distances[track_index, detection_index]

            if distance > self.max_distance:
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
                self.add_track(detection)

        self.remove_old_tracks()
        return self.tracks

    def add_track(self, detection: Detection) -> None:
        self.tracks[self.next_id] = PlaneTrack(
            track_id=self.next_id,
            detection=detection,
            use_kalman=self.use_kalman
        )

        self.next_id += 1

    def mark_all_missing(self) -> None:
        for track in self.tracks.values():
            track.mark_missing()

        self.remove_old_tracks()

    def remove_old_tracks(self) -> None:
        old_ids = []

        for track_id, track in self.tracks.items():
            if track.missed > self.max_missing:
                old_ids.append(track_id)

        for track_id in old_ids:
            del self.tracks[track_id]

    def get_distances(self, tracks: np.ndarray, detections: np.ndarray) -> np.ndarray:
        difference = tracks[:, None, :] - detections[None, :, :]
        return np.sqrt(np.sum(difference * difference, axis=2))

    def match_items(self, distances: np.ndarray) -> List[Tuple[int, int]]:
        if has_scipy:
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
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.35,
        iou: float = 0.45,
        targets: Optional[List[str]] = None,
        device: str = "auto",
        max_detections: int = 100
    ):
        if not has_yolo:
            raise RuntimeError("Ultralytics is not installed. Run: pip install ultralytics")

        self.model_name = model_name
        self.confidence = confidence
        self.iou = iou
        self.targets = [name.lower() for name in (targets or ["airplane"])]
        self.device = self.choose_device(device)
        self.max_detections = max_detections
        self.model = YOLO(model_name)
        self.names = getattr(self.model, "names", {})

    def choose_device(self, device: str) -> str:
        if device != "auto":
            return device

        if has_torch and torch.cuda.is_available():
            return "cuda"

        if has_torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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
            name = str(self.names.get(class_id, class_id)).lower()

            if not self.is_target(name):
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
                    name=name
                )
            )

        return detections

    def is_target(self, name: str) -> bool:
        return any(target in name for target in self.targets)


def read_source(value: str) -> Union[int, str]:
    value = value.strip()

    if value.isdigit():
        return int(value)

    return value


def draw_detections(frame: np.ndarray, detections: List[Detection]) -> None:
    for detection in detections:
        cv2.rectangle(
            frame,
            (detection.left, detection.top),
            (detection.right, detection.bottom),
            (0, 180, 0),
            2
        )

        text = f"{detection.name} {detection.score:.2f}"

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
        if track.missed > 3:
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

            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], (0, 128, 255), 2)


def draw_info(frame: np.ndarray, speed_meter: SpeedMeter, detection_count: int, track_count: int) -> None:
    text = (
        f"FPS: {speed_meter.fps:.1f} | "
        f"Detection: {speed_meter.detection_ms:.1f} ms | "
        f"Planes: {detection_count} | "
        f"Tracks: {track_count}"
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


def make_video_writer(path: str, video: cv2.VideoCapture) -> cv2.VideoWriter:
    fps = video.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 1:
        fps = 30.0

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def run(args: argparse.Namespace) -> None:
    source = read_source(args.source)
    video = cv2.VideoCapture(source)

    if not video.isOpened():
        raise RuntimeError(f"Could not open this source: {args.source}")

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
        use_kalman=not args.no_kalman
    )

    speed_meter = SpeedMeter()
    writer = make_video_writer(args.output, video) if args.output else None

    try:
        while True:
            frame_start = time.perf_counter()

            ok, frame = video.read()

            if not ok:
                break

            detection_start = time.perf_counter()
            detections = detector.detect(frame)
            speed_meter.add_detection(time.perf_counter() - detection_start)

            tracks = tracker.update(detections)

            output = frame.copy()

            draw_detections(output, detections)
            draw_tracks(output, tracks, speed_meter.fps or 30.0, args.trails)
            draw_info(output, speed_meter, len(detections), len(tracks))

            if writer is not None:
                writer.write(output)

            if args.display:
                cv2.imshow("Plane Monitor", output)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            speed_meter.add_frame(time.perf_counter() - frame_start)

    finally:
        video.release()

        if writer is not None:
            writer.release()

        if args.display:
            cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plane_monitor",
        description="Detect planes, follow them, and show their movement."
    )

    parser.add_argument("--source", default="0", help="Camera number, video file, or stream link")
    parser.add_argument("--output", help="Save the result as a video file")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model name or path")
    parser.add_argument("--confidence", type=float, default=0.35, help="Minimum detection score")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU value")
    parser.add_argument("--target", nargs="+", default=["airplane"], help="Objects to detect")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--display", action="store_true", help="Show the video window")
    parser.add_argument("--trails", action="store_true", help="Show movement lines")
    parser.add_argument("--no-kalman", action="store_true", help="Turn off smooth tracking")
    parser.add_argument("--max-missing", type=int, default=30, help="Frames to keep a missing plane")
    parser.add_argument("--max-distance", type=float, default=120.0, help="Maximum distance for matching")
    parser.add_argument("--max-detections", type=int, default=100, help="Maximum detections per frame")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        run(args)
    except KeyboardInterrupt:
        print("Stopped by user")
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
