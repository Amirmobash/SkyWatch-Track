#!/usr/bin/env python3
"""
Airplane Monitor & Direction Checker – Professional Edition

Author: Amir Mobasheraghdam (nivta.de)
License: MIT
Version: 2.0

Purpose:
    - Capture frames from a webcam or video file.
    - Detect airplanes (or any COCO classes) using YOLOv5/YOLOv8.
    - Track objects with centroid tracking (optional Kalman filtering).
    - Compute heading vectors and compare with dominant scene direction (dense optical flow).
    - Generate alerts when an object's heading deviates from the scene flow.
    - Save logs (CSV, JSON) and snapshots on alerts.
    - Visualize results in real time with rich overlays.
"""

import argparse
import base64
import csv
import datetime
import json
import logging
import math
import os
import sys
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import cv2
import numpy as np

# Optional dependencies
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("Scipy not installed. Falling back to greedy matching for tracking.")

try:
    from filterpy.kalman import KalmanFilter
    HAS_FILTERPY = True
except ImportError:
    HAS_FILTERPY = False
    warnings.warn("Filterpy not installed. Kalman filtering disabled.")

# ----------------------------------------------------------------------
# Metadata & Author Information
# ----------------------------------------------------------------------
AUTHOR_NAME = "Amir Mobasheraghdam"
AUTHOR_SITE = "nivta.de"
HIDDEN_METADATA = {
    "author_b64": base64.b64encode(AUTHOR_NAME.encode()).decode(),
    "site_b64": base64.b64encode(AUTHOR_SITE.encode()).decode(),
    "date": "2025-03-15",
    "version": "2.0"
}

def reveal_author() -> Dict[str, str]:
    """Return author information."""
    return {
        "author": AUTHOR_NAME,
        "site": AUTHOR_SITE,
        "hidden": HIDDEN_METADATA
    }

# ----------------------------------------------------------------------
# Geometry Helpers
# ----------------------------------------------------------------------
def angle_between_vectors(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Smallest angle (degrees) between two vectors."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cosang))

def unit_vector(v: Tuple[float, float]) -> Tuple[float, float]:
    """Return unit vector."""
    n = math.hypot(*v)
    return (0.0, 0.0) if n == 0 else (v[0] / n, v[1] / n)

def vector_to_compass(v: Tuple[float, float]) -> str:
    """
    Convert image vector (dx, dy) to compass direction.
    In image coordinates y increases downwards; upward is considered North.
    """
    dx, dy = v
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "Static"
    angle = math.degrees(math.atan2(-dy, dx))   # 0° = East
    angle = (angle + 360.0) % 360.0
    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int(((angle + 22.5) % 360) / 45)
    return f"{dirs[idx]} ({angle:.0f}°)"

# ----------------------------------------------------------------------
# Tracked Object with optional Kalman filter
# ----------------------------------------------------------------------
class TrackedObject:
    """Holds state of one tracked object."""
    def __init__(self, obj_id: int, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int],
                 timestamp: str, use_kalman: bool = False):
        self.id = obj_id
        self.centroids = deque(maxlen=30)          # for heading calculation
        self.centroids.append(centroid)
        self.bbox = bbox
        self.disappeared = 0
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.alerted = False
        self.alert_cooldown = 0                    # frames until next alert
        self.use_kalman = use_kalman and HAS_FILTERPY
        if self.use_kalman:
            self.kalman = self._create_kalman()
            self.kalman.predict()
            self.kalman.update(centroid)

    def _create_kalman(self) -> KalmanFilter:
        """Create a simple constant-velocity Kalman filter."""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0
        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        kf.R = np.eye(2) * 5.0
        kf.Q = np.eye(4) * 0.1
        kf.P *= 10.0
        return kf

    def update(self, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int], timestamp: str) -> None:
        """Update object with new detection."""
        if self.use_kalman:
            self.kalman.predict()
            self.kalman.update(centroid)
            filtered = (int(self.kalman.x[0]), int(self.kalman.x[1]))
            self.centroids.append(filtered)
        else:
            self.centroids.append(centroid)
        self.bbox = bbox
        self.disappeared = 0
        self.last_seen = timestamp
        if self.alert_cooldown > 0:
            self.alert_cooldown -= 1

    def mark_missing(self) -> None:
        """Mark object as missing in current frame."""
        self.disappeared += 1
        if self.use_kalman:
            self.kalman.predict()
            self.centroids.append((int(self.kalman.x[0]), int(self.kalman.x[1])))

    def compute_heading_vector(self) -> Tuple[float, float]:
        """Compute heading from oldest to newest centroid."""
        if len(self.centroids) < 2:
            return (0.0, 0.0)
        p0 = self.centroids[0]
        p1 = self.centroids[-1]
        return (p1[0] - p0[0], p1[1] - p0[1])

    def last_centroid(self) -> Tuple[int, int]:
        """Return the most recent centroid (filtered if Kalman is used)."""
        if self.use_kalman:
            return (int(self.kalman.x[0]), int(self.kalman.x[1]))
        return tuple(self.centroids[-1])

# ----------------------------------------------------------------------
# Centroid Tracker (Hungarian or greedy association)
# ----------------------------------------------------------------------
class CentroidTracker:
    """Tracks objects across frames using centroid distance and optional Kalman."""
    def __init__(self, max_disappeared: int = 20, max_distance: float = 100.0,
                 use_kalman: bool = False, use_hungarian: bool = True):
        self.next_id = 1
        self.objects: Dict[int, TrackedObject] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.use_kalman = use_kalman
        self.use_hungarian = use_hungarian and HAS_SCIPY

    def register(self, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int],
                 timestamp: str) -> TrackedObject:
        """Register a new object."""
        obj = TrackedObject(self.next_id, centroid, bbox, timestamp, self.use_kalman)
        self.objects[self.next_id] = obj
        self.next_id += 1
        return obj

    def deregister(self, obj_id: int) -> None:
        """Remove an object from tracking."""
        self.objects.pop(obj_id, None)

    def update(self, detections: List[Tuple[int, int, Tuple[int, int, int, int]]],
               timestamp: str) -> Dict[int, TrackedObject]:
        """
        Update tracker with current frame detections.
        detections: list of (cx, cy, (x1, y1, x2, y2))
        """
        if not detections:
            # Mark all as missing
            for obj in list(self.objects.values()):
                obj.mark_missing()
                if obj.disappeared > self.max_disappeared:
                    self.deregister(obj.id)
            return self.objects

        input_centroids = np.array([[d[0], d[1]] for d in detections])
        input_bboxes = [d[2] for d in detections]

        # No existing objects -> register all
        if not self.objects:
            for c, b in zip(input_centroids, input_bboxes):
                self.register(tuple(c), b, timestamp)
            return self.objects

        # Prepare existing object centroids
        obj_ids = list(self.objects.keys())
        obj_centroids = np.array([self.objects[oid].last_centroid() for oid in obj_ids])

        # Compute cost matrix (Euclidean distance)
        cost_matrix = distance.cdist(obj_centroids, input_centroids)

        # Association
        if self.use_hungarian and cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned_rows = set(row_ind)
            assigned_cols = set(col_ind)
        else:
            # Greedy assignment
            rows = cost_matrix.min(axis=1).argsort()
            cols = cost_matrix.argmin(axis=1)[rows]
            assigned_rows, assigned_cols = set(), set()
            for r, c in zip(rows, cols):
                if r in assigned_rows or c in assigned_cols:
                    continue
                assigned_rows.add(r)
                assigned_cols.add(c)

        # Update matched objects
        for r, c in zip(assigned_rows, assigned_cols):
            if cost_matrix[r, c] > self.max_distance:
                continue
            oid = obj_ids[r]
            centroid = tuple(input_centroids[c])
            bbox = input_bboxes[c]
            self.objects[oid].update(centroid, bbox, timestamp)

        # Mark unmatched existing objects as missing
        for i, oid in enumerate(obj_ids):
            if i not in assigned_rows:
                self.objects[oid].mark_missing()
                if self.objects[oid].disappeared > self.max_disappeared:
                    self.deregister(oid)

        # Register unmatched new detections
        for j in range(len(input_centroids)):
            if j not in assigned_cols:
                self.register(tuple(input_centroids[j]), input_bboxes[j], timestamp)

        return self.objects

# ----------------------------------------------------------------------
# YOLO Detector Wrapper (Ultralytics or torch.hub)
# ----------------------------------------------------------------------
class YOLODetector:
    """Unified YOLO detector supporting both Ultralytics and torch.hub."""
    def __init__(self, model_name: str = "yolov5s", use_ultralytics: bool = False,
                 device: str = "cpu", conf_threshold: float = 0.35,
                 target_classes: List[Union[str, int]] = None):
        self.model_name = model_name
        self.use_ultralytics = use_ultralytics
        self.device = device
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes or ["airplane"]
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model (Ultralytics preferred, else torch.hub)."""
        if self.use_ultralytics:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_name)
                logging.info(f"Loaded Ultralytics YOLO model: {self.model_name}")
                return
            except Exception as e:
                logging.warning(f"Ultralytics failed: {e}. Falling back to torch.hub.")
                self.use_ultralytics = False

        # Fallback to torch.hub YOLOv5
        try:
            import torch
            self.model = torch.hub.load("ultralytics/yolov5", self.model_name, pretrained=True)
            self.model.eval()
            if self.device != "cpu":
                self.model.to(self.device)
            logging.info(f"Loaded torch.hub YOLOv5 model: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Could not load any YOLO model: {e}")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection on a BGR frame.
        Returns list of dicts with keys: xmin, ymin, xmax, ymax, conf, class, name.
        """
        if self.use_ultralytics:
            results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
            if isinstance(results, list):
                results = results[0]
            boxes = results.boxes
            detections = []
            for b in boxes:
                cls = int(b.cls.cpu().numpy())
                conf = float(b.conf.cpu().numpy())
                x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
                name = self.model.names.get(cls, str(cls))
                if self._is_target(name, cls):
                    detections.append({
                        "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
                        "conf": conf, "class": cls, "name": name
                    })
            return detections
        else:
            # torch.hub YOLOv5
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(rgb)
            xyxy = results.xyxy[0].cpu().numpy()
            names = results.names if hasattr(results, "names") else self.model.names
            detections = []
            for row in xyxy:
                x1, y1, x2, y2, conf, cls = row
                cls = int(cls)
                if conf >= self.conf_threshold and self._is_target(names.get(cls, str(cls)), cls):
                    detections.append({
                        "xmin": int(x1), "ymin": int(y1), "xmax": int(x2), "ymax": int(y2),
                        "conf": float(conf), "class": cls, "name": names.get(cls, str(cls))
                    })
            return detections

    def _is_target(self, name: str, class_id: int) -> bool:
        """Check if detection matches target classes (name or ID)."""
        if not self.target_classes:
            return True
        if isinstance(self.target_classes[0], int):
            return class_id in self.target_classes
        else:
            return any(t.lower() in name.lower() for t in self.target_classes)

# ----------------------------------------------------------------------
# Dominant Direction Estimator (Dense Optical Flow)
# ----------------------------------------------------------------------
class DominantDirectionEstimator:
    """Estimates dominant motion direction using Farneback optical flow."""
    def __init__(self, smoothing_frames: int = 30, skip_frames: int = 1,
                 pyr_scale: float = 0.5, levels: int = 3, winsize: int = 15,
                 iterations: int = 3, poly_n: int = 5, poly_sigma: float = 1.2):
        self.smoothing_frames = smoothing_frames
        self.skip_frames = skip_frames          # process flow every N frames
        self.frame_counter = 0
        self.prev_gray = None
        self.history = deque(maxlen=smoothing_frames)
        self.flow_params = {
            "pyr_scale": pyr_scale, "levels": levels, "winsize": winsize,
            "iterations": iterations, "poly_n": poly_n, "poly_sigma": poly_sigma,
            "flags": 0
        }

    def feed_frame(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Feed a new frame and return dominant flow vector (dx, dy) or None."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        self.frame_counter += 1
        if self.frame_counter % self.skip_frames != 0:
            # Reuse previous flow (optional: still update gray but skip heavy calc)
            if self.history:
                arr = np.array(self.history)
                return (float(np.nanmean(arr[:, 0])), float(np.nanmean(arr[:, 1])))
            return None

        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, **self.flow_params)
        self.prev_gray = gray

        # Compute mean flow (robust to outliers by using nanmean)
        fx = np.nanmean(flow[..., 0])
        fy = np.nanmean(flow[..., 1])
        self.history.append((fx, fy))

        if not self.history:
            return (0.0, 0.0)
        arr = np.array(self.history)
        return (float(np.nanmean(arr[:, 0])), float(np.nanmean(arr[:, 1])))

# ----------------------------------------------------------------------
# Main Application
# ----------------------------------------------------------------------
@dataclass
class AppConfig:
    """Configuration dataclass with defaults."""
    source: str = "0"
    model: str = "yolov5s"
    ultralytics: bool = False
    device: str = "cpu"
    confidence: float = 0.35
    classes: List[str] = field(default_factory=lambda: ["airplane"])
    max_disappeared: int = 20
    assoc_distance: float = 100.0
    use_hungarian: bool = True
    kalman: bool = False
    dom_smoothing: int = 30
    dom_skip_frames: int = 1
    angle_threshold: float = 45.0
    alert_cooldown: int = 10          # frames before triggering another alert for same object
    snapshot_dir: str = "snapshots"
    log_csv: str = "airplane_events.csv"
    log_json: str = "airplane_events.json"
    save_video: bool = False
    output_video: str = "output.avi"
    width: int = 1280
    height: int = 720
    resize: bool = False
    max_frames: int = 0
    show_fps: bool = True
    config_file: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        """Create config from dictionary (e.g., JSON)."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

class AirplaneMonitor:
    """Main application class."""
    def __init__(self, config: AppConfig):
        self.config = config
        self.detector = YOLODetector(
            model_name=config.model,
            use_ultralytics=config.ultralytics,
            device=config.device,
            conf_threshold=config.confidence,
            target_classes=config.classes
        )
        self.tracker = CentroidTracker(
            max_disappeared=config.max_disappeared,
            max_distance=config.assoc_distance,
            use_kalman=config.kalman,
            use_hungarian=config.use_hungarian
        )
        self.dom_est = DominantDirectionEstimator(
            smoothing_frames=config.dom_smoothing,
            skip_frames=config.dom_skip_frames
        )
        self.log_events: List[Dict] = []
        self._setup_dirs()
        self.cap = None
        self.writer = None
        self.frame_width = config.width
        self.frame_height = config.height
        self.fps = 0
        self.last_frame_time = time.time()

    def _setup_dirs(self) -> None:
        """Create necessary directories."""
        Path(self.config.snapshot_dir).mkdir(parents=True, exist_ok=True)
        for f in [self.config.log_csv, self.config.log_json]:
            Path(f).parent.mkdir(parents=True, exist_ok=True)

    def open_video(self) -> None:
        """Open video source (camera or file)."""
        src = self.config.source
        if src.isdigit():
            src = int(src)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {self.config.source}")

        # Set desired resolution (may not be honored by all cameras)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        # Get actual properties
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
        logging.info(f"Video source opened: {actual_w}x{actual_h} @ {self.fps:.2f} fps")

        if self.config.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(self.config.output_video, fourcc, self.fps, (actual_w, actual_h))
            logging.info(f"Recording output to {self.config.output_video}")

    def close(self) -> None:
        """Release resources and save logs."""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

        # Save event logs
        if self.log_events:
            df = pd.DataFrame(self.log_events)
            df.to_csv(self.config.log_csv, index=False)
            with open(self.config.log_json, 'w') as f:
                json.dump(self.log_events, f, default=str, indent=2)
            logging.info(f"Saved {len(self.log_events)} events to {self.config.log_csv} and {self.config.log_json}")

    def run(self) -> None:
        """Main processing loop."""
        self.open_video()
        frame_idx = 0
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logging.info("End of stream or cannot read frame.")
                    break

                frame_idx += 1
                if self.config.max_frames and frame_idx > self.config.max_frames:
                    logging.info(f"Reached max_frames limit ({self.config.max_frames}).")
                    break

                timestamp = datetime.datetime.utcnow().isoformat() + "Z"

                # Optional resize for performance
                if self.config.resize:
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))

                # Estimate dominant motion
                dom_vec = self.dom_est.feed_frame(frame)

                # Detect objects
                detections = self.detector.detect(frame)

                # Prepare for tracker
                tracker_input = []
                for d in detections:
                    cx = (d['xmin'] + d['xmax']) // 2
                    cy = (d['ymin'] + d['ymax']) // 2
                    bbox = (d['xmin'], d['ymin'], d['xmax'], d['ymax'])
                    tracker_input.append((cx, cy, bbox))

                # Update tracker
                tracked_objects = self.tracker.update(tracker_input, timestamp)

                # Visualize and process alerts
                self._draw_and_alert(frame, tracked_objects, dom_vec, frame_idx, timestamp)

                # Display FPS
                if self.config.show_fps:
                    current_time = time.time()
                    fps_display = 1.0 / (current_time - self.last_frame_time) if frame_idx > 1 else 0
                    self.last_frame_time = current_time
                    cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Show output
                cv2.imshow("Airplane Monitor - Amir Mobasheraghdam", frame)
                if self.writer:
                    self.writer.write(frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logging.info("Quit requested by user.")
                    break
                if key == ord('s'):
                    snap_path = Path(self.config.snapshot_dir) / f"manual_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(snap_path), frame)
                    logging.info(f"Manual snapshot saved: {snap_path}")

        except KeyboardInterrupt:
            logging.info("Interrupted by user.")
        except Exception as e:
            logging.exception(f"Unexpected error: {e}")
        finally:
            self.close()

    def _draw_and_alert(self, frame: np.ndarray, tracked: Dict[int, TrackedObject],
                        dom_vec: Optional[Tuple[float, float]], frame_idx: int, timestamp: str) -> None:
        """Draw bounding boxes, headings, and handle alerts."""
        h, w = frame.shape[:2]

        for oid, obj in tracked.items():
            x1, y1, x2, y2 = obj.bbox
            cx, cy = obj.last_centroid()
            heading_vec = obj.compute_heading_vector()
            heading_unit = unit_vector(heading_vec)
            compass = vector_to_compass(heading_vec)

            # Check alert condition
            alert = False
            angle_diff = None
            if dom_vec is not None and obj.alert_cooldown == 0:
                angle_diff = angle_between_vectors(heading_vec, dom_vec)
                speed = math.hypot(*heading_vec)
                if angle_diff > self.config.angle_threshold and speed > 2.0:
                    alert = True

            # Determine color
            color = (0, 0, 255) if alert else (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Draw label
            label = f"ID{oid} {compass}"
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Centroid
            cv2.circle(frame, (cx, cy), 4, color, -1)
            # Heading arrow
            tip = (int(cx + heading_unit[0] * 50), int(cy + heading_unit[1] * 50))
            cv2.arrowedLine(frame, (cx, cy), tip, color, 2, tipLength=0.3)
            # Trail
            trail = list(obj.centroids)[-10:]
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i-1], trail[i], (200, 200, 200), 1)

            # Alert handling
            if alert and not obj.alerted:
                obj.alerted = True
                obj.alert_cooldown = self.config.alert_cooldown
                snap_name = Path(self.config.snapshot_dir) / f"alert_id{oid}_frame{frame_idx:06d}.jpg"
                cv2.imwrite(str(snap_name), frame)
                event = {
                    "timestamp": timestamp,
                    "frame": frame_idx,
                    "object_id": oid,
                    "compass": compass,
                    "angle_vs_dom": round(angle_diff, 2) if angle_diff is not None else None,
                    "snapshot": str(snap_name),
                    "first_seen": obj.first_seen,
                    "last_seen": obj.last_seen,
                }
                self.log_events.append(event)
                logging.info(f"ALERT: Object {oid} deviates by {angle_diff:.1f}°, snapshot saved.")

        # Draw dominant direction arrow
        if dom_vec is not None:
            center = (w // 2, h // 2)
            dv = unit_vector(dom_vec)
            tip = (int(center[0] + dv[0] * 100), int(center[1] + dv[1] * 100))
            cv2.arrowedLine(frame, center, tip, (255, 255, 0), 3, tipLength=0.3)
            cv2.putText(frame, f"Scene: {vector_to_compass(dom_vec)}", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Object count
        cv2.putText(frame, f"Airplanes: {len(tracked)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

# ----------------------------------------------------------------------
# Command-line Interface
# ----------------------------------------------------------------------
def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Airplane Monitor with Direction Checking - Professional Edition")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam or file path)")
    parser.add_argument("--model", type=str, default="yolov5s", help="YOLO model name/path")
    parser.add_argument("--ultralytics", action="store_true", help="Use Ultralytics YOLO (preferred)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--confidence", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--classes", nargs="+", default=["airplane"], help="Target class names or IDs")
    parser.add_argument("--max-disappeared", type=int, default=20, help="Frames before losing a track")
    parser.add_argument("--assoc-distance", type=float, default=100.0, help="Max pixel distance for association")
    parser.add_argument("--no-hungarian", action="store_true", help="Disable Hungarian algorithm (use greedy)")
    parser.add_argument("--kalman", action="store_true", help="Use Kalman filter per track")
    parser.add_argument("--dom-smoothing", type=int, default=30, help="Frames to smooth optical flow")
    parser.add_argument("--dom-skip-frames", type=int, default=1, help="Compute flow every N frames")
    parser.add_argument("--angle-threshold", type=float, default=45.0, help="Max allowed angle deviation (deg)")
    parser.add_argument("--alert-cooldown", type=int, default=10, help="Cooldown frames between alerts per object")
    parser.add_argument("--snapshot-dir", type=str, default="snapshots", help="Directory for snapshots")
    parser.add_argument("--log-csv", type=str, default="airplane_events.csv", help="CSV log file")
    parser.add_argument("--log-json", type=str, default="airplane_events.json", help="JSON log file")
    parser.add_argument("--save-video", action="store_true", help="Save output video")
    parser.add_argument("--output-video", type=str, default="output.avi", help="Output video filename")
    parser.add_argument("--width", type=int, default=1280, help="Resize width (if --resize)")
    parser.add_argument("--height", type=int, default=720, help="Resize height (if --resize)")
    parser.add_argument("--resize", action="store_true", help="Resize frames to width/height")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = infinite)")
    parser.add_argument("--no-fps", action="store_true", help="Hide FPS display")
    parser.add_argument("--config", type=str, help="JSON config file (overrides command line)")

    args = parser.parse_args()

    # Load config from JSON if provided
    config_dict = vars(args)
    if args.config:
        try:
            with open(args.config, 'r') as f:
                file_cfg = json.load(f)
            config_dict.update(file_cfg)
        except Exception as e:
            logging.error(f"Failed to load config file: {e}")

    # Convert back to AppConfig
    config_dict["use_hungarian"] = not config_dict.pop("no_hungarian", False)
    config_dict["show_fps"] = not config_dict.pop("no_fps", False)
    return AppConfig.from_dict(config_dict)

# ----------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------
def main() -> None:
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Display author info
    author_info = reveal_author()
    logging.info(f"Airplane Monitor by {author_info['author']} ({author_info['site']}) - Version 2.0")

    config = parse_args()
    if config.kalman and not HAS_FILTERPY:
        logging.warning("Kalman filtering requested but filterpy not installed. Disabling Kalman.")
        config.kalman = False

    monitor = AirplaneMonitor(config)
    monitor.run()

if __name__ == "__main__":
    main()
