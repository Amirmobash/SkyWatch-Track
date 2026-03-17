#!/usr/bin/env python3
"""
Airplane Monitor & Direction Checker
Author: Amir Mobasheraghdam (nivta.de) – see hidden metadata below
Enhanced version with full configurability, logging, and optional Kalman tracking.

Purpose:
 - Capture frames from a webcam or video file.
 - Detect airplanes (or any user‑defined COCO classes) using YOLOv5/YOLOv8.
 - Track objects with a centroid tracker (optionally with Kalman filtering).
 - Compute heading vectors and compare with a dominant scene direction (dense optical flow).
 - Generate alerts when an object's heading deviates from the scene flow.
 - Save logs (CSV, JSON) and snapshots on alerts.
 - Visualize results in real time.
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
from collections import deque, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance

# ----------------------------------------------------------------------
# Optional Kalman filter for smoother tracking
try:
    from filterpy.kalman import KalmanFilter
    HAS_FILTERPY = True
except ImportError:
    HAS_FILTERPY = False

# ----------------------------------------------------------------------
# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AirplaneMonitor")

# ----------------------------------------------------------------------
# Hidden metadata (base64) – just for fun
def _hidden_metadata():
    return {
        "author_b64": "QW1pcg==",
        "lastname_b64": "TW9iYXNoZXJhZ2hkYW0=",
        "site_b64": "bml2dGEuZGU=",
    }

def reveal_author():
    md = _hidden_metadata()
    return {
        "author": base64.b64decode(md["author_b64"]).decode(errors="ignore"),
        "lastname": base64.b64decode(md["lastname_b64"]).decode(errors="ignore"),
        "site": base64.b64decode(md["site_b64"]).decode(errors="ignore"),
    }

# ----------------------------------------------------------------------
# Geometry helpers
def angle_between_vectors(v1, v2):
    """Smallest angle (degrees) between two vectors."""
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    cosang = max(-1.0, min(1.0, dot / (n1*n2)))
    return math.degrees(math.acos(cosang))

def unit_vector(v):
    n = math.hypot(*v)
    return (0.0, 0.0) if n == 0 else (v[0]/n, v[1]/n)

def vector_to_compass(v):
    """Convert image vector (dx,dy) to compass direction (N, NE, E, ...)."""
    dx, dy = v
    if dx == 0 and dy == 0:
        return "Static"
    # In image coordinates y increases downwards; treat upward as north.
    angle = math.degrees(math.atan2(-dy, dx))   # 0° = East
    angle = (angle + 360.0) % 360.0
    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int(((angle + 22.5) % 360) / 45)
    return f"{dirs[idx]} ({angle:.0f}°)"

# ----------------------------------------------------------------------
# Tracked object with optional Kalman filter
class TrackedObject:
    """Holds state of one tracked object."""
    def __init__(self, obj_id, centroid, bbox, timestamp, use_kalman=False):
        self.id = obj_id
        self.centroids = deque(maxlen=30)          # for heading calculation
        self.centroids.append(centroid)
        self.bbox = bbox                            # (x1,y1,x2,y2)
        self.disappeared = 0
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.alerted = False
        self.use_kalman = use_kalman and HAS_FILTERPY
        if self.use_kalman:
            self.kalman = self._create_kalman()
            self.kalman.predict()
            self.kalman.update(centroid)

    def _create_kalman(self):
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

    def update(self, centroid, bbox, timestamp):
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

    def mark_missing(self):
        self.disappeared += 1
        if self.use_kalman:
            # still predict position even when not detected
            self.kalman.predict()
            self.centroids.append((int(self.kalman.x[0]), int(self.kalman.x[1])))

    def compute_heading_vector(self):
        if len(self.centroids) < 2:
            return (0.0, 0.0)
        p0 = self.centroids[0]
        p1 = self.centroids[-1]
        return (p1[0] - p0[0], p1[1] - p0[1])

    def last_centroid(self):
        if self.use_kalman:
            return (int(self.kalman.x[0]), int(self.kalman.x[1]))
        return tuple(self.centroids[-1])

# ----------------------------------------------------------------------
# Simple centroid tracker (can use Kalman internally)
class SimpleTracker:
    def __init__(self, max_disappeared=20, max_distance=100, use_kalman=False):
        self.next_id = 1
        self.objects = {}          # id -> TrackedObject
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.use_kalman = use_kalman

    def register(self, centroid, bbox, timestamp):
        obj = TrackedObject(self.next_id, centroid, bbox, timestamp, self.use_kalman)
        self.objects[self.next_id] = obj
        self.next_id += 1
        return obj

    def deregister(self, obj_id):
        self.objects.pop(obj_id, None)

    def update(self, detections, timestamp):
        """
        detections: list of (centroid_x, centroid_y, bbox)
        bbox = (x1,y1,x2,y2)
        """
        if not detections:
            # Mark all as disappeared
            for obj in list(self.objects.values()):
                obj.mark_missing()
                if obj.disappeared > self.max_disappeared:
                    self.deregister(obj.id)
            return self.objects

        input_centroids = np.array([[d[0], d[1]] for d in detections])
        input_bboxes = [d[2] for d in detections]

        if not self.objects:
            # First frame: register all detections
            for c, b in zip(input_centroids, input_bboxes):
                self.register(tuple(c), b, timestamp)
            return self.objects

        # Prepare object centroids
        obj_ids = list(self.objects.keys())
        obj_centroids = np.array([self.objects[oid].last_centroid() for oid in obj_ids])

        # Compute distance matrix
        D = distance.cdist(obj_centroids, input_centroids)

        # Greedy association
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_rows, assigned_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in assigned_rows or c in assigned_cols:
                continue
            if D[r, c] > self.max_distance:
                continue
            oid = obj_ids[r]
            centroid = tuple(input_centroids[c])
            bbox = input_bboxes[c]
            self.objects[oid].update(centroid, bbox, timestamp)
            assigned_rows.add(r)
            assigned_cols.add(c)

        # Mark unassigned objects as missing
        for i, oid in enumerate(obj_ids):
            if i not in assigned_rows:
                self.objects[oid].mark_missing()
                if self.objects[oid].disappeared > self.max_disappeared:
                    self.deregister(oid)

        # Register new detections
        for j in range(len(input_centroids)):
            if j not in assigned_cols:
                self.register(tuple(input_centroids[j]), input_bboxes[j], timestamp)

        return self.objects

# ----------------------------------------------------------------------
# YOLO detector wrapper (supports ultralytics & torch.hub)
class Detector:
    def __init__(self, model_name="yolov5s", use_ultralytics=False, device="cpu",
                 conf_threshold=0.35, target_classes=None):
        self.use_ultralytics = use_ultralytics
        self.device = device
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes or ["airplane"]   # list of names or ids
        self.model = None
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        if self.use_ultralytics:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_name)
                logger.info(f"Loaded ultralytics YOLO model: {self.model_name}")
                return
            except Exception as e:
                logger.warning(f"ultralytics failed, falling back to torch.hub: {e}")
                self.use_ultralytics = False

        # Fallback to torch.hub YOLOv5
        try:
            import torch
            self.model = torch.hub.load("ultralytics/yolov5", self.model_name, pretrained=True)
            self.model.eval()
            if self.device != "cpu":
                self.model.to(self.device)
            logger.info(f"Loaded torch.hub YOLOv5 model: {self.model_name}")
        except Exception as e:
            logger.error(f"Could not load any YOLO model: {e}")
            raise

    def detect(self, frame):
        """Return list of detections: [{'xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class', 'name'}]"""
        results = []
        if self.use_ultralytics and hasattr(self.model, "predict"):
            out = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)
            if isinstance(out, list):
                out = out[0]
            boxes = out.boxes
            for b in boxes:
                cls = int(b.cls.cpu().numpy())
                conf = float(b.conf.cpu().numpy())
                x1, y1, x2, y2 = map(float, b.xyxy[0].cpu().numpy())
                name = self.model.names.get(cls, str(cls))
                if self._is_target(name, cls):
                    results.append({
                        "xmin": int(x1), "ymin": int(y1), "xmax": int(x2), "ymax": int(y2),
                        "conf": conf, "class": cls, "name": name
                    })
            return results

        # torch.hub YOLOv5
        import torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.model(rgb)
        xyxy = res.xyxy[0].cpu().numpy()
        names = res.names if hasattr(res, "names") else self.model.names
        for row in xyxy:
            x1, y1, x2, y2, conf, cls = row
            cls = int(cls)
            name = names.get(cls, str(cls))
            if conf >= self.conf_threshold and self._is_target(name, cls):
                results.append({
                    "xmin": int(x1), "ymin": int(y1), "xmax": int(x2), "ymax": int(y2),
                    "conf": float(conf), "class": cls, "name": name
                })
        return results

    def _is_target(self, name, class_id):
        """Check if detection belongs to one of the target classes."""
        if isinstance(self.target_classes[0], int):
            return class_id in self.target_classes
        else:
            return any(t in name.lower() for t in self.target_classes)

# ----------------------------------------------------------------------
# Optical flow dominant direction estimator
class DominantDirectionEstimator:
    def __init__(self, smoothing_frames=30):
        self.prev_gray = None
        self.history = deque(maxlen=smoothing_frames)

    def feed_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        self.prev_gray = gray
        fx = np.nanmean(flow[..., 0])
        fy = np.nanmean(flow[..., 1])
        self.history.append((fx, fy))
        if len(self.history) == 0:
            return (0.0, 0.0)
        arr = np.array(self.history)
        return (float(np.nanmean(arr[:, 0])), float(np.nanmean(arr[:, 1])))

# ----------------------------------------------------------------------
# Main application
class AirplaneMonitorApp:
    def __init__(self, config):
        self.config = config
        self.detector = Detector(
            model_name=config.model,
            use_ultralytics=config.ultralytics,
            device=config.device,
            conf_threshold=config.confidence,
            target_classes=config.classes
        )
        self.tracker = SimpleTracker(
            max_disappeared=config.max_disappeared,
            max_distance=config.assoc_distance,
            use_kalman=config.kalman
        )
        self.dom_est = DominantDirectionEstimator(smoothing_frames=config.dom_smoothing)
        self.log_events = []
        self._setup_dirs()
        self.cap = None
        self.writer = None
        self.frame_width = config.width
        self.frame_height = config.height

    def _setup_dirs(self):
        Path(self.config.snapshot_dir).mkdir(parents=True, exist_ok=True)
        for f in [self.config.log_csv, self.config.log_json]:
            Path(f).parent.mkdir(parents=True, exist_ok=True)

    def open_video(self):
        src = self.config.source
        self.cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        # Attempt to set resolution (may not work with all cameras/files)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        if self.config.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 20
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.writer = cv2.VideoWriter(self.config.output_video, fourcc, fps, (w, h))

    def close(self):
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        # Save logs
        if self.log_events:
            df = pd.DataFrame(self.log_events)
            df.to_csv(self.config.log_csv, index=False)
            with open(self.config.log_json, 'w') as f:
                json.dump(self.log_events, f, default=str, indent=2)
            logger.info(f"Saved logs to {self.config.log_csv} and {self.config.log_json}")

    def run(self):
        self.open_video()
        if not self.cap or not self.cap.isOpened():
            logger.error(f"Cannot open video source: {self.config.source}")
            return

        frame_idx = 0
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("End of stream or cannot read frame.")
                    break

                timestamp = datetime.datetime.utcnow().isoformat() + "Z"
                frame_idx += 1

                # Resize if needed (for performance)
                if self.config.resize:
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))

                # Dominant direction from optical flow
                dom_vec = self.dom_est.feed_frame(frame)

                # Detect objects
                dets = self.detector.detect(frame)
                # Prepare tracker input
                dets_for_tracker = []
                for d in dets:
                    cx = (d['xmin'] + d['xmax']) // 2
                    cy = (d['ymin'] + d['ymax']) // 2
                    bbox = (d['xmin'], d['ymin'], d['xmax'], d['ymax'])
                    dets_for_tracker.append((cx, cy, bbox))

                # Update tracker
                tracked = self.tracker.update(dets_for_tracker, timestamp)

                # Visualization
                self._draw(frame, tracked, dom_vec, frame_idx, timestamp)

                # Show frame
                cv2.imshow("AirplaneMonitor", frame)
                if self.writer:
                    self.writer.write(frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested.")
                    break
                if key == ord('s'):
                    snap = Path(self.config.snapshot_dir) / f"manual_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(snap), frame)
                    logger.info(f"Manual snapshot saved: {snap}")

                if self.config.max_frames and frame_idx >= self.config.max_frames:
                    logger.info(f"Reached max_frames limit ({self.config.max_frames}).")
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self.close()

    def _draw(self, frame, tracked, dom_vec, frame_idx, timestamp):
        h, w = frame.shape[:2]
        # Draw tracked objects
        for oid, obj in tracked.items():
            x1, y1, x2, y2 = obj.bbox
            cx, cy = obj.last_centroid()
            heading_vec = obj.compute_heading_vector()
            heading_unit = unit_vector(heading_vec)
            compass = vector_to_compass(heading_vec)

            # Angle against dominant direction
            alert = False
            if dom_vec is not None:
                ang = angle_between_vectors(heading_vec, dom_vec)
                if ang > self.config.angle_threshold and math.hypot(*heading_vec) > 2.0:
                    alert = True

            color = (0, 0, 255) if alert else (0, 255, 0)
            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Label
            label = f"ID{oid} {compass}"
            cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Centroid
            cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)
            # Heading arrow
            tip = (int(cx + heading_unit[0]*50), int(cy + heading_unit[1]*50))
            cv2.arrowedLine(frame, (int(cx), int(cy)), tip, color, 2, tipLength=0.3)

            # Trail (last 10 positions)
            trail = list(obj.centroids)[-10:]
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i-1], trail[i], (200,200,200), 1)

            # Alert snapshot
            if alert and not obj.alerted:
                obj.alerted = True
                snap_name = Path(self.config.snapshot_dir) / f"alert_id{oid}_{frame_idx:06d}.jpg"
                cv2.imwrite(str(snap_name), frame)
                ev = {
                    "timestamp": timestamp,
                    "object_id": oid,
                    "compass": compass,
                    "angle_vs_dom": round(ang, 2) if dom_vec is not None else None,
                    "snapshot": str(snap_name),
                    "first_seen": obj.first_seen,
                    "last_seen": obj.last_seen,
                }
                self.log_events.append(ev)
                logger.info(f"ALERT: ID {oid} deviates by {ang:.1f}°, snapshot {snap_name}")

        # Draw dominant direction
        if dom_vec is not None:
            center = (w//2, h//2)
            dv = unit_vector(dom_vec)
            tip = (int(center[0] + dv[0]*100), int(center[1] + dv[1]*100))
            cv2.arrowedLine(frame, center, tip, (255,255,0), 3, tipLength=0.3)
            text = f"Scene: {vector_to_compass(dom_vec)}"
            cv2.putText(frame, text, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # Statistics
        cv2.putText(frame, f"Airplanes: {len(tracked)}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

# ----------------------------------------------------------------------
# Configuration (from argparse + optional JSON file)
class Config:
    def __init__(self, **kwargs):
        self.source = kwargs.get("source", "0")
        self.model = kwargs.get("model", "yolov5s")
        self.ultralytics = kwargs.get("ultralytics", False)
        self.device = kwargs.get("device", "cpu")
        self.confidence = kwargs.get("confidence", 0.35)
        self.classes = kwargs.get("classes", ["airplane"])
        self.max_disappeared = kwargs.get("max_disappeared", 20)
        self.assoc_distance = kwargs.get("assoc_distance", 100)
        self.kalman = kwargs.get("kalman", False)
        self.dom_smoothing = kwargs.get("dom_smoothing", 30)
        self.angle_threshold = kwargs.get("angle_threshold", 45.0)
        self.snapshot_dir = kwargs.get("snapshot_dir", "snapshots")
        self.log_csv = kwargs.get("log_csv", "airplane_events.csv")
        self.log_json = kwargs.get("log_json", "airplane_events.json")
        self.save_video = kwargs.get("save_video", False)
        self.output_video = kwargs.get("output_video", "output.avi")
        self.width = kwargs.get("width", 1280)
        self.height = kwargs.get("height", 720)
        self.resize = kwargs.get("resize", False)
        self.max_frames = kwargs.get("max_frames", 0)

def load_config_from_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Airplane monitor with direction checking")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam or path)")
    parser.add_argument("--model", type=str, default="yolov5s", help="YOLO model name/path")
    parser.add_argument("--ultralytics", action="store_true", help="Use ultralytics YOLO (if available)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--confidence", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--classes", nargs="+", default=["airplane"], help="Target class names or IDs")
    parser.add_argument("--max-disappeared", type=int, default=20, help="Frames before losing a track")
    parser.add_argument("--assoc-distance", type=int, default=100, help="Max pixel distance for association")
    parser.add_argument("--kalman", action="store_true", help="Use Kalman filter per track (if filterpy installed)")
    parser.add_argument("--dom-smoothing", type=int, default=30, help="Frames to smooth optical flow")
    parser.add_argument("--angle-threshold", type=float, default=45.0, help="Max allowed angle deviation")
    parser.add_argument("--snapshot-dir", type=str, default="snapshots", help="Directory for alert snapshots")
    parser.add_argument("--log-csv", type=str, default="airplane_events.csv", help="CSV log file")
    parser.add_argument("--log-json", type=str, default="airplane_events.json", help="JSON log file")
    parser.add_argument("--save-video", action="store_true", help="Save output video")
    parser.add_argument("--output-video", type=str, default="output.avi", help="Output video file")
    parser.add_argument("--width", type=int, default=1280, help="Frame width (if resizing)")
    parser.add_argument("--height", type=int, default=720, help="Frame height (if resizing)")
    parser.add_argument("--resize", action="store_true", help="Resize frames to width/height")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = infinite)")
    parser.add_argument("--config", type=str, help="JSON config file (overrides command line)")
    return parser.parse_args()

def main():
    args = parse_args()
    config_dict = vars(args)

    # If config file provided, load and merge (command line overrides file)
    if args.config:
        file_cfg = load_config_from_json(args.config)
        # Command line args override file
        file_cfg.update({k: v for k, v in config_dict.items() if v is not None})
        config_dict = file_cfg

    config = Config(**config_dict)

    # Show author info
    author = reveal_author()
    logger.info(f"Airplane Monitor by {author['author']} {author['lastname']} – {author['site']}")

    # Warn if Kalman requested but not available
    if config.kalman and not HAS_FILTERPY:
        logger.warning("Kalman filtering requested but filterpy not installed. Falling back to raw centroids.")
        config.kalman = False

    app = AirplaneMonitorApp(config)
    logger.info("Starting monitor. Press 'q' to quit, 's' to save a snapshot.")
    app.run()

if __name__ == "__main__":
    main()
