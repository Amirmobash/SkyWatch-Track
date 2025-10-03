#!/usr/bin/env python3
"""
Airplane Monitor & Direction Checker
Author: (hidden)  -- see hidden metadata function below
Language: English (code and UI strings)

Purpose:
 - Capture frames from webcam
 - Detect airplanes (COCO class "airplane") using YOLOv5/YOLOv8 (via torch.hub or ultralytics)
 - Track individual airplanes across frames (simple centroid tracker)
 - Compute heading vectors for each tracked airplane
 - Infer dominant scene direction using dense optical flow
 - Compare airplane headings vs dominant direction and produce alerts
 - Save logs, snapshots on alerts, and overlay visualization on frames
 - Minimal external dependencies; explained in README below.
"""

import argparse
import base64
import csv
import datetime
import json
import math
import os
import sys
import time
from collections import deque, defaultdict

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance

# ---- Configuration (tweak these) ----
CONFIDENCE_THRESHOLD = 0.35   # YOLO detection confidence threshold
IOU_THRESHOLD = 0.5          # for NMS (if used)
MAX_DISAPPEARED = 12         # how many frames to keep object without detection
MAX_DISTANCE_ASSOC = 100     # max pixels to associate detections to existing trackers
HEADING_HISTORY = 10         # number of previous centroids to keep to compute heading
DOMINANT_DIRECTION_SMOOTHING = 30  # frames to smooth optical flow-based expected direction
DIRECTION_MATCH_ANGLE_THRESHOLD = 45.0  # degrees allowed difference before alert
SNAPSHOT_DIR = "snapshots"
LOG_CSV = "airplane_events.csv"
LOG_JSON = "airplane_events.json"
SAVE_VIDEO = False
OUTPUT_VIDEO = "output.avi"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ---- Utility helpers ----
def ensure_dirs():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def angle_between_vectors(v1, v2):
    # return smallest angle in degrees between 2 vectors
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 == 0 or n2 == 0:
        return 180.0
    cosang = max(-1.0, min(1.0, dot / (n1*n2)))
    ang = math.degrees(math.acos(cosang))
    return ang

def unit_vector(v):
    n = math.hypot(v[0], v[1])
    if n == 0:
        return (0.0, 0.0)
    return (v[0]/n, v[1]/n)

def vector_to_compass(v):
    # Convert vector (dx, dy) in image coords (x to right, y down) to compass-like direction.
    # We'll treat image-up (negative y) as North.
    dx, dy = v
    if dx == 0 and dy == 0:
        return "Static"
    angle = math.degrees(math.atan2(-dy, dx))  # invert y because image y grows downwards; 0 deg = East
    # Normalize to [0,360)
    angle = (angle + 360.0) % 360.0
    # map to 8 directions
    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int(((angle + 22.5) % 360) / 45)
    return f"{dirs[idx]} ({angle:.0f}°)"

# ---- Hidden metadata (base64) ----
def hidden_metadata():
    # encoded pieces (so casual readers don't immediately spot name/site)
    pieces = {
        "author_b64": "QW1pcg==",  # "Amir"
        "lastname_b64": "TW9iYXNoZXJhZ2hkYW0=",  # "Mobasheraghdam"
        "site_b64": "bml2dGEuZGU=",  # "nivta.de"
    }
    return pieces

def reveal_hidden_metadata():
    md = hidden_metadata()
    return {
        "author": base64.b64decode(md["author_b64"]).decode(errors="ignore"),
        "lastname": base64.b64decode(md["lastname_b64"]).decode(errors="ignore"),
        "site": base64.b64decode(md["site_b64"]).decode(errors="ignore"),
    }

# ---- Simple Centroid Tracker + minimal state per object ----
class TrackedObject:
    def __init__(self, object_id, centroid, bbox, timestamp):
        self.id = object_id
        self.centroids = deque(maxlen=HEADING_HISTORY)
        self.centroids.append(centroid)
        self.bbox = bbox  # latest bbox (x1,y1,x2,y2)
        self.disappeared = 0
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.alerted = False

    def update(self, centroid, bbox, timestamp):
        self.centroids.append(centroid)
        self.bbox = bbox
        self.disappeared = 0
        self.last_seen = timestamp

    def mark_missing(self):
        self.disappeared += 1

    def compute_heading_vector(self):
        if len(self.centroids) < 2:
            return (0.0, 0.0)
        p0 = self.centroids[0]
        p1 = self.centroids[-1]
        return (p1[0] - p0[0], p1[1] - p0[1])  # dx, dy

    def last_centroid(self):
        return tuple(self.centroids[-1])

class SimpleTracker:
    def __init__(self, max_disappeared=MAX_DISAPPEARED, max_distance=MAX_DISTANCE_ASSOC):
        self.next_object_id = 1
        self.objects = dict()  # id -> TrackedObject
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox, timestamp):
        obj = TrackedObject(self.next_object_id, centroid, bbox, timestamp)
        self.objects[self.next_object_id] = obj
        self.next_object_id += 1
        return obj

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]

    def update(self, detections, timestamp):
        """
        detections: list of tuples [(centroid_x, centroid_y, bbox), ...]
        bbox = (x1,y1,x2,y2)
        """
        if len(detections) == 0:
            # mark all as disappeared
            for obj in list(self.objects.values()):
                obj.mark_missing()
                if obj.disappeared > self.max_disappeared:
                    self.deregister(obj.id)
            return self.objects

        input_centroids = np.array([[int(d[0]), int(d[1])] for d in detections])
        input_bboxes = [d[2] for d in detections]

        if len(self.objects) == 0:
            # register all
            for (c, b) in zip(input_centroids, input_bboxes):
                self.register(tuple(c), b, timestamp)
            return self.objects

        # build object id -> centroid array
        object_ids = list(self.objects.keys())
        object_centroids = np.array([self.objects[oid].last_centroid() for oid in object_ids])

        # distance matrix
        D = distance.cdist(object_centroids, input_centroids)
        # for each object find closest detection
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_rows, assigned_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in assigned_rows or c in assigned_cols:
                continue
            if D[r, c] > self.max_distance:
                continue
            oid = object_ids[r]
            centroid = tuple(input_centroids[c])
            bbox = input_bboxes[c]
            self.objects[oid].update(centroid, bbox, timestamp)
            assigned_rows.add(r)
            assigned_cols.add(c)

        # mark unassigned objects disappeared
        for i, oid in enumerate(object_ids):
            if i not in assigned_rows:
                self.objects[oid].mark_missing()
                if self.objects[oid].disappeared > self.max_disappeared:
                    self.deregister(oid)

        # register unassigned detections
        for j in range(len(input_centroids)):
            if j not in assigned_cols:
                self.register(tuple(input_centroids[j]), input_bboxes[j], timestamp)

        return self.objects

# ---- YOLO-based detector wrapper ----
class Detector:
    def __init__(self, model_name=None, use_ultralytics=False, device='cpu'):
        """
        model_name: e.g., 'yolov5s' (torch.hub) or path to .pt
        use_ultralytics: if True, try to use ultralytics YOLO interface (if installed)
        """
        self.use_ultralytics = use_ultralytics
        self.device = device
        self.model = None
        self.model_name = model_name or "yolov5s"
        self.load_model()

    def load_model(self):
        # Try ultralytics first if requested
        if self.use_ultralytics:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_name)
                print("[INFO] Loaded ultralytics YOLO model:", self.model_name)
                return
            except Exception as e:
                print("[WARN] ultralytics not available or failed to load:", e)
                self.use_ultralytics = False

        # fallback to torch.hub yolov5
        try:
            import torch
            self.model = torch.hub.load('ultralytics/yolov5', self.model_name, pretrained=True)
            self.model.eval()
            if self.device != 'cpu':
                self.model.to(self.device)
            print("[INFO] Loaded torch.hub YOLOv5 model:", self.model_name)
            return
        except Exception as e:
            print("[ERROR] Failed to load YOLO model via torch.hub:", e)
            raise RuntimeError("YOLO model not available. Please install ultralytics or allow torch.hub to download models.")

    def detect(self, frame):
        """
        Returns detections as list of dicts:
         [{'xmin':, 'ymin':, 'xmax':, 'ymax':, 'conf':, 'class':, 'name':}, ...]
        """
        results = []
        if self.use_ultralytics and hasattr(self.model, 'predict'):
            out = self.model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            # ultralytics returns a Results object or list; normalize
            if isinstance(out, list):
                out = out[0]
            boxes = out.boxes
            for b in boxes:
                cls = int(b.cls.cpu().numpy())
                conf = float(b.conf.cpu().numpy())
                x1, y1, x2, y2 = map(float, b.xyxy[0].cpu().numpy())
                name = self.model.names.get(cls, str(cls))
                results.append({'xmin': int(x1), 'ymin': int(y1), 'xmax': int(x2), 'ymax': int(y2), 'conf': conf, 'class': cls, 'name': name})
            return results

        # else torch.hub yolov5
        import torch
        # model expects PIL, ndarray or path. We can pass numpy array (BGR -> RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.model(rgb)  # inference
        # res.xyxy[0] -> [x1,y1,x2,y2,conf,class]
        if hasattr(res, 'xyxy'):
            xyxy = res.xyxy[0].cpu().numpy()
        else:
            xyxy = res[0].cpu().numpy()
        names = res.names if hasattr(res, 'names') else self.model.names
        for row in xyxy:
            x1, y1, x2, y2, conf, cls = row
            cls = int(cls)
            name = names.get(cls, str(cls))
            results.append({'xmin': int(x1), 'ymin': int(y1), 'xmax': int(x2), 'ymax': int(y2), 'conf': float(conf), 'class': cls, 'name': name})
        return results

# ---- Optical flow dominant direction estimator ----
class DominantDirectionEstimator:
    def __init__(self, smoothing_frames=DOMINANT_DIRECTION_SMOOTHING):
        self.prev_gray = None
        self.history = deque(maxlen=smoothing_frames)

    def feed_frame(self, frame):
        # compute dense optical flow relative to previous frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        self.prev_gray = gray
        # compute average flow vector
        fx = np.nanmean(flow[..., 0])
        fy = np.nanmean(flow[..., 1])
        # push to history
        self.history.append((fx, fy))
        # compute smoothed vector
        if len(self.history) == 0:
            return (0.0, 0.0)
        arr = np.array(self.history)
        avg = (float(np.nanmean(arr[:, 0])), float(np.nanmean(arr[:, 1])))
        return avg

# ---- Main monitoring application ----
class AirplaneMonitorApp:
    def __init__(self, src=0, device='cpu', model_name="yolov5s", use_ultralytics=False):
        self.src = int(src) if str(src).isdigit() else src
        self.device = device
        self.detector = Detector(model_name=model_name, use_ultralytics=use_ultralytics, device=device)
        self.tracker = SimpleTracker()
        self.dom_est = DominantDirectionEstimator()
        self.log_events = []
        ensure_dirs()
        self.cap = None
        self.writer = None

    def open_video(self):
        self.cap = cv2.VideoCapture(self.src)
        # attempt to set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if SAVE_VIDEO:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = int(self.cap.get(cv2.CAP_PROP_FPS) or 20)
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

    def close(self):
        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()
        # flush logs to CSV and JSON
        if len(self.log_events) > 0:
            df = pd.DataFrame(self.log_events)
            df.to_csv(LOG_CSV, index=False)
            with open(LOG_JSON, 'w') as f:
                json.dump(self.log_events, f, default=str, indent=2)
            print(f"[INFO] Saved event logs: {LOG_CSV}, {LOG_JSON}")

    def run(self, max_frames=0):
        self.open_video()
        if self.cap is None or not self.cap.isOpened():
            print("[ERROR] Cannot open video source:", self.src)
            return

        frame_idx = 0
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[INFO] End of stream or cannot read frame.")
                    break
                timestamp = datetime.datetime.utcnow().isoformat()
                frame_idx += 1

                # resize for speed if needed
                h, w = frame.shape[:2]
                # feed to optical flow estimator
                dom_vec = self.dom_est.feed_frame(frame)
                # detect with YOLO
                dets = self.detector.detect(frame)
                # filter only class 'airplane' (COCO name: 'airplane' or 'aeroplane' on some models)
                airplane_dets = []
                for d in dets:
                    nm = d.get('name', '').lower()
                    if 'airplane' in nm or 'aeroplane' in nm or 'plane' in nm:
                        cx = int((d['xmin'] + d['xmax']) / 2)
                        cy = int((d['ymin'] + d['ymax']) / 2)
                        airplane_dets.append((cx, cy, (d['xmin'], d['ymin'], d['xmax'], d['ymax']), d['conf']))

                # convert detections to tracker format (centroid,bbox)
                dets_for_tracker = [(c[0], c[1], c[2]) for c in airplane_dets]
                tracked = self.tracker.update(dets_for_tracker, timestamp)

                # For visualization: draw detections and tracked IDs
                for oid, obj in tracked.items():
                    x1, y1, x2, y2 = obj.bbox
                    cx, cy = obj.last_centroid()
                    heading_vec = obj.compute_heading_vector()
                    heading_unit = unit_vector(heading_vec)
                    compass = vector_to_compass(heading_vec)
                    # compute angle between heading and dominant scene vector (if available)
                    alert = False
                    if dom_vec is not None:
                        ang = angle_between_vectors(heading_vec, dom_vec)
                        if ang > DIRECTION_MATCH_ANGLE_THRESHOLD and math.hypot(*heading_vec) > 2.0:
                            alert = True
                    # overlay
                    color = (0, 255, 0) if not alert else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID {oid} {compass}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    # draw centroid
                    cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)
                    # draw heading arrow
                    ax = int(cx + heading_unit[0]*50)
                    ay = int(cy + heading_unit[1]*50)
                    cv2.arrowedLine(frame, (int(cx), int(cy)), (ax, ay), color, 2, tipLength=0.3)

                    # If alert and not yet alerted recently, log and snapshot
                    if alert and not obj.alerted:
                        obj.alerted = True
                        snap_name = os.path.join(SNAPSHOT_DIR, f"alert_id{oid}_{frame_idx}_{int(time.time())}.jpg")
                        cv2.imwrite(snap_name, frame)
                        ev = {
                            "timestamp": timestamp,
                            "object_id": oid,
                            "compass": compass,
                            "angle_vs_dom": float(angle_between_vectors(heading_vec, dom_vec)) if dom_vec is not None else None,
                            "snapshot": snap_name,
                            "first_seen": obj.first_seen,
                            "last_seen": obj.last_seen,
                        }
                        self.log_events.append(ev)
                        print("[ALERT] Orientation mismatch for ID", oid, "-> snapshot:", snap_name)

                # draw dominant vector on frame
                if dom_vec is not None:
                    center = (int(w * 0.5), int(h * 0.5))
                    dv = unit_vector(dom_vec)
                    edx = int(center[0] + dv[0] * 100)
                    edy = int(center[1] + dv[1] * 100)
                    cv2.arrowedLine(frame, center, (edx, edy), (255, 255, 0), 3, tipLength=0.3)
                    dd_text = f"Scene dominant: {vector_to_compass(dom_vec)}"
                    cv2.putText(frame, dd_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                # show count
                num_planes = len([o for o in tracked.values()])
                cv2.putText(frame, f"Airplanes: {num_planes}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

                # show frame
                cv2.imshow("AirplaneMonitor", frame)
                if self.writer:
                    self.writer.write(frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] Quit requested by user.")
                    break
                if key == ord('s'):
                    snap = os.path.join(SNAPSHOT_DIR, f"manual_snap_{frame_idx}.jpg")
                    cv2.imwrite(snap, frame)
                    print("[INFO] Saved manual snapshot:", snap)

                # optional max frames exit
                if max_frames > 0 and frame_idx >= max_frames:
                    print("[INFO] Reached max_frames limit.")
                    break

        except KeyboardInterrupt:
            print("[INFO] Interrupted by user (KeyboardInterrupt).")
        finally:
            self.close()

# ---- CLI / Running logic ----
def parse_args():
    parser = argparse.ArgumentParser(description="Airplane monitor and direction checker")
    parser.add_argument("--src", type=str, default="0", help="Video source (0 for webcam or path to file)")
    parser.add_argument("--model", type=str, default="yolov5s", help="YOLO model name or path (e.g., yolov5s or path/to/weights.pt)")
    parser.add_argument("--ultralytics", action="store_true", help="Use ultralytics YOLO if available")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--max-frames", type=int, default=0, help="Process N frames then exit (0 = infinite)")
    return parser.parse_args()

def main():
    args = parse_args()
    app = AirplaneMonitorApp(src=args.src, device=args.device, model_name=args.model, use_ultralytics=args.ultralytics)
    print("[INFO] Starting Airplane Monitor (press 'q' to quit, 's' to save snapshot).")
    app.run(max_frames=args.max_frames)

if __name__ == "__main__":
    main()
