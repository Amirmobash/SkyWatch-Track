#!/usr/bin/env python3

import argparse
import asyncio
import base64
import csv
import datetime
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
import warnings
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from functools import lru_cache, wraps

import cv2
import numpy as np

try:
    import torch
    import torchvision
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

try:
    import tensorflow as tf
    HAS_TF = False
except ImportError:
    HAS_TF = False

try:
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    from scipy.signal import savgol_filter
    from scipy.interpolate import splprep, splev
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
    HAS_FILTERPY = True
except ImportError:
    HAS_FILTERPY = False

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    HAS_DEEPSORT = True
except ImportError:
    HAS_DEEPSORT = False

try:
    import pyav
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False

try:
    import aiohttp
    from aiohttp import web
    import socketio
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import sqlite3
    import psycopg2
    HAS_SQL = True
except ImportError:
    HAS_SQL = True

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle
    from matplotlib.animation import FuncAnimation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from skimage import exposure, restoration, filters
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

def timing_decorator(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if hasattr(wrapper, 'timings'):
            wrapper.timings.append(elapsed)
        else:
            wrapper.timings = [elapsed]
        return result
    return wrapper

class PerformanceMonitor:
    def __init__(self):
        self.frame_times = deque(maxlen=100)
        self.detection_times = deque(maxlen=100)
        self.tracking_times = deque(maxlen=100)
        self.processing_times = deque(maxlen=100)

    def add_frame_time(self, dt: float):
        self.frame_times.append(dt)

    def add_detection_time(self, dt: float):
        self.detection_times.append(dt)

    def add_tracking_time(self, dt: float):
        self.tracking_times.append(dt)

    def add_processing_time(self, dt: float):
        self.processing_times.append(dt)

    @property
    def avg_fps(self) -> float:
        if not self.frame_times:
            return 0.0
        return 1.0 / (sum(self.frame_times) / len(self.frame_times))

    @property
    def detection_latency(self) -> float:
        if not self.detection_times:
            return 0.0
        return sum(self.detection_times) / len(self.detection_times)

    def get_stats(self) -> Dict[str, float]:
        return {
            "fps": self.avg_fps,
            "detection_latency_ms": self.detection_latency * 1000,
            "tracking_latency_ms": (sum(self.tracking_times) / len(self.tracking_times) * 1000) if self.tracking_times else 0,
            "processing_latency_ms": (sum(self.processing_times) / len(self.processing_times) * 1000) if self.processing_times else 0
        }

class CircularBuffer:
    def __init__(self, max_size: int = 100):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def push(self, item: Any):
        with self.lock:
            self.buffer.append(item)

    def pop(self) -> Optional[Any]:
        with self.lock:
            if self.buffer:
                return self.buffer.popleft()
        return None

    def peek(self, index: int = -1) -> Optional[Any]:
        with self.lock:
            if self.buffer:
                return self.buffer[index]
        return None

    def clear(self):
        with self.lock:
            self.buffer.clear()

    def __len__(self) -> int:
        with self.lock:
            return len(self.buffer)

class TrackState(Enum):
    NEW = 1
    TRACKING = 2
    LOST = 3
    REMOVED = 4
    CONFIRMED = 5

@dataclass
class TrackMetadata:
    first_seen: datetime.datetime
    last_seen: datetime.datetime
    avg_speed: float = 0.0
    max_speed: float = 0.0
    avg_heading: float = 0.0
    trajectory: List[Tuple[int, int]] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    alerts_triggered: int = 0
    classification: str = "unknown"
    color: Tuple[int, int, int] = (0, 255, 0)

class EnhancedTracker:
    def __init__(self,
                 max_disappeared: int = 30,
                 max_distance: float = 100.0,
                 use_kalman: bool = True,
                 use_deepsort: bool = False,
                 use_optical_flow_refinement: bool = True,
                 trajectory_length: int = 50):

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.use_kalman = use_kalman and HAS_FILTERPY
        self.use_deepsort = use_deepsort and HAS_DEEPSORT
        self.use_flow_refinement = use_optical_flow_refinement

        self.next_id = 1
        self.objects: Dict[int, 'EnhancedTrackedObject'] = {}
        self.metadata: Dict[int, TrackMetadata] = {}

        if self.use_deepsort:
            self.deepsort = DeepSort(max_age=max_disappeared, n_init=3)

    def update(self, detections: List[Dict], frame: np.ndarray) -> Dict[int, 'EnhancedTrackedObject']:
        if self.use_deepsort and HAS_DEEPSORT:
            return self._update_deepsort(detections, frame)
        else:
            return self._update_centroid(detections, frame)

    def _update_deepsort(self, detections: List[Dict], frame: np.ndarray) -> Dict[int, 'EnhancedTrackedObject']:
        deepsort_dets = []
        for det in detections:
            deepsort_dets.append([
                det['xmin'], det['ymin'],
                det['xmax'], det['ymax'],
                det['conf']
            ])

        if deepsort_dets:
            tracks = self.deepsort.update_tracks(deepsort_dets, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                centroid = (
                    int((ltrb[0] + ltrb[2]) / 2),
                    int((ltrb[1] + ltrb[3]) / 2)
                )

                if track_id not in self.objects:
                    self.objects[track_id] = EnhancedTrackedObject(
                        track_id, centroid, ltrb,
                        datetime.datetime.now().isoformat(),
                        use_kalman=self.use_kalman
                    )
                else:
                    self.objects[track_id].update(centroid, ltrb, datetime.datetime.now().isoformat())

        return self.objects

    def _update_centroid(self, detections: List[Dict], frame: np.ndarray) -> Dict[int, 'EnhancedTrackedObject']:
        if not detections:
            for obj in list(self.objects.values()):
                obj.mark_missing()
                if obj.disappeared > self.max_disappeared:
                    self._remove_object(obj.id)
            return self.objects

        input_centroids = np.array([[(d['xmin'] + d['xmax'])//2, (d['ymin'] + d['ymax'])//2] for d in detections])
        input_bboxes = [(d['xmin'], d['ymin'], d['xmax'], d['ymax']) for d in detections]

        if not self.objects:
            for centroid, bbox in zip(input_centroids, input_bboxes):
                self._register_object(tuple(centroid), bbox)
            return self.objects

        obj_ids = list(self.objects.keys())
        obj_centroids = np.array([self.objects[oid].last_centroid() for oid in obj_ids])

        cost_matrix = cdist(obj_centroids, input_centroids)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        assigned_rows = set()
        assigned_cols = set()

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] < self.max_distance:
                assigned_rows.add(r)
                assigned_cols.add(c)
                oid = obj_ids[r]
                self.objects[oid].update(tuple(input_centroids[c]), input_bboxes[c], datetime.datetime.now().isoformat())

        for i, oid in enumerate(obj_ids):
            if i not in assigned_rows:
                self.objects[oid].mark_missing()
                if self.objects[oid].disappeared > self.max_disappeared:
                    self._remove_object(oid)

        for j in range(len(input_centroids)):
            if j not in assigned_cols:
                self._register_object(tuple(input_centroids[j]), input_bboxes[j])

        return self.objects

    def _register_object(self, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> None:
        obj = EnhancedTrackedObject(self.next_id, centroid, bbox, datetime.datetime.now().isoformat(), self.use_kalman)
        self.objects[self.next_id] = obj
        self.metadata[self.next_id] = TrackMetadata(
            first_seen=datetime.datetime.now(),
            last_seen=datetime.datetime.now(),
            trajectory=[centroid]
        )
        self.next_id += 1

    def _remove_object(self, obj_id: int) -> None:
        if obj_id in self.objects:
            del self.objects[obj_id]

class EnhancedTrackedObject:
    def __init__(self, obj_id: int, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int],
                 timestamp: str, use_kalman: bool = True):

        self.id = obj_id
        self.centroids = deque(maxlen=100)
        self.centroids.append(centroid)
        self.bbox = bbox
        self.disappeared = 0
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.alerted = False
        self.alert_cooldown = 0
        self.use_kalman = use_kalman and HAS_FILTERPY
        self.state = TrackState.NEW
        self.confidence = 1.0
        self.predicted_trajectory = []

        if self.use_kalman:
            self.kalman = self._create_kalman_filter()
            self.kalman.predict()
            self.kalman.update(centroid)

    def _create_kalman_filter(self):
        kf = KalmanFilter(dim_x=6, dim_z=2)

        dt = 1.0
        kf.F = np.array([
            [1, 0, dt, 0, 0.5*dt*dt, 0],
            [0, 1, 0, dt, 0, 0.5*dt*dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        kf.R = np.eye(2) * 2.0
        kf.Q = np.eye(6) * 0.1

        kf.P *= 100.0

        return kf

    def update(self, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int], timestamp: str) -> None:
        if self.use_kalman:
            self.kalman.predict()
            self.kalman.update(centroid)
            filtered = (int(self.kalman.x[0]), int(self.kalman.x[1]))
            self.centroids.append(filtered)
            self.velocity = (self.kalman.x[2], self.kalman.x[3])
        else:
            self.centroids.append(centroid)

        self.bbox = bbox
        self.disappeared = 0
        self.last_seen = timestamp

        if self.alert_cooldown > 0:
            self.alert_cooldown -= 1

        if self.state == TrackState.NEW and len(self.centroids) >= 3:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.CONFIRMED:
            self.state = TrackState.TRACKING

    def mark_missing(self) -> None:
        self.disappeared += 1

        if self.use_kalman:
            self.kalman.predict()
            predicted = (int(self.kalman.x[0]), int(self.kalman.x[1]))
            self.centroids.append(predicted)

        if self.disappeared > 5:
            self.state = TrackState.LOST

    def compute_heading_vector(self) -> Tuple[float, float]:
        if len(self.centroids) < 2:
            return (0.0, 0.0)

        recent = list(self.centroids)[-10:]
        if len(recent) >= 2:
            p0 = recent[0]
            p1 = recent[-1]
            return (p1[0] - p0[0], p1[1] - p0[1])

        p0 = self.centroids[0]
        p1 = self.centroids[-1]
        return (p1[0] - p0[0], p1[1] - p0[1])

    def predict_future_position(self, steps: int = 5) -> List[Tuple[int, int]]:
        predictions = []

        if self.use_kalman:
            kf_temp = self.kalman.copy()
            for _ in range(steps):
                kf_temp.predict()
                predictions.append((int(kf_temp.x[0]), int(kf_temp.x[1])))
        else:
            if len(self.centroids) >= 5:
                recent = list(self.centroids)[-5:]
                velocities = []
                for i in range(1, len(recent)):
                    velocities.append((
                        recent[i][0] - recent[i-1][0],
                        recent[i][1] - recent[i-1][1]
                    ))
                avg_vx = sum(v[0] for v in velocities) / len(velocities)
                avg_vy = sum(v[1] for v in velocities) / len(velocities)

                last_pos = self.centroids[-1]
                for i in range(1, steps + 1):
                    predictions.append((
                        int(last_pos[0] + avg_vx * i),
                        int(last_pos[1] + avg_vy * i)
                    ))

        return predictions

    def last_centroid(self) -> Tuple[int, int]:
        if self.use_kalman:
            return (int(self.kalman.x[0]), int(self.kalman.x[1]))
        return tuple(self.centroids[-1])

    def compute_speed(self, fps: float = 30.0) -> float:
        if len(self.centroids) < 2:
            return 0.0

        recent = list(self.centroids)[-10:]
        if len(recent) >= 2:
            total_distance = 0
            for i in range(1, len(recent)):
                dx = recent[i][0] - recent[i-1][0]
                dy = recent[i][1] - recent[i-1][1]
                total_distance += math.hypot(dx, dy)
            return (total_distance / (len(recent) - 1)) * fps

        return 0.0

class DetectionBackend(Enum):
    ULTRALYTICS = "ultralytics"
    TORCH_HUB = "torch_hub"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    CUSTOM = "custom"

class YOLODetectorAdvanced:
    def __init__(self,
                 model_name: str = "yolov8n",
                 backend: Union[str, DetectionBackend] = DetectionBackend.ULTRALYTICS,
                 device: str = "auto",
                 conf_threshold: float = 0.35,
                 iou_threshold: float = 0.45,
                 target_classes: List[Union[str, int]] = None,
                 half_precision: bool = False,
                 max_detections: int = 300,
                 augment: bool = False):

        self.model_name = model_name
        self.backend = DetectionBackend(backend) if isinstance(backend, str) else backend
        self.device = self._auto_select_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or ["airplane"]
        self.half_precision = half_precision and self.device == "cuda"
        self.max_detections = max_detections
        self.augment = augment

        self.model = None
        self.class_names = {}
        self._load_model()

        self.detection_count = 0
        self.total_inference_time = 0.0

    def _auto_select_device(self, device: str) -> str:
        if device != "auto":
            return device

        if HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        elif HAS_TORCH and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_model(self) -> None:
        if self.backend == DetectionBackend.ULTRALYTICS and HAS_ULTRALYTICS:
            self._load_ultralytics()
        elif self.backend == DetectionBackend.TORCH_HUB and HAS_TORCH:
            self._load_torch_hub()
        else:
            raise RuntimeError(f"No suitable detection backend available. Backend: {self.backend}")

    def _load_ultralytics(self) -> None:
        try:
            self.model = YOLO(self.model_name)

            if self.device == "cuda":
                self.model.to("cuda")

            if hasattr(self.model, 'names'):
                self.class_names = self.model.names

        except Exception as e:
            raise RuntimeError(f"Failed to load Ultralytics model: {e}")

    def _load_torch_hub(self) -> None:
        try:
            self.model = torch.hub.load(
                'ultralytics/yolov5',
                self.model_name,
                pretrained=True,
                trust_repo=True
            )

            if self.device != "cpu":
                self.model.to(self.device)

            if self.half_precision:
                self.model.half()

            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold
            self.model.max_det = self.max_detections

            self.class_names = self.model.names

        except Exception as e:
            raise RuntimeError(f"Failed to load torch.hub model: {e}")

    @timing_decorator
    def detect(self, frame: np.ndarray, return_visualization: bool = False) -> Union[List[Dict], Tuple[List[Dict], np.ndarray]]:
        start_time = time.perf_counter()

        if self.backend == DetectionBackend.ULTRALYTICS:
            detections, vis_img = self._detect_ultralytics(frame, return_visualization)
        else:
            detections, vis_img = self._detect_torch_hub(frame, return_visualization)

        self.detection_count += 1
        self.total_inference_time += time.perf_counter() - start_time

        if return_visualization:
            return detections, vis_img
        return detections

    def _detect_ultralytics(self, frame: np.ndarray, return_vis: bool) -> Tuple[List[Dict], Optional[np.ndarray]]:
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            augment=self.augment,
            verbose=False,
            device=self.device
        )

        if isinstance(results, list):
            results = results[0]

        detections = []
        vis_img = frame.copy() if return_vis else None

        if results.boxes is not None:
            boxes = results.boxes
            for box in boxes:
                cls = int(box.cls.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                name = self.class_names.get(cls, str(cls))

                if self._is_target(name, cls):
                    detection = {
                        "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
                        "conf": conf, "class": cls, "name": name,
                        "area": (x2 - x1) * (y2 - y1)
                    }
                    detections.append(detection)

                    if return_vis and vis_img is not None:
                        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{name}: {conf:.2f}"
                        cv2.putText(vis_img, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return detections, vis_img

    def _detect_torch_hub(self, frame: np.ndarray, return_vis: bool) -> Tuple[List[Dict], Optional[np.ndarray]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            results = self.model(rgb)

        detections = []
        vis_img = frame.copy() if return_vis else None

        if hasattr(results, 'xyxy'):
            xyxy = results.xyxy[0].cpu().numpy()

            for row in xyxy:
                x1, y1, x2, y2, conf, cls = row
                cls = int(cls)

                if conf >= self.conf_threshold and self._is_target(self.class_names.get(cls, str(cls)), cls):
                    detection = {
                        "xmin": int(x1), "ymin": int(y1), "xmax": int(x2), "ymax": int(y2),
                        "conf": float(conf), "class": cls, "name": self.class_names.get(cls, str(cls)),
                        "area": (int(x2) - int(x1)) * (int(y2) - int(y1))
                    }
                    detections.append(detection)

                    if return_vis and vis_img is not None:
                        cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f"{detection['name']}: {detection['conf']:.2f}"
                        cv2.putText(vis_img, label, (int(x1), int(y1) - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return detections, vis_img

    def _is_target(self, name: str, class_id: int) -> bool:
        if not self.target_classes:
            return True

        for target in self.target_classes:
            if isinstance(target, int):
                if class_id == target:
                    return True
            else:
                if target.lower() in name.lower():
                    return True
        return False

    def get_performance_stats(self) -> Dict[str, float]:
        avg_time = self.total_inference_time / max(1, self.detection_count)
        return {
            "avg_inference_time_ms": avg_time * 1000,
            "total_detections": self.detection_count,
            "fps_capability": 1.0 / avg_time if avg_time > 0 else 0
        }

class EnhancedOpticalFlow:
    def __init__(self,
                 method: str = "farneback",
                 smoothing_frames: int = 30,
                 skip_frames: int = 1,
                 pyramid_scale: float = 0.5,
                 levels: int = 3,
                 winsize: int = 15,
                 iterations: int = 3,
                 use_gpu: bool = False):

        self.method = method
        self.smoothing_frames = smoothing_frames
        self.skip_frames = skip_frames
        self.frame_counter = 0
        self.prev_gray = None
        self.prev_points = None
        self.history = deque(maxlen=smoothing_frames)

        self.farneback_params = {
            "pyr_scale": pyramid_scale,
            "levels": levels,
            "winsize": winsize,
            "iterations": iterations,
            "poly_n": 5,
            "poly_sigma": 1.2,
            "flags": 0
        }

        self.lk_params = {
            "winSize": (winsize, winsize),
            "maxLevel": levels,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }

        self.use_gpu = use_gpu and HAS_TORCH and torch.cuda.is_available()
        self.flow_vectors = None
        self.magnitude = None
        self.angle = None

    def compute_flow(self, frame: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        self.frame_counter += 1
        if self.frame_counter % self.skip_frames != 0:
            return self.flow_vectors

        if self.method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, **self.farneback_params
            )
        elif self.method == "lucas_kanade":
            flow = self._compute_lk_flow(self.prev_gray, gray)
        elif self.method == "rlof":
            flow = self._compute_rlof_flow(self.prev_gray, gray)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, **self.farneback_params
            )

        self.flow_vectors = flow
        self.magnitude, self.angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        self.prev_gray = gray
        return flow

    def _compute_lk_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        if self.prev_points is None:
            self.prev_points = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7
            )

        if self.prev_points is not None and len(self.prev_points) > 0:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, self.prev_points, None, **self.lk_params
            )

            flow = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)

            for i, (new, old, st) in enumerate(zip(next_points, self.prev_points, status)):
                if st[0] == 1:
                    x_old, y_old = old.ravel()
                    x_new, y_new = new.ravel()

                    if 0 <= x_old < flow.shape[1] and 0 <= y_old < flow.shape[0]:
                        flow[int(y_old), int(x_old)] = [x_new - x_old, y_new - y_old]

            self.prev_points = next_points[status == 1]
            return flow

        return np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)

    def _compute_rlof_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        try:
            rlof = cv2.optflow.createOptFlow_RLOF()
            flow = rlof.calc(prev_gray, curr_gray, None)
            return flow
        except:
            return cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **self.farneback_params)

def main():
    print("Airplane Monitor & Direction Checker - Ultimate Professional Edition")
    print("This script requires proper configuration and input source.")
    print("Example usage: python script.py --source video.mp4 --output output.mp4")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam or path)")
    parser.add_argument("--output", type=str, help="Output video file")
    parser.add_argument("--model", type=str, default="yolov8n", help="YOLO model name")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--display", action="store_true", help="Display output window")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    detector = YOLODetectorAdvanced(
        model_name=args.model,
        conf_threshold=args.conf,
        target_classes=["airplane"]
    )

    tracker = EnhancedTracker(use_kalman=True)
    flow = EnhancedOpticalFlow(method="farneback")

    out = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        flow.compute_flow(frame)

        detections, vis = detector.detect(frame, return_visualization=True)

        tracker.update(detections, frame)

        for obj_id, obj in tracker.objects.items():
            centroid = obj.last_centroid()
            cv2.circle(vis, centroid, 5, (0, 0, 255), -1)
            cv2.putText(vis, f"ID:{obj_id}", (centroid[0]+5, centroid[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        if args.display:
            cv2.imshow("Airplane Monitor", vis)

        if out:
            out.write(vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
