# SkyWatch-Track

**A real‑time airplane monitoring system**  
Detect, track, and alert on aircraft heading anomalies using your webcam or video files.  
Built with YOLO + optical flow – ideal for runway monitoring, drone detection, or general aviation observation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![YOLOv5](https://img.shields.io/badge/YOLO-v5/v8-red)](https://github.com/ultralytics/yolov5)

---

## ✈️ Overview

SkyWatch‑Track captures live video from a webcam or a file, detects airplanes (or any COCO object), tracks them across frames, and computes their heading direction. It simultaneously estimates the **dominant scene motion** using dense optical flow (e.g., the movement of clouds, background, or a runway). When an airplane’s heading deviates significantly from the scene flow, the system raises an alert and saves a snapshot.

**Perfect for:**
- Runway approach monitoring
- Drone / UAV tracking
- General aviation observation
- Teaching computer vision & object tracking

---

## 📦 Features

- **Real‑time object detection** – YOLOv5 / YOLOv8 (Ultralytics or torch.hub)
- **Robust centroid tracking** – with Hungarian algorithm (greedy fallback) and optional Kalman filtering
- **Dominant scene direction** – dense Farneback optical flow with temporal smoothing
- **Heading alerts** – configurable angle threshold, per‑object cooldown
- **Logging & snapshots** – CSV/JSON event logs and automatic snapshots on alert
- **Video output** – save annotated video to disk
- **Fully configurable** – command line or JSON config file
- **Author metadata** – base64‑encoded author info inside the script (see below)

---

## 📋 Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- PyTorch (for YOLO)
- NumPy, SciPy (optional, for Hungarian matching)
- Ultralytics (optional, for YOLOv8)
- FilterPy (optional, for Kalman tracking)

All dependencies are listed in `requirements.txt`.

---

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/SkyWatch-Track.git
   cd SkyWatch-Track
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/macOS
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** For GPU acceleration with Ultralytics YOLO, install `torch` with CUDA support separately.

4. **Run the monitor**
   ```bash
   python airplane_monitor.py --source 0
   ```

---

## 🕹️ Usage

```bash
python airplane_monitor.py [options]
```

### Basic examples

| Command | Description |
|---------|-------------|
| `python airplane_monitor.py --source 0` | Use default webcam (device 0) |
| `python airplane_monitor.py --source video.mp4` | Process a video file |
| `python airplane_monitor.py --ultralytics --model yolov8n.pt` | Use Ultralytics YOLOv8 nano |
| `python airplane_monitor.py --save-video --output result.avi` | Save annotated video |
| `python airplane_monitor.py --config settings.json` | Load configuration from JSON file |

### Key interactive commands

- Press **`q`** – quit the application
- Press **`s`** – save a manual snapshot (to `snapshots/` folder)

---

## ⚙️ Configuration

All settings can be passed as command‑line arguments or via a JSON config file.

### Important options

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `0` | Video source (camera index or file path) |
| `--model` | `yolov5s` | YOLO model name (e.g., `yolov5s`, `yolov8n.pt`) |
| `--ultralytics` | `False` | Use Ultralytics YOLO (instead of torch.hub) |
| `--confidence` | `0.35` | Detection confidence threshold |
| `--classes` | `airplane` | Target class(es) – e.g., `airplane drone` |
| `--kalman` | `False` | Enable Kalman filtering per track (requires `filterpy`) |
| `--angle-threshold` | `45.0` | Max allowed angle deviation (degrees) |
| `--max-disappeared` | `20` | Frames before losing a track |
| `--snapshot-dir` | `snapshots` | Folder for alert / manual snapshots |
| `--save-video` | `False` | Write output video to disk |
| `--config` | `None` | JSON configuration file (overrides CLI) |

For a complete list, run:
```bash
python airplane_monitor.py --help
```

### Example JSON config (`settings.json`)
```json
{
  "source": "0",
  "model": "yolov5s",
  "confidence": 0.4,
  "kalman": true,
  "angle_threshold": 30.0,
  "save_video": true,
  "output_video": "runway_monitor.avi"
}
```

---

## 🧠 How It Works

1. **Object Detection** – Each frame is passed to YOLO (v5 or v8). Only the configured classes (e.g., `airplane`) are kept.
2. **Centroid Tracking** – Detections are matched to existing tracks using Euclidean distance (Hungarian algorithm if SciPy available). Tracks that disappear for too long are removed.
3. **Heading Computation** – For each tracked object, a heading vector is derived from the line connecting its first and last stored centroid (rolling window of up to 30 positions).
4. **Scene Flow** – Dense optical flow (Farneback) is computed between consecutive frames. The average flow vector over a sliding window gives the dominant scene direction.
5. **Alert Logic** – The angle between the object’s heading and the scene flow is calculated. If it exceeds `--angle-threshold` and the object is moving fast enough, an alert is triggered. A cooldown prevents repeated alerts for the same object.
6. **Output** – Annotated frames are shown live (and optionally saved). Alert snapshots and structured logs (CSV/JSON) are stored.

![SkyWatch-Track Schematic](assets/schematic.png)  
*Conceptual diagram – see `prompt.txt` for the prompt used to generate this image.*

---

## 📁 Outputs

| Output | Location | Description |
|--------|----------|-------------|
| **Alert snapshots** | `snapshots/alert_id*_frame*.jpg` | JPEG images captured when an anomaly is detected |
| **Manual snapshots** | `snapshots/manual_*.jpg` | User‑requested snapshots (press `s`) |
| **Event logs (CSV)** | `airplane_events.csv` | Tabular log of each alert (timestamp, object ID, angle, snapshot path) |
| **Event logs (JSON)** | `airplane_events.json` | Same data in JSON format |
| **Output video** | `output.avi` (if `--save-video`) | Full annotated video with bounding boxes and arrows |

---

## 👨‍💻 Author & Metadata

**Author:** Amir Mobasheraghdam  


The source code contains base64‑encoded author metadata (see function `_hidden_metadata()` in `airplane_monitor.py`). This is purely for attribution and does not affect functionality.

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Ultralytics for YOLOv5/YOLOv8
- OpenCV community for optical flow and tracking utilities
- FilterPy library for Kalman filtering (optional)

---

**Happy monitoring!** ✈️🛩️
