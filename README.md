# SkyWatch-Track

A webcam-based airplane monitoring system that detects, counts, and tracks airplanes in a live video feed,
computes their heading, compares it to the dominant scene direction (runway / flow), and raises alerts when
orientation mismatches are detected.

**This repository was prepared for:** Amir Mobasheraghdam (metadata obfuscated in code).

## Contents

- `airplane_monitor.py` - Main Python script to run the monitor.
- `requirements.txt` - Python packages required.
- `README.md` - This file.
- `.gitignore` - Common ignores.
- `LICENSE` - MIT license (suggested).
- `assets/` - Generated images and engineering diagram.
- `prompt.txt` - The image-generation prompt used to create the schematic.

## Quickstart

Create a virtual environment, install dependencies and run:

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python airplane_monitor.py --src 0 --model yolov5s
```

For using `--ultralytics` you must install the `ultralytics` package and optionally GPU support.

## Notes

- The author/site metadata is base64-encoded inside `airplane_monitor.py` (see `hidden_metadata()`).
- Do not store real secrets in source files.
