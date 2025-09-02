# ARS-GUI: An AI-Powered Image Analysis Toolkit

ARS-GUI is a desktop application built with Python and CustomTkinter. It provides three tools in one UI: (1) Predict ARS ODs, (2) Falsely Classified Image Detector, and (3) Clustering Analysis.

## Features

1. **Predict ARS ODs**
   - Batch-predict OD values via a TensorFlow Lite regression model (`regression_model.tflite`).
   - Shows image name, predicted OD, and path; export to CSV.

2. **Falsely Classified Image Detector**
   - Uses a TFLite classifier (`classifier_model.tflite`) to flag potentially incorrect images.
   - Produces a pie chart of “Correct” vs “False”.

3. **Clustering Analysis**
   - Extracts features (EfficientNetB3), labels groups, and visualizes with PCA + UMAP.

---

## Getting Started

### Prerequisites
- Python 3.9+  
- Git  
- For Docker GUI usage on Windows/macOS, an X server (e.g. VcXsrv on Windows, XQuartz on macOS).

### Installation & Usage

1) **Clone the repository**
```bash
git clone https://github.com/as4401s/ARS-GUI.git
cd ARS-GUI
```

2) **Create a virtual environment**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

3) **Install dependencies**
```bash
pip install -r requirements.txt
```

4) **Run the app**
```bash
python 5.Final_GUI.py
```

---

## Building the Application

You can package the app either as a standalone executable or with Docker.

### 1) Standalone Executable (PyInstaller)

First, install PyInstaller and project deps. The following **one-liner** installs `requirements.txt` and then builds an exe:

**Windows**
```bash
pip install -r requirements.txt && pip install pyinstaller && ^
pyinstaller --onefile --windowed ^
  --add-data "ARS.png;." ^
  --add-data "classifier_model.tflite;." ^
  --add-data "efficientnetb3_notop.h5;." ^
  --add-data "regression_model.tflite;." ^
  --add-data "standard_scaler.pkl;." ^
  5.Final_GUI.py
```

**macOS/Linux** (note the `:` separator instead of `;`)
```bash
pip install -r requirements.txt && pip install pyinstaller && \
pyinstaller --onefile --windowed \
  --add-data "ARS.png:." \
  --add-data "classifier_model.tflite:." \
  --add-data "efficientnetb3_notop.h5:." \
  --add-data "regression_model.tflite:." \
  --add-data "standard_scaler.pkl:." \
  5.Final_GUI.py
```

The final binary will be in the `dist/` folder.

> Tip: The repo also contains a spec file `5.Final_GUI.spec`. If you prefer using it:
> ```bash
> pip install -r requirements.txt && pip install pyinstaller
> pyinstaller 5.Final_GUI.spec
> ```

### 2) Docker

Build:
```bash
docker build -t ars-gui-app .
```

Run (Linux):
```bash
xhost +local:root
docker run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  ars-gui-app
```

Run (Windows/macOS with an X server running):
- Start VcXsrv (Windows) or XQuartz (macOS) and allow network clients.
- Find your host IP and run:
```bash
docker run --rm \
  -e DISPLAY=HOST_IP:0.0 \
  ars-gui-app
```

Or use `docker-compose`:
```bash
docker-compose up
```

---

## Continuous Integration (optional)

A sample `Jenkinsfile` is included as a starting point for CI builds. You’ll need to adapt registry credentials and, if headless testing is required, add a virtual display (e.g., Xvfb).

---

## License

MIT
