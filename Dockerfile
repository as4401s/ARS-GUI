# Use slim base; Tkinter, OpenCV and UMAP need extra system libs and build tools
FROM python:3.10-slim

# Avoid interactive tzdata prompts etc.
ENV DEBIAN_FRONTEND=noninteractive     PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

# System deps:
# - tk/tcl (Tkinter GUI), X11 libs for display, OpenGL bits for cv2
# - build tools for numpy/scipy/umap-learn (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends     python3-tk tk     libx11-6 libxext6 libxrender1 libxft2 libxtst6 libxau6 libxdmcp6     libxcb1 libxdamage1 libxfixes3 libxi6 libxrandr2     libgl1 libglib2.0-0     gcc g++ make     && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy only requirements first for layer caching
COPY requirements.txt /app/requirements.txt

# Install python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code and models
COPY 1.predict_od.py 2.falsely_classifier.py 3.clustering.py 5.Final_GUI.py /app/
COPY ARS.png classifier_model.tflite efficientnetb3_notop.h5 regression_model.tflite standard_scaler.pkl /app/

# Set default display; will be overridden at runtime via ENV
ENV DISPLAY=:0

# Run the GUI (user can override the CMD in docker run)
CMD ["python", "5.Final_GUI.py"]
