# Use a base image with Python and necessary GUI libraries installed
# This image includes Python 3.9 and libraries for Tkinter to function.
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV and Tkinter GUI
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your application files into the container's working directory
# This includes all Python scripts, models (.tflite, .h5, .pkl), and image assets (.jpg)
COPY . .

# Set the entrypoint to run the main GUI script
# This command will be executed when the container starts
CMD ["python", "5.Final_GUI.py"]