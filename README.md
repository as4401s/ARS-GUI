# ARS-GUI: An AI-Powered Image Analysis Toolkit

ARS-GUI is a comprehensive desktop application built with Python and CustomTkinter, designed to provide a suite of tools for analyzing images using machine learning models. The application integrates three distinct functionalities into a single, user-friendly interface with a sidebar for easy navigation.

## Features

The application is organized into three main modules:

1.  **Predict ARS ODs**:
    * Load a batch of images to predict their Optical Density (OD) values using a pre-trained regression model (`regression_model.tflite`).
    * Displays the results in a clear, tabular format showing the image name, predicted OD, and file path.
    * Provides functionality to export the prediction results to a CSV file.

2.  **Falsely Classified Image Detector**:
    * Uses a classification model (`classifier_model.tflite`) to identify images that may be "falsely classified" (e.g., images that do not fit the expected category).
    * Processes a directory of images and displays the predicted label for each one.
    * Generates a pie chart to visualize the proportion of "Correct" vs. "False" predictions.

3.  **Clustering Analysis**:
    * Extracts deep features from images using a pre-trained `EfficientNetB3` model.
    * Allows users to assign labels to different groups of images.
    * Visualizes the high-dimensional image features in 2D space using both PCA (Principal Component Analysis) and UMAP to show how images cluster together based on their content.

---

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* **Python**: Version 3.9 or newer.
* **Git**: To clone the repository.
* An **X Server** is required for running the GUI inside a Docker container on Windows or macOS (e.g., [VcXsrv](https://sourceforge.net/projects/vcxsrv/) for Windows, [XQuartz](https://www.xquartz.org/) for macOS).

---

## Installation & Usage

Follow these steps to set up your local environment.

1.  **Clone the repository:**
    Open your terminal and clone the GitLab repository to your local machine.
    ```bash
    git clone [https://asb-git.hki-jena.de/applied-systems-biology/ars-gui.git](https://asb-git.hki-jena.de/applied-systems-biology/ars-gui.git)
    cd ars-gui
    ```

2.  **Create a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    # For Windows
    python -m venv .venv
    .venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    Install all the required Python packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application:**
    Once the dependencies are installed, you can start the GUI by running the main script.
    ```bash
    python 5.Final_GUI.py
    ```

---

## Building the Application

You can package the application in two ways: as a standalone executable or as a Docker container.

### 1. Creating a Standalone Executable (.exe)

To create a single `.exe` file that can be run on any Windows machine without needing Python installed, use **PyInstaller**.

1.  **Install PyInstaller:**
    ```bash
    pip install pyinstaller
    ```

2.  **Build the Executable:**
    Run the following command from the root of your project directory. This command bundles your main script with all necessary data files (models, images, etc.) into one file.
    ```bash
    pyinstaller --onefile --windowed --add-data "ARS.jpg;." --add-data "classifier_model.tflite;." --add-data "efficientnetb3_notop.h5;." --add-data "regression_model.tflite;." --add-data "standard_scaler.pkl;." 5.Final_GUI.py
    ```
    * `--onefile`: Creates a single executable file.
    * `--windowed`: Prevents a console window from appearing when the GUI is run.
    * `--add-data`: Bundles necessary assets. The format is `"source;destination"`.

    The final `.exe` file will be located in the `dist` folder.

### 2. Building with Docker

Docker allows you to package the application and its environment into a portable container.

1.  **Build the Docker Image:**
    Make sure you have Docker Desktop running. In your terminal, run:
    ```bash
    docker build -t ars-gui-app .
    ```

2.  **Run the Docker Container:**
    To run the GUI from the container, you need to connect it to your host's display.

    * **On Linux:**
        ```bash
        docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix ars-gui-app
        ```
    * **On Windows/macOS (with an X Server running):**
        You first need to find your host's IP address and allow connections to the X server. Then run:
        ```bash
        # Replace YOUR_IP_ADDRESS with your actual IP
        docker run --rm -e DISPLAY=YOUR_IP_ADDRESS:0.0 ars-gui-app
        ```
    You can also use the provided `docker-compose.yml` file for easier management:
    ```bash
    docker-compose up
    ```

---

## Continuous Integration with Jenkins

The `Jenkinsfile` in this repository defines a basic Continuous Integration (CI) pipeline to automate the build process.

### Pipeline Stages

1.  **Checkout**: Clones the source code from the GitLab repository.
2.  **Build Docker Image**: Builds a Docker image using the `Dockerfile`. The image is tagged with the build number (e.g., `hki-jena/ars-gui:build-12`).
3.  **Run/Test (Placeholder)**: This stage is a placeholder for running automated tests. Since this is a GUI application, running it in a typical CI environment is non-trivial and would require a virtual display server like Xvfb.
4.  **Cleanup**: After the pipeline runs, the Jenkins workspace is cleaned up.

To use this pipeline, you would need to set up a "Pipeline" job in Jenkins and point it to the `Jenkinsfile` in your repository. You would also need to configure Jenkins with credentials to access the private GitLab repository and potentially a Docker registry if you choose to push the built images.