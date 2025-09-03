import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk, ImageOps
import tensorflow as tf
import numpy as np
import os, sys, time
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import preprocess_input, EfficientNetB3
from sklearn.decomposition import PCA
import umap
import joblib

# Set the dark appearance mode
ctk.set_appearance_mode("Dark")

# ------------------------------------------------------------------------------
# Helper function for resource paths (for PyInstaller compatibility)
# ------------------------------------------------------------------------------
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ==============================================================================
# 1. Predict ARS ODs Tool (Integrated from predict_od.py)
# ==============================================================================
class PredictODFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        # Variables for OD prediction
        self.od_zero_values = []
        self.od_zero_images_list = []  # now will store up to 4 images
        self.ars_images_list = []       # now will store up to 4 images
        self.results = []
        self.stop_process = False

        # Load TFLite model and scaler for OD prediction
        tflite_model_path = resource_path("regression_model.tflite")
        scaler_path = resource_path("standard_scaler.pkl")
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.scaler = joblib.load(scaler_path)

        # -------------------------- Layout Setup --------------------------
        self.grid_columnconfigure(0, weight=1)

        # Title Label
        title_label = ctk.CTkLabel(self, text="ARS OD Prediction Tool", font=("Helvetica", 24, "bold"))
        title_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # ---- OD Zero Section ----
        od_zero_frame = ctk.CTkFrame(self, corner_radius=15)
        od_zero_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        for i in range(4):
            od_zero_frame.grid_columnconfigure(i, weight=0)
        od_zero_frame.grid_columnconfigure(3, weight=1)

        od_zero_label = ctk.CTkLabel(od_zero_frame, text="Step 1: Upload OD Zero Images", font=("Helvetica", 18, "bold"))
        od_zero_label.grid(row=0, column=0, padx=20, pady=10, sticky="w")

        od_zero_button = ctk.CTkButton(od_zero_frame, text="Upload OD Zero Images", command=self.upload_od_zero)
        od_zero_button.grid(row=0, column=1, padx=20, pady=10, sticky="w")

        od_zero_spacer = ctk.CTkLabel(od_zero_frame, text="", width=150)
        od_zero_spacer.grid(row=0, column=2, padx=20, pady=10, sticky="w")

        self.od_zero_images_frame = ctk.CTkFrame(od_zero_frame, width=600, height=150)
        self.od_zero_images_frame.grid(row=0, column=3, padx=20, pady=10)
        self.od_zero_images_frame.grid_propagate(False)

        # OD Zero progress bar
        self.od_zero_progress_bar = ctk.CTkProgressBar(od_zero_frame, width=600)
        self.od_zero_progress_bar.grid(row=1, column=0, columnspan=4, sticky="ew", padx=20, pady=(0, 10))
        self.od_zero_progress_bar.set(0)

        # ---- ARS Section ----
        ars_frame = ctk.CTkFrame(self, corner_radius=15)
        ars_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=10)
        for i in range(4):
            ars_frame.grid_columnconfigure(i, weight=0)
        ars_frame.grid_columnconfigure(3, weight=1)

        ars_label = ctk.CTkLabel(ars_frame, text="Step 2: Upload ARS Images", font=("Helvetica", 18, "bold"))
        ars_label.grid(row=0, column=0, padx=20, pady=10, sticky="w")

        ars_button = ctk.CTkButton(ars_frame, text="Upload ARS Images", command=self.predict_images)
        ars_button.grid(row=0, column=1, padx=20, pady=10, sticky="w")

        stop_button = ctk.CTkButton(ars_frame, text="Stop Prediction", command=self.stop_prediction)
        stop_button.grid(row=0, column=2, padx=20, pady=10, sticky="w")

        self.ars_images_frame = ctk.CTkFrame(ars_frame, width=600, height=150)
        self.ars_images_frame.grid(row=0, column=3, padx=20, pady=10)
        self.ars_images_frame.grid_propagate(False)

        # Main Progress Bar for ARS predictions
        self.progress_label = ctk.CTkLabel(self, text="Progress: 0/0 images processed", font=("Helvetica", 18))
        self.progress_label.grid(row=3, column=0, sticky="ew", padx=20, pady=(10, 5))

        self.progress_bar = ctk.CTkProgressBar(self, width=600)
        self.progress_bar.grid(row=4, column=0, sticky="ew", padx=20, pady=(20, 20))
        self.progress_bar.set(0)

        # Results Table
        table_frame = ctk.CTkFrame(self)
        table_frame.grid(row=5, column=0, sticky="nsew", padx=10, pady=10)
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)
        self.table = ttk.Treeview(table_frame, columns=("Image Name", "Predicted OD", "Image Path"), show="headings", height=10)
        for col in ("Image Name", "Predicted OD", "Image Path"):
            self.table.heading(col, text=col)
            self.table.column(col, width=200, anchor="center", stretch=True)
        v_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=v_scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        v_scroll.pack(side="right", fill="y")

        # Control Buttons
        buttons_frame = ctk.CTkFrame(self)
        buttons_frame.grid(row=6, column=0, sticky="ew", padx=10, pady=10)
        clear_button = ctk.CTkButton(buttons_frame, text="Clear Predictions", command=self.clear_predictions)
        clear_button.pack(side="left", padx=10)
        plot_button = ctk.CTkButton(buttons_frame, text="Plot Histogram", command=self.plot_histogram)
        plot_button.pack(side="left", padx=10)
        export_button = ctk.CTkButton(buttons_frame, text="Export as Excel File", command=self.export_to_excel)
        export_button.pack(side="left", padx=10)

    # ----- Helper Methods -----
    def preprocess_image(self, image_path, target_size=(512, 512)):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0).astype(np.float32)

    def resize_image_for_gui(self, image_path, display_size=(128, 128)):
        img = Image.open(image_path)
        img.thumbnail(display_size)
        img = ImageOps.expand(img, border=5, fill="black")
        return ctk.CTkImage(light_image=img, size=display_size)

    def update_od_zero_images_display(self):
        for widget in self.od_zero_images_frame.winfo_children():
            widget.destroy()
        spacer = ctk.CTkLabel(self.od_zero_images_frame, text="")
        spacer.pack(side="left", fill="x", expand=True)
        for img in self.od_zero_images_list:
            lbl = ctk.CTkLabel(self.od_zero_images_frame, image=img, text="")
            lbl.image = img  # keep reference
            lbl.pack(side="left", padx=5)

    def update_ars_images_display(self):
        for widget in self.ars_images_frame.winfo_children():
            widget.destroy()
        spacer = ctk.CTkLabel(self.ars_images_frame, text="")
        spacer.pack(side="left", fill="x", expand=True)
        for img in self.ars_images_list:
            lbl = ctk.CTkLabel(self.ars_images_frame, image=img, text="")
            lbl.image = img
            lbl.pack(side="left", padx=5)

    # ----- Button Callback Methods -----
    def upload_od_zero(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_paths:
            return
        total_images = len(file_paths)
        self.od_zero_progress_bar.set(0)
        for index, file_path in enumerate(file_paths):
            try:
                image_data = self.preprocess_image(file_path)
                self.interpreter.set_tensor(self.input_details[0]["index"], image_data)
                self.interpreter.invoke()
                pred = self.interpreter.get_tensor(self.output_details[0]["index"])
                od_zero_value = self.scaler.inverse_transform([[pred[0][0]]])[0][0]
                self.od_zero_values.append(od_zero_value)
                img_ctk = self.resize_image_for_gui(file_path)
                self.od_zero_images_list.append(img_ctk)
                if len(self.od_zero_images_list) > 4:
                    self.od_zero_images_list.pop(0)
                self.update_od_zero_images_display()
                progress = (index + 1) / total_images
                self.od_zero_progress_bar.set(progress)
                self.update_idletasks()
            except Exception as e:
                messagebox.showerror("Error", f"Error processing OD Zero image: {e}")

    def predict_images(self):
        self.stop_process = False
        if self.od_zero_values:
            od_zero_threshold = np.mean(self.od_zero_values)
        else:
            od_zero_threshold = 0
            messagebox.showwarning("Warning", "No OD Zero image uploaded. Using threshold = 0.")
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_paths:
            return
        total_images = len(file_paths)
        self.progress_label.configure(text=f"Progress: 0/{total_images} images processed")
        self.progress_bar.set(0)
        for index, file_path in enumerate(file_paths):
            if self.stop_process:
                break
            try:
                image_data = self.preprocess_image(file_path)
                self.interpreter.set_tensor(self.input_details[0]["index"], image_data)
                self.interpreter.invoke()
                pred = self.interpreter.get_tensor(self.output_details[0]["index"])
                predicted_od = self.scaler.inverse_transform([[pred[0][0]]])[0][0] - od_zero_threshold
                self.results.append({
                    "Image Name": os.path.basename(file_path),
                    "Predicted OD": predicted_od,
                    "Image Path": file_path,
                })
                self.table.insert("", "end", values=(os.path.basename(file_path), f"{predicted_od:.4f}", file_path))
                img_ctk = self.resize_image_for_gui(file_path)
                self.ars_images_list.append(img_ctk)
                if len(self.ars_images_list) > 4:
                    self.ars_images_list.pop(0)
                self.update_ars_images_display()
                progress = (index + 1) / total_images
                self.progress_bar.set(progress)
                self.progress_label.configure(text=f"Progress: {index + 1}/{total_images} images processed")
                self.update_idletasks()
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    def export_to_excel(self):
        if not self.results:
            messagebox.showwarning("Warning", "No results to export.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel files", "*.xlsx")],
                                                 title="Save Results as Excel File")
        if not file_path:
            return
        try:
            df = pd.DataFrame(self.results)
            df.to_excel(file_path, index=False)
            messagebox.showinfo("Success", f"Results saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {e}")

    def stop_prediction(self):
        self.stop_process = True

    def clear_predictions(self):
        for item in self.table.get_children():
            self.table.delete(item)
        self.results = []
        self.ars_images_list = []
        self.update_ars_images_display()
        self.progress_bar.set(0)
        self.progress_label.configure(text="Progress: 0/0 images processed")

    def plot_histogram(self):
        if not self.results:
            messagebox.showwarning("Warning", "No results to plot.")
            return
        od_values = [result["Predicted OD"] for result in self.results]
        plt.figure(figsize=(8, 5))
        plt.hist(od_values, bins=10, edgecolor='black', alpha=0.7)
        plt.xlabel("Predicted OD Values")
        plt.ylabel("Count")
        plt.title("Distribution of Predicted Optical Densities")
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png")],
                                                 title="Save Histogram As")
        if file_path:
            plt.savefig(file_path, dpi=400)
            messagebox.showinfo("Success", f"Histogram saved to {file_path}")
        plt.close()

# ==============================================================================
# 2. Falsely Triggered Classifier (Integrated from falsely_classifier.py)
# ==============================================================================
class FalselyClassifierFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.results = []
        self.last_two_images = []  # will now store up to 4 images
        self.correct_count = 0
        self.false_count = 0

        # Load the TFLite classifier model
        tflite_model_path = resource_path("classifier_model.tflite")
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # -------------------------- Layout Setup --------------------------
        title_label = ctk.CTkLabel(self, text="Falsely Triggered Image Classification Tool", font=("Helvetica", 24, "bold"))
        title_label.pack(pady=20)

        # Image Upload Frame
        image_frame = ctk.CTkFrame(self, corner_radius=15)
        image_frame.pack(pady=20, padx=20, fill="x")

        image_label = ctk.CTkLabel(image_frame, text="Upload Images for Classification", font=("Helvetica", 18, "bold"))
        image_label.pack(side="left", padx=20)

        upload_button = ctk.CTkButton(image_frame, text="Upload Images", command=self.predict_images)
        upload_button.pack(side="left", padx=20, pady=20)

        # Create 4 image labels instead of 2
        self.image_labels = [ctk.CTkLabel(image_frame, text="") for _ in range(4)]
        for label in self.image_labels:
            label.pack(side="left", padx=10)

        # Progress Bar
        progress_frame = ctk.CTkFrame(self, corner_radius=15)
        progress_frame.pack(pady=20, fill="x")

        self.progress_label = ctk.CTkLabel(progress_frame, text="Progress: 0/0 images processed", font=("Helvetica", 18))
        self.progress_label.pack(pady=10)

        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=600)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)

        # Results Table
        results_frame = ctk.CTkFrame(self, corner_radius=15)
        results_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.table = ttk.Treeview(results_frame, columns=("Image Name", "Predicted Label"), show="headings", height=15)
        self.table.heading("Image Name", text="Image Name")
        self.table.heading("Predicted Label", text="Predicted Label")
        self.table.column("Image Name", width=300, anchor="center")
        self.table.column("Predicted Label", width=250, anchor="center")
        self.table.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        # Buttons for exporting results
        button_frame = ctk.CTkFrame(self, corner_radius=15)
        button_frame.pack(pady=20, fill="x")

        plot_button = ctk.CTkButton(button_frame, text="Plot Results", command=self.plot_results)
        plot_button.pack(side="left", padx=20, pady=10)

        export_button = ctk.CTkButton(button_frame, text="Export as Excel File", command=self.export_excel)
        export_button.pack(side="left", padx=20, pady=10)

        clear_button = ctk.CTkButton(button_frame, text="Clear Predictions", command=self.clear_predictions)
        clear_button.pack(side="left", padx=20, pady=10)

    # ----- Helper Method -----
    def preprocess_image(self, image_path, target_size=512):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (target_size, target_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = preprocess_input(img)
        return np.expand_dims(img, axis=0)

    # ----- Button Callback Methods -----
    def predict_images(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_paths:
            return
        total_images = len(file_paths)
        self.progress_label.configure(text=f"Progress: 0/{total_images} images processed")
        self.progress_bar.set(0)
        for index, file_path in enumerate(file_paths):
            try:
                image_data = self.preprocess_image(file_path)
                self.interpreter.set_tensor(self.input_details[0]['index'], image_data)
                self.interpreter.invoke()
                pred = self.interpreter.get_tensor(self.output_details[0]['index'])
                predicted_label = np.argmax(pred, axis=1)[0]
                label_text = "Correctly Triggered" if predicted_label == 0 else "Falsely Triggered"
                if predicted_label == 0:
                    self.correct_count += 1
                else:
                    self.false_count += 1
                self.results.append({"Image Name": os.path.basename(file_path), "Predicted Label": label_text})
                self.table.insert("", "end", values=(os.path.basename(file_path), label_text))

                img = Image.open(file_path)
                img.thumbnail((200, 200))
                img = ImageOps.expand(img, border=10, fill="black")
                img_tk = ImageTk.PhotoImage(img)
                self.last_two_images.append(img_tk)
                if len(self.last_two_images) > 4:
                    self.last_two_images.pop(0)
                for i, label in enumerate(self.image_labels):
                    if i < len(self.last_two_images):
                        label.configure(image=self.last_two_images[i])
                        label.image = self.last_two_images[i]
                    else:
                        label.configure(image=None)
                        label.image = None
                progress = (index + 1) / total_images
                self.progress_bar.set(progress)
                self.progress_label.configure(text=f"Progress: {index + 1}/{total_images} images processed")
                self.update_idletasks()
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    def plot_results(self):
        if not self.results:
            messagebox.showerror("No Data", "Please run predictions first.")
            return
        correct_count = sum(1 for row in self.results if row["Predicted Label"] == "Correctly Triggered")
        false_count = sum(1 for row in self.results if row["Predicted Label"] == "Falsely Triggered")
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG Image", "*.png")])
        if not file_path:
            return
        plt.figure(figsize=(6, 4))
        plt.bar(["Correctly Triggered", "Falsely Triggered"], [correct_count, false_count],
                color=["green", "red"])
        plt.title("Prediction Summary")
        plt.ylabel("Number of Images")
        plt.savefig(file_path, dpi=400)
        plt.close()
        messagebox.showinfo("Saved", f"Results saved as {file_path}")

    def export_excel(self):
        if not self.results:
            messagebox.showerror("No Data", "No results to export.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel File", "*.xlsx")])
        if not file_path:
            return
        df = pd.DataFrame(self.results)
        df.to_excel(file_path, index=False)
        messagebox.showinfo("Excel Saved", f"Results saved to {file_path}")

    def clear_predictions(self):
        self.results.clear()
        self.correct_count = 0
        self.false_count = 0
        for item in self.table.get_children():
            self.table.delete(item)
        messagebox.showinfo("Cleared", "All predictions have been removed.")

# ==============================================================================
# 3. ARS Image Clustering Tool (Integrated from clustering.py)
# ==============================================================================
class ClusteringFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.image_data = []     # list of image file paths
        self.features_list = []  # list of feature vectors
        self.labels_list = []    # user-provided labels for each image
        self.pca_result = None
        self.pca_labels = None
        self.umap_result = None
        self.umap_labels = None

        # Load EfficientNetB3 model (without top layers) from local weights
        weights_path = resource_path("efficientnetb3_notop.h5")
        self.model = EfficientNetB3(weights=weights_path, include_top=False, pooling="avg")

        # -------------------------- Layout Setup --------------------------
        title_label = ctk.CTkLabel(self, text="Image Clustering Tool", font=("Helvetica", 24, "bold"))
        title_label.pack(pady=20)

        top_button_frame = ctk.CTkFrame(self, corner_radius=15)
        top_button_frame.pack(pady=20, padx=20, fill="x")

        upload_button = ctk.CTkButton(top_button_frame, text="Upload Image Set", command=self.upload_images)
        upload_button.pack(side="left", padx=20, pady=20)

        pca_button = ctk.CTkButton(top_button_frame, text="Perform PCA Clustering", command=self.perform_pca_clustering)
        pca_button.pack(side="left", padx=20, pady=20)

        umap_button = ctk.CTkButton(top_button_frame, text="Perform UMAP Clustering", command=self.perform_umap_clustering)
        umap_button.pack(side="left", padx=20, pady=20)

        recent_frame = ctk.CTkFrame(self, corner_radius=15)
        recent_frame.pack(pady=10, padx=20, fill="x")
        recent_label = ctk.CTkLabel(recent_frame, text="Recent 4 Images:", font=("Helvetica", 18))
        recent_label.pack(side="left", padx=10)
        self.recent_frame = recent_frame

        progress_frame = ctk.CTkFrame(self, corner_radius=15)
        progress_frame.pack(pady=20, fill="x")
        self.progress_label = ctk.CTkLabel(progress_frame, text="Progress: 0%", font=("Helvetica", 18))
        self.progress_label.pack(pady=10)
        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=600)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)

        results_frame = ctk.CTkFrame(self, corner_radius=15)
        results_frame.pack(pady=20, padx=20, fill="both", expand=True)
        self.table = ttk.Treeview(results_frame, columns=("Image Name", "User Label"), show="headings", height=15)
        self.table.heading("Image Name", text="Image Name")
        self.table.heading("User Label", text="User Label")
        self.table.column("Image Name", width=300, anchor="center")
        self.table.column("User Label", width=250, anchor="center")
        self.table.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        bottom_button_frame = ctk.CTkFrame(self, corner_radius=15)
        bottom_button_frame.pack(side="bottom", pady=20, padx=20, fill="x")
        plot_pca_button = ctk.CTkButton(bottom_button_frame, text="Plot PCA Clustering", command=self.plot_pca_clustering)
        plot_pca_button.pack(side="left", padx=20, pady=20)
        plot_umap_button = ctk.CTkButton(bottom_button_frame, text="Plot UMAP Clustering", command=self.plot_umap_clustering)
        plot_umap_button.pack(side="left", padx=20, pady=20)

    # ----- Helper Methods -----
    def preprocess_image(self, image_path, target_size=300):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image.")
        img = cv2.resize(img, (target_size, target_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = preprocess_input(img)
        return np.expand_dims(img, axis=0)

    def upload_images(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_paths:
            return
        label = simpledialog.askstring("Input Label", "Enter a label for this set of images:")
        if not label:
            messagebox.showwarning("Warning", "Label cannot be empty. Please enter a valid label.")
            return
        total = len(file_paths)
        for i, file_path in enumerate(file_paths):
            try:
                img_data = self.preprocess_image(file_path)
                features = self.model.predict(img_data)
                self.image_data.append(file_path)
                self.features_list.append(features.flatten())
                self.labels_list.append(label)
            except Exception as e:
                self.progress_label.configure(text=f"Error processing {os.path.basename(file_path)}: {e}")
            self.progress_bar.set((i + 1) / total)
            self.progress_label.configure(text=f"Uploaded {i + 1}/{total} images")
            self.update_idletasks()
            time.sleep(0.1)  # simulate processing delay
        self.progress_label.configure(text="Image upload complete!")
        self.update_table()
        self.update_recent_images()

    def update_table(self):
        self.table.delete(*self.table.get_children())
        for i, file_path in enumerate(self.image_data):
            image_name = os.path.basename(file_path)
            lab = self.labels_list[i]
            self.table.insert("", "end", values=(image_name, lab))

    def update_recent_images(self):
        for widget in self.recent_frame.winfo_children():
            if isinstance(widget, ctk.CTkLabel) and widget.cget("text") == "":
                widget.destroy()
        recent_count = min(4, len(self.image_data))
        for file_path in self.image_data[-recent_count:]:
            try:
                img = Image.open(file_path)
                img.thumbnail((100, 100))
                img_tk = ImageTk.PhotoImage(img)
                lbl = ctk.CTkLabel(self.recent_frame, image=img_tk, text="")
                lbl.image = img_tk  # keep reference
                lbl.pack(side="left", padx=5)
            except Exception as e:
                print(f"Error loading thumbnail for {file_path}: {e}")

    def perform_pca_clustering(self):
        if not self.features_list:
            messagebox.showwarning("Warning", "No images uploaded for clustering!")
            return
        self.progress_label.configure(text="Performing PCA clustering...")
        self.progress_bar.set(0)
        self.update_idletasks()
        total = len(self.features_list)
        for i in range(total):
            self.progress_bar.set(((i + 1) / total) * 0.5)
            self.progress_label.configure(text=f"PCA: Processing image {i + 1}/{total}")
            self.update_idletasks()
            time.sleep(0.05)
        try:
            data = np.array(self.features_list)
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(data)
            self.pca_result = principal_components
            self.pca_labels = np.array(self.labels_list)
        except Exception as e:
            self.progress_label.configure(text=f"Error during PCA: {e}")
            return
        self.progress_bar.set(1.0)
        self.progress_label.configure(text="PCA Clustering complete!")
        self.update_idletasks()

    def perform_umap_clustering(self):
        if not self.features_list:
            messagebox.showwarning("Warning", "No images uploaded for clustering!")
            return
        self.progress_label.configure(text="Performing UMAP clustering...")
        self.progress_bar.set(0)
        self.update_idletasks()
        total = len(self.features_list)
        for i in range(total):
            self.progress_bar.set(((i + 1) / total) * 0.5)
            self.progress_label.configure(text=f"UMAP: Processing image {i + 1}/{total}")
            self.update_idletasks()
            time.sleep(0.05)
        try:
            data = np.array(self.features_list)
            reducer = umap.UMAP(n_components=2, random_state=42)
            self.umap_result = reducer.fit_transform(data)
            self.umap_labels = np.array(self.labels_list)
        except Exception as e:
            self.progress_label.configure(text=f"Error during UMAP: {e}")
            return
        self.progress_bar.set(1.0)
        self.progress_label.configure(text="UMAP Clustering complete!")
        self.update_idletasks()

    def plot_pca_clustering(self):
        if self.pca_result is None:
            messagebox.showwarning("Warning", "No PCA clustering data available. Please run PCA clustering first.")
            return
        plt.figure(figsize=(15, 10), dpi=400)
        unique_labels = np.unique(self.pca_labels)
        for lab in unique_labels:
            mask = (self.pca_labels == lab)
            plt.scatter(self.pca_result[mask, 0], self.pca_result[mask, 1], label=lab)
        plt.xlabel("Principal Component 1", fontsize=24)
        plt.ylabel("Principal Component 2", fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        lgnd = plt.legend(fontsize=22)
        for handle in lgnd.legend_handles:
            handle.set_sizes([100])
        plt.gca().set_facecolor("white")
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png")],
                                                 title="Save PCA Clustering Plot")
        if file_path:
            try:
                plt.savefig(file_path, bbox_inches="tight", facecolor="white", edgecolor="white", dpi=400)
                messagebox.showinfo("Success", f"PCA Clustering plot saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving PCA plot: {e}")
        plt.close()

    def plot_umap_clustering(self):
        if self.umap_result is None:
            messagebox.showwarning("Warning", "No UMAP clustering data available. Please run UMAP clustering first.")
            return
        plt.figure(figsize=(15, 10), dpi=400)
        unique_labels = np.unique(self.umap_labels)
        for lab in unique_labels:
            mask = (self.umap_labels == lab)
            plt.scatter(self.umap_result[mask, 0], self.umap_result[mask, 1], label=lab)
        plt.xlabel("UMAP Dimension 1", fontsize=24)
        plt.ylabel("UMAP Dimension 2", fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        lgnd = plt.legend(fontsize=22)
        for handle in lgnd.legend_handles:
            handle.set_sizes([100])
        plt.gca().set_facecolor("white")
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png")],
                                                 title="Save UMAP Clustering Plot")
        if file_path:
            try:
                plt.savefig(file_path, bbox_inches="tight", facecolor="white", edgecolor="white", dpi=400)
                messagebox.showinfo("Success", f"UMAP Clustering plot saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving UMAP plot: {e}")
        plt.close()

# ==============================================================================
# Main Application with Sidebar Menu
# ==============================================================================
class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ARS Image Analysis Software")
        # Dynamically get the screen dimensions
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        # Set the window to fill the screen and maximize the window
        self.geometry(f"{screen_width}x{screen_height}")
        self.state("zoomed")

        # Configure grid for a responsive layout
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar Frame (menu) with a fixed width of 200
        sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        sidebar_frame.grid(row=0, column=0, sticky="ns")
        sidebar_frame.grid_propagate(False)

        title_label = ctk.CTkLabel(sidebar_frame, text="Menu", font=("Helvetica", 24, "bold"))
        title_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        button_predict = ctk.CTkButton(sidebar_frame, text="Predict ARS ODs", command=self.show_predict)
        button_predict.grid(row=1, column=0, padx=20, pady=10)

        button_classifier = ctk.CTkButton(sidebar_frame, text="Falsely Triggered Classifier", command=self.show_classifier)
        button_classifier.grid(row=2, column=0, padx=20, pady=10)

        button_clustering = ctk.CTkButton(sidebar_frame, text="ARS Image Clustering", command=self.show_clustering)
        button_clustering.grid(row=3, column=0, padx=20, pady=10)

        # Display a 200x200 ARS.png image below the menu buttons
        try:
            ars_image = Image.open(resource_path("ARS.png"))
            ars_image = ars_image.resize((200, 200), Image.LANCZOS)
            ars_image_tk = ImageTk.PhotoImage(ars_image)
            icon_label = ctk.CTkLabel(sidebar_frame, image=ars_image_tk, text="")
            icon_label.image = ars_image_tk  # keep a reference
            icon_label.grid(row=4, column=0, padx=20, pady=(20, 10))
        except Exception as e:
            print("Error loading ARS.png for sidebar:", e)

        # Container for pages (the three tools)
        container_width = screen_width - 200
        container_height = screen_height
        self.container = ctk.CTkFrame(self, width=container_width, height=container_height)
        self.container.grid(row=0, column=1, sticky="nsew")
        self.container.grid_propagate(False)

        # Instantiate each tool’s frame
        self.predict_frame = PredictODFrame(self.container)
        self.classifier_frame = FalselyClassifierFrame(self.container)
        self.clustering_frame = ClusteringFrame(self.container)

        for frame in (self.predict_frame, self.classifier_frame, self.clustering_frame):
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_predict()

    def show_predict(self):
        self.predict_frame.tkraise()

    def show_classifier(self):
        self.classifier_frame.tkraise()

    def show_clustering(self):
        self.clustering_frame.tkraise()

# ==============================================================================
# Run the Application
# ==============================================================================
if __name__ == "__main__":
    app = MainApp()
    app.mainloop()

