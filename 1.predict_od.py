import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import joblib
import os
import pandas as pd
import sys
from tkinter import VERTICAL
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt

# Configure CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Helper function for resource paths (for PyInstaller compatibility)
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Load the saved TFLite model and scaler
tflite_model_path = resource_path("regression_model.tflite")
scaler_path = resource_path("standard_scaler.pkl")

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scaler = joblib.load(scaler_path)

# Global variables
od_zero_values = []
od_zero_images_list = []  # For storing OD Zero thumbnail images
ars_images_list = []      # For storing ARS thumbnail images
stop_process = False
results = []

# Preprocess image for model inference (always 512x512)
def preprocess_image(image_path, target_size=(512, 512)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0).astype(np.float32)

# Resize image for GUI display (always 128x128)
def resize_image_for_gui(image_path, display_size=(128, 128)):
    img = Image.open(image_path)
    img.thumbnail(display_size)
    img = ImageOps.expand(img, border=5, fill="black")
    return ctk.CTkImage(light_image=img, size=display_size)

def update_od_zero_images_display():
    # Clear previous thumbnails
    for widget in od_zero_images_frame.winfo_children():
        widget.destroy()
    # Add a left spacer so the thumbnails are right-aligned
    spacer = ctk.CTkLabel(od_zero_images_frame, text="")
    spacer.pack(side="left", fill="x", expand=True)
    # Pack each thumbnail
    for img in od_zero_images_list:
        lbl = ctk.CTkLabel(od_zero_images_frame, image=img, text="")
        lbl.image = img  # Keep reference to avoid garbage collection
        lbl.pack(side="left", padx=5)

def update_ars_images_display():
    for widget in ars_images_frame.winfo_children():
        widget.destroy()
    spacer = ctk.CTkLabel(ars_images_frame, text="")
    spacer.pack(side="left", fill="x", expand=True)
    for img in ars_images_list:
        lbl = ctk.CTkLabel(ars_images_frame, image=img, text="")
        lbl.image = img
        lbl.pack(side="left", padx=5)

def upload_od_zero():
    global od_zero_values, od_zero_images_list
    file_paths = filedialog.askopenfilenames(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_paths:
        return
    total_images = len(file_paths)
    od_zero_progress_bar.set(0)
    for index, file_path in enumerate(file_paths):
        try:
            image_data = preprocess_image(file_path)
            # Run inference using TFLite
            interpreter.set_tensor(input_details[0]["index"], image_data)
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]["index"])
            # Calculate the OD Zero value
            od_zero_value = scaler.inverse_transform([[pred[0][0]]])[0][0]
            od_zero_values.append(od_zero_value)
            # Create and store thumbnail image
            img_ctk = resize_image_for_gui(file_path)
            od_zero_images_list.append(img_ctk)
            if len(od_zero_images_list) > 5:
                od_zero_images_list.pop(0)
            update_od_zero_images_display()
            # Update the OD Zero progress bar
            progress = (index + 1) / total_images
            od_zero_progress_bar.set(progress)
            app.update_idletasks()
        except Exception as e:
            messagebox.showerror("Error", f"Error processing OD Zero image: {e}")

def predict_images():
    global stop_process, results, ars_images_list
    stop_process = False
    # Calculate OD Zero threshold from uploaded OD Zero images
    if od_zero_values:
        od_zero_threshold = np.mean(od_zero_values)
    else:
        od_zero_threshold = 0
        messagebox.showwarning("Warning", "No OD Zero image uploaded. Using threshold = 0.")
    file_paths = filedialog.askopenfilenames(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_paths:
        return
    total_images = len(file_paths)
    progress_label.configure(text=f"Progress: 0/{total_images} images processed")
    progress_bar.set(0)
    for index, file_path in enumerate(file_paths):
        if stop_process:
            break
        try:
            image_data = preprocess_image(file_path)
            interpreter.set_tensor(input_details[0]["index"], image_data)
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]["index"])
            predicted_od = scaler.inverse_transform([[pred[0][0]]])[0][0] - od_zero_threshold
            results.append({
                "Image Name": os.path.basename(file_path),
                "Predicted OD": predicted_od,
                "Image Path": file_path,
            })
            table.insert("", "end", values=(os.path.basename(file_path), f"{predicted_od:.4f}", file_path))
            img_ctk = resize_image_for_gui(file_path)
            ars_images_list.append(img_ctk)
            if len(ars_images_list) > 5:
                ars_images_list.pop(0)
            update_ars_images_display()
            progress = (index + 1) / total_images
            progress_bar.set(progress)
            progress_label.configure(text=f"Progress: {index + 1}/{total_images} images processed")
            app.update_idletasks()
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def export_to_excel():
    if not results:
        messagebox.showwarning("Warning", "No results to export.")
        return
    file_path = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx")],
        title="Save Results as Excel File",
    )
    if not file_path:
        return
    try:
        df = pd.DataFrame(results)
        df.to_excel(file_path, index=False)
        messagebox.showinfo("Success", f"Results saved to {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save results: {e}")

def stop_prediction():
    global stop_process
    stop_process = True

def clear_predictions():
    global results, ars_images_list
    for item in table.get_children():
        table.delete(item)
    results = []
    ars_images_list = []
    update_ars_images_display()
    progress_bar.set(0)
    progress_label.configure(text="Progress: 0/0 images processed")

def plot_histogram():
    if not results:
        messagebox.showwarning("Warning", "No results to plot.")
        return
    od_values = [result["Predicted OD"] for result in results]
    plt.figure(figsize=(8, 5))
    n, bins, patches = plt.hist(od_values, bins=10, edgecolor='black', alpha=0.7)
    plt.xlabel("Predicted OD Values")
    plt.ylabel("Count")
    plt.title("Distribution of Predicted Optical Densities")
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png")],
        title="Save Histogram As"
    )
    if file_path:
        plt.savefig(file_path, dpi=400)
        messagebox.showinfo("Success", f"Histogram saved to {file_path}")
    plt.close()

# ---------------------- Main GUI Layout Using Grid ----------------------
app = ctk.CTk()
app.title("ARS OD Prediction Tool")
app.geometry("1200x800")
app.grid_columnconfigure(0, weight=1)
app.grid_rowconfigure(5, weight=1)  # Table frame gets extra vertical space

# Title Label
title_label = ctk.CTkLabel(app, text="ARS OD Prediction Tool", font=("Helvetica", 24, "bold"))
title_label.grid(row=0, column=0, padx=20, pady=(20, 10))

# ----- OD Zero Section -----
# OD Zero Frame now has two rows: row 0 for header & images; row 1 for the OD Zero progress bar.
od_zero_frame = ctk.CTkFrame(app, corner_radius=15)
od_zero_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
# Use 4 columns to mimic ARS section: label, upload button, spacer, image display.
od_zero_frame.grid_columnconfigure(0, weight=0)
od_zero_frame.grid_columnconfigure(1, weight=0)
od_zero_frame.grid_columnconfigure(2, weight=0)
od_zero_frame.grid_columnconfigure(3, weight=1)

od_zero_label = ctk.CTkLabel(od_zero_frame, text="Step 1: Upload OD Zero Images", font=("Helvetica", 18, "bold"))
od_zero_label.grid(row=0, column=0, padx=20, pady=10, sticky="w")

od_zero_button = ctk.CTkButton(od_zero_frame, text="Upload OD Zero Images", command=upload_od_zero)
od_zero_button.grid(row=0, column=1, padx=20, pady=10, sticky="w")

# Spacer to mimic the stop button space in ARS section
od_zero_spacer = ctk.CTkLabel(od_zero_frame, text="", width=150)
od_zero_spacer.grid(row=0, column=2, padx=20, pady=10, sticky="w")

od_zero_images_frame = ctk.CTkFrame(od_zero_frame, width=600, height=150)
od_zero_images_frame.grid(row=0, column=3, padx=20, pady=10)
od_zero_images_frame.grid_propagate(False)

# New OD Zero progress bar (for dynamic progress during OD Zero uploads)
od_zero_progress_bar = ctk.CTkProgressBar(od_zero_frame, width=600)
od_zero_progress_bar.grid(row=1, column=0, columnspan=4, sticky="ew", padx=20, pady=(0,10))
od_zero_progress_bar.set(0)

# ----- ARS Section -----
ars_frame = ctk.CTkFrame(app, corner_radius=15)
ars_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=10)
ars_frame.grid_columnconfigure(0, weight=0)
ars_frame.grid_columnconfigure(1, weight=0)
ars_frame.grid_columnconfigure(2, weight=0)
ars_frame.grid_columnconfigure(3, weight=1)

ars_label = ctk.CTkLabel(ars_frame, text="Step 2: Upload ARS Images", font=("Helvetica", 18, "bold"))
ars_label.grid(row=0, column=0, padx=20, pady=10, sticky="w")

ars_button = ctk.CTkButton(ars_frame, text="Upload ARS Images", command=predict_images)
ars_button.grid(row=0, column=1, padx=20, pady=10, sticky="w")

stop_button = ctk.CTkButton(ars_frame, text="Stop Prediction", command=stop_prediction)
stop_button.grid(row=0, column=2, padx=20, pady=10, sticky="w")

ars_images_frame = ctk.CTkFrame(ars_frame, width=600, height=150)
ars_images_frame.grid(row=0, column=3, padx=20, pady=10)
ars_images_frame.grid_propagate(False)

# ----- Main Progress Bar for ARS Predictions -----
progress_label = ctk.CTkLabel(app, text="Progress: 0/0 images processed", font=("Helvetica", 18))
progress_label.grid(row=3, column=0, sticky="ew", padx=20, pady=(10, 5))

progress_bar = ctk.CTkProgressBar(app, width=600)
progress_bar.grid(row=4, column=0, sticky="ew", padx=20, pady=(20,20))
progress_bar.set(0)

# ----- Table for Results -----
table_frame = ctk.CTkFrame(app)
table_frame.grid(row=5, column=0, sticky="nsew", padx=20, pady=10)

columns = ("Image Name", "Predicted OD", "Image Path")
table = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
for col in columns:
    table.heading(col, text=col)
    table.column(col, width=200, anchor="center")

v_scroll = ttk.Scrollbar(table_frame, orient=VERTICAL, command=table.yview)
table.configure(yscrollcommand=v_scroll.set)
table.pack(side="left", fill="both", expand=True)
v_scroll.pack(side="right", fill="y")

# ----- Control Buttons (Always Visible at the Bottom) -----
buttons_frame = ctk.CTkFrame(app)
buttons_frame.grid(row=6, column=0, sticky="ew", padx=20, pady=10)

clear_button = ctk.CTkButton(buttons_frame, text="Clear Predictions", command=clear_predictions)
clear_button.pack(side="left", padx=10)

plot_button = ctk.CTkButton(buttons_frame, text="Plot Histogram", command=plot_histogram)
plot_button.pack(side="left", padx=10)

export_button = ctk.CTkButton(buttons_frame, text="Export as Excel File", command=export_to_excel)
export_button.pack(side="left", padx=10)

# Run the application
app.mainloop()







