import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageOps
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import sys
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import preprocess_input

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

# Load the saved TFLite model
tflite_model_path = resource_path('classifier_model.tflite')

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Global variables
results = []
last_five_images = []  
correct_count = 0
false_count = 0

# Function to preprocess images
def preprocess_image(image_path, target_size=512):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (target_size, target_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = preprocess_input(img)  
    return np.expand_dims(img, axis=0)

# Predict images
def predict_images():
    global results, last_five_images, correct_count, false_count

    file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_paths:
        return

    total_images = len(file_paths)
    progress_label.configure(text=f"Progress: 0/{total_images} images processed")
    progress_bar.set(0)

    for index, file_path in enumerate(file_paths):
        try:
            image_data = preprocess_image(file_path)

            # Run inference using TFLite
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]['index'])

            # Corrected prediction handling
            predicted_label = np.argmax(pred, axis=1)[0]
            label_text = "Correctly Triggered" if predicted_label == 0 else "Falsely Triggered"

            # Count correctly vs falsely triggered
            if predicted_label == 0:
                correct_count += 1
            else:
                false_count += 1

            # Add result to the table
            results.append({"Image Name": os.path.basename(file_path), "Predicted Label": label_text})
            table.insert("", "end", values=(os.path.basename(file_path), label_text))

            # Update last five images
            img = Image.open(file_path)
            img.thumbnail((200, 200))
            img = ImageOps.expand(img, border=10, fill="black")
            img_tk = ImageTk.PhotoImage(img)
            last_five_images.append(img_tk)
            if len(last_five_images) > 5:
                last_five_images.pop(0)

            # Display the last five images
            for i, label in enumerate(image_labels):
                if i < len(last_five_images):
                    label.configure(image=last_five_images[i])
                    label.image = last_five_images[i]
                else:
                    label.configure(image=None)
                    label.image = None

            # Update progress bar
            progress = (index + 1) / total_images
            progress_bar.set(progress)
            progress_label.configure(text=f"Progress: {index + 1}/{total_images} images processed")
            app.update_idletasks()

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Function to plot and save results
def plot_results():
    if not results:
        messagebox.showerror("No Data", "Please run predictions first.")
        return

    # Count labels in the current table
    correct_count = sum(1 for row in results if row["Predicted Label"] == "Correctly Triggered")
    false_count = sum(1 for row in results if row["Predicted Label"] == "Falsely Triggered")

    file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG Image", "*.png")])
    if not file_path:
        return

    # Generate bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(["Correctly Triggered", "Falsely Triggered"], [correct_count, false_count], color=["green", "red"])
    plt.title("Prediction Summary")
    plt.ylabel("Number of Images")

    # Save chart as a high-resolution PNG file
    plt.savefig(file_path, dpi=400)
    plt.close()
    messagebox.showinfo("Saved", f"Results saved as {file_path}")

# Function to export results to an Excel file
def export_excel():
    if not results:
        messagebox.showerror("No Data", "No results to export.")
        return

    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                             filetypes=[("Excel File", "*.xlsx")])
    if not file_path:
        return

    df = pd.DataFrame(results)
    df.to_excel(file_path, index=False)
    messagebox.showinfo("Excel Saved", f"Results saved to {file_path}")

# Function to clear all predictions
def clear_predictions():
    global results, correct_count, false_count
    results.clear()
    correct_count = 0
    false_count = 0
    table.delete(*table.get_children())  # Clear table
    messagebox.showinfo("Cleared", "All predictions have been removed.")

# Create the GUI application
app = ctk.CTk()
app.title("Falsely Triggered Image Classification Tool")
app.geometry("1920x1080")

# Title Label
title_label = ctk.CTkLabel(app, text="Falsely Triggered Image Classification Tool", font=("Helvetica", 24, "bold"))
title_label.pack(pady=20)

# Image Upload Frame
image_frame = ctk.CTkFrame(app, corner_radius=15)
image_frame.pack(pady=20, padx=20, fill="x")

image_label = ctk.CTkLabel(image_frame, text="Upload Images for Classification", font=("Helvetica", 18, "bold"))
image_label.pack(side="left", padx=20)

upload_button = ctk.CTkButton(image_frame, text="Upload Images", command=predict_images)
upload_button.pack(side="left", padx=20, pady=20)

# Display Last Five Images
image_labels = [ctk.CTkLabel(image_frame, text="") for _ in range(5)]
for label in image_labels:
    label.pack(side="left", padx=10)

# Progress Bar
progress_frame = ctk.CTkFrame(app, corner_radius=15)
progress_frame.pack(pady=20, fill="x")

progress_label = ctk.CTkLabel(progress_frame, text="Progress: 0/0 images processed", font=("Helvetica", 18))
progress_label.pack(pady=10)

progress_bar = ctk.CTkProgressBar(progress_frame, width=600)
progress_bar.pack(pady=10)
progress_bar.set(0)

# Results Table
results_frame = ctk.CTkFrame(app, corner_radius=15)
results_frame.pack(pady=20, padx=20, fill="both", expand=True)

table = ttk.Treeview(results_frame, columns=("Image Name", "Predicted Label"), show="headings", height=15)
table.heading("Image Name", text="Image Name")
table.heading("Predicted Label", text="Predicted Label")
table.column("Image Name", width=300, anchor="center")
table.column("Predicted Label", width=250, anchor="center")
table.pack(side="left", fill="both", expand=True)

scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=table.yview)
table.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")

# Buttons for exporting results
button_frame = ctk.CTkFrame(app, corner_radius=15)
button_frame.pack(pady=20, fill="x")

plot_button = ctk.CTkButton(button_frame, text="Plot Results", command=plot_results)
plot_button.pack(side="left", padx=20, pady=10)

export_button = ctk.CTkButton(button_frame, text="Export as Excel File", command=export_excel)
export_button.pack(side="left", padx=20, pady=10)

clear_button = ctk.CTkButton(button_frame, text="Clear Predictions", command=clear_predictions)
clear_button.pack(side="left", padx=20, pady=10)

# Run the application
app.mainloop()



