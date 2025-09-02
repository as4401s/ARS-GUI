import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.decomposition import PCA
import cv2
import time
import umap  # Make sure you have installed umap-learn

# ---------------------- Configuration ----------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Load EfficientNetB3 model (without top layers)
model = EfficientNetB3(weights="imagenet", include_top=False, pooling="avg")

# ---------------------- Global Variables ----------------------
image_data = []       # List of image file paths
features_list = []    # List of extracted features (flattened vectors)
labels_list = []      # List of user-provided labels corresponding to each image
pca_result = None     # Will hold the 2D PCA result
pca_labels = None     # Will hold the labels for PCA
umap_result = None    # Will hold the 2D UMAP result
umap_labels = None    # Will hold the labels for UMAP

# ---------------------- Functions ----------------------
def preprocess_image(image_path, target_size=300):
    """Preprocess image for EfficientNetB3."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image.")
    img = cv2.resize(img, (target_size, target_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

def upload_images():
    """Upload images, extract features, and update progress."""
    global image_data, features_list, labels_list
    file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_paths:
        return

    # Ask user for a label for this set of images
    label = simpledialog.askstring("Input Label", "Enter a label for this set of images:")
    if not label:
        messagebox.showwarning("Warning", "Label cannot be empty. Please enter a valid label.")
        return

    total = len(file_paths)
    for i, file_path in enumerate(file_paths):
        try:
            img_data = preprocess_image(file_path)
            features = model.predict(img_data)
            image_data.append(file_path)
            features_list.append(features.flatten())
            labels_list.append(label)
        except Exception as e:
            progress_label.configure(text=f"Error processing {os.path.basename(file_path)}: {e}")
        progress_bar.set((i+1) / total)
        progress_label.configure(text=f"Uploaded {i+1}/{total} images")
        app.update_idletasks()
        time.sleep(0.1)  # Simulate processing delay

    progress_label.configure(text="Image upload complete!")
    update_table()
    update_recent_images()

def update_table():
    """Update the table with image names and their labels."""
    table.delete(*table.get_children())
    for i, file_path in enumerate(image_data):
        image_name = os.path.basename(file_path)
        lab = labels_list[i]
        table.insert("", "end", values=(image_name, lab))

def update_recent_images():
    """Display thumbnails of the most recent 5 uploaded images."""
    for widget in recent_frame.winfo_children():
        widget.destroy()
    recent_count = min(5, len(image_data))
    # Show the last 'recent_count' images
    for file_path in image_data[-recent_count:]:
        try:
            img = Image.open(file_path)
            img.thumbnail((100, 100))
            img_tk = ImageTk.PhotoImage(img)
            lbl = ctk.CTkLabel(recent_frame, image=img_tk, text="")
            lbl.image = img_tk  # Keep a reference
            lbl.pack(side="left", padx=5)
        except Exception as e:
            print(f"Error loading thumbnail for {file_path}: {e}")

def perform_pca_clustering():
    """Perform PCA clustering with dynamic progress updates."""
    global pca_result, pca_labels
    if not features_list:
        messagebox.showwarning("Warning", "No images uploaded for clustering!")
        return

    progress_label.configure(text="Performing PCA clustering...")
    progress_bar.set(0)
    app.update_idletasks()
    total = len(features_list)
    # Simulate per-image processing progress (first half of progress)
    for i in range(total):
        progress_bar.set(((i+1)/total) * 0.5)
        progress_label.configure(text=f"PCA: Processing image {i+1}/{total}")
        app.update_idletasks()
        time.sleep(0.05)
    try:
        data = np.array(features_list)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(data)
        pca_result = principal_components
        pca_labels = np.array(labels_list)
    except Exception as e:
        progress_label.configure(text=f"Error during PCA: {e}")
        return
    progress_bar.set(1.0)
    progress_label.configure(text="PCA Clustering complete!")
    app.update_idletasks()

def perform_umap_clustering():
    """Perform UMAP clustering with dynamic progress updates."""
    global umap_result, umap_labels
    if not features_list:
        messagebox.showwarning("Warning", "No images uploaded for clustering!")
        return

    progress_label.configure(text="Performing UMAP clustering...")
    progress_bar.set(0)
    app.update_idletasks()
    total = len(features_list)
    # Simulate per-image processing progress (first half of progress)
    for i in range(total):
        progress_bar.set(((i+1)/total) * 0.5)
        progress_label.configure(text=f"UMAP: Processing image {i+1}/{total}")
        app.update_idletasks()
        time.sleep(0.05)
    try:
        data = np.array(features_list)
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_result = reducer.fit_transform(data)
        umap_labels = np.array(labels_list)
    except Exception as e:
        progress_label.configure(text=f"Error during UMAP: {e}")
        return
    progress_bar.set(1.0)
    progress_label.configure(text="UMAP Clustering complete!")
    app.update_idletasks()

def plot_pca_clustering():
    """Plot the PCA clustering result and save as PNG at 400 dpi."""
    if pca_result is None:
        messagebox.showwarning("Warning", "No PCA clustering data available. Please run PCA clustering first.")
        return

    plt.figure(figsize=(15, 10), dpi=400)
    unique_labels = np.unique(pca_labels)
    for lab in unique_labels:
        mask = (pca_labels == lab)
        plt.scatter(pca_result[mask, 0], pca_result[mask, 1], label=lab)
    plt.xlabel("Principal Component 1", fontsize=24)
    plt.ylabel("Principal Component 2", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    lgnd = plt.legend(fontsize=22)
    for handle in lgnd.legend_handles:
        handle.set_sizes([100])
    plt.gca().set_facecolor("white")

    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png")],
        title="Save PCA Clustering Plot"
    )
    if file_path:
        try:
            plt.savefig(file_path, bbox_inches="tight", facecolor="white", edgecolor="white", dpi=400)
            messagebox.showinfo("Success", f"PCA Clustering plot saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving PCA plot: {e}")
    plt.close()

def plot_umap_clustering():
    """Plot the UMAP clustering result and save as PNG at 400 dpi."""
    if umap_result is None:
        messagebox.showwarning("Warning", "No UMAP clustering data available. Please run UMAP clustering first.")
        return

    plt.figure(figsize=(15, 10), dpi=400)
    unique_labels = np.unique(umap_labels)
    for lab in unique_labels:
        mask = (umap_labels == lab)
        plt.scatter(umap_result[mask, 0], umap_result[mask, 1], label=lab)
    plt.xlabel("UMAP Dimension 1", fontsize=24)
    plt.ylabel("UMAP Dimension 2", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    lgnd = plt.legend(fontsize=22)
    for handle in lgnd.legend_handles:
        handle.set_sizes([100])
    plt.gca().set_facecolor("white")

    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png")],
        title="Save UMAP Clustering Plot"
    )
    if file_path:
        try:
            plt.savefig(file_path, bbox_inches="tight", facecolor="white", edgecolor="white", dpi=400)
            messagebox.showinfo("Success", f"UMAP Clustering plot saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving UMAP plot: {e}")
    plt.close()

# ---------------------- GUI Layout ----------------------
app = ctk.CTk()
app.title("Image Clustering Tool")
app.geometry("1920x1080")

# Title Label
title_label = ctk.CTkLabel(app, text="Image Clustering Tool", font=("Helvetica", 24, "bold"))
title_label.pack(pady=20)

# Top Button Frame: Upload and Clustering Methods
top_button_frame = ctk.CTkFrame(app, corner_radius=15)
top_button_frame.pack(pady=20, padx=20, fill="x")

upload_button = ctk.CTkButton(top_button_frame, text="Upload Image Set", command=upload_images)
upload_button.pack(side="left", padx=20, pady=20)

pca_button = ctk.CTkButton(top_button_frame, text="Perform PCA Clustering", command=perform_pca_clustering)
pca_button.pack(side="left", padx=20, pady=20)

umap_button = ctk.CTkButton(top_button_frame, text="Perform UMAP Clustering", command=perform_umap_clustering)
umap_button.pack(side="left", padx=20, pady=20)

# Recent Images Frame (show the most recent 5 images)
recent_frame = ctk.CTkFrame(app, corner_radius=15)
recent_frame.pack(pady=10, padx=20, fill="x")
recent_label = ctk.CTkLabel(recent_frame, text="Recent 5 Images:", font=("Helvetica", 18))
recent_label.pack(side="left", padx=10)

# Progress Bar Frame
progress_frame = ctk.CTkFrame(app, corner_radius=15)
progress_frame.pack(pady=20, fill="x")
progress_label = ctk.CTkLabel(progress_frame, text="Progress: 0%", font=("Helvetica", 18))
progress_label.pack(pady=10)
progress_bar = ctk.CTkProgressBar(progress_frame, width=600)
progress_bar.pack(pady=10)
progress_bar.set(0)

# Results Table Frame
results_frame = ctk.CTkFrame(app, corner_radius=15)
results_frame.pack(pady=20, padx=20, fill="both", expand=True)
table = ttk.Treeview(results_frame, columns=("Image Name", "User Label"), show="headings", height=15)
table.heading("Image Name", text="Image Name")
table.heading("User Label", text="User Label")
table.column("Image Name", width=300, anchor="center")
table.column("User Label", width=250, anchor="center")
table.pack(side="left", fill="both", expand=True)
scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=table.yview)
table.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")

# Bottom Button Frame: Plot Buttons (placed at bottom left)
bottom_button_frame = ctk.CTkFrame(app, corner_radius=15)
bottom_button_frame.pack(side="bottom", pady=20, padx=20, fill="x")
plot_pca_button = ctk.CTkButton(bottom_button_frame, text="Plot PCA Clustering", command=plot_pca_clustering)
plot_pca_button.pack(side="left", padx=20, pady=20)
plot_umap_button = ctk.CTkButton(bottom_button_frame, text="Plot UMAP Clustering", command=plot_umap_clustering)
plot_umap_button.pack(side="left", padx=20, pady=20)

# Run the application
app.mainloop()




