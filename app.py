# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 19:41:13 2025

@author: sinethi
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import shutil
import gradio as gr

# === Constants ===
MODEL_PATH = "copd_model.pth"
CLASS_NAMES = ['Normal', 'COPD']
IMAGES_FOLDER = "images"

# === Research folder (source) ===
RESEARCH_FOLDER = r"C:\Users\sinethi\OneDrive\Documents\Research"

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Auto-sync function ===
def sync_from_research():
    # Ensure images folder exists
    os.makedirs(IMAGES_FOLDER, exist_ok=True)

    # Sync model
    source_model = os.path.join(RESEARCH_FOLDER, "copd_model.pth")
    if os.path.exists(source_model):
        shutil.copy2(source_model, MODEL_PATH)
        print(f"✅ Model synced: {source_model} -> {MODEL_PATH}")
    else:
        print(f"⚠️ Model not found at {source_model}")

    # Sync images
    source_images_folder = os.path.join(RESEARCH_FOLDER, "images")
    if os.path.exists(source_images_folder):
        for filename in os.listdir(source_images_folder):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                src = os.path.join(source_images_folder, filename)
                dst = os.path.join(IMAGES_FOLDER, filename)
                try:
                    shutil.copy2(src, dst)
                    print(f"✅ Copied image: {filename}")
                except Exception as e:
                    print(f"⚠️ Failed to copy {filename}: {e}")
    else:
        print(f"⚠️ Images folder not found at {source_images_folder}")

# === Model Loader (auto reload latest) ===
_model_cache = None
_model_mtime = None

def load_model():
    global _model_cache, _model_mtime
    if not os.path.exists(MODEL_PATH):
        return None
    mtime = os.path.getmtime(MODEL_PATH)
    if _model_cache is None or _model_mtime != mtime:
        model_instance = models.resnet18(pretrained=False)
        model_instance.fc = nn.Linear(model_instance.fc.in_features, 2)
        model_instance.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model_instance = model_instance.to(device)
        model_instance.eval()
        _model_cache = model_instance
        _model_mtime = mtime
    return _model_cache

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Auto-load patient images dynamically ===
def get_patient_images():
    images = {}
    for filename in os.listdir(IMAGES_FOLDER):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            patient_name = os.path.splitext(filename)[0]
            images[patient_name] = os.path.join(IMAGES_FOLDER, filename)
    return images

# === Prediction Function ===
def predict_copd(patient_name):
    images_dict = get_patient_images()
    image_path = images_dict.get(patient_name)
    if not image_path or not os.path.exists(image_path):
        return None, f"Image not found for {patient_name}"

    model_instance = load_model()
    if model_instance is None:
        return None, "Model not found"

    image = Image.open(image_path)
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model_instance(img_tensor)
    prediction = {CLASS_NAMES[0]: float(output[0][0]), CLASS_NAMES[1]: float(output[0][1])}
    return image, prediction

# === Main ===
if __name__ == "__main__":
    sync_from_research()  # Sync files at startup

    patient_names = list(get_patient_images().keys())
    if not patient_names:
        patient_names = ["No images found"]

    iface = gr.Interface(
        fn=predict_copd,
        inputs=gr.Dropdown(choices=patient_names, label="Select Patient Name"),
        outputs=[gr.Image(label="X-ray Image"), gr.Label(num_top_classes=2, label="COPD Prediction")],
        title="COPD Chest X-ray Classifier",
        description="Select a patient to view X-ray and COPD prediction."
    )

    iface.launch(server_name="0.0.0.0", server_port=10000)



