# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 19:41:13 2025

@author: sinethi
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sqlite3
import os
import gradio as gr

# === Constants ===
DB_PATH = "xrays.db"
MODEL_PATH = "copd_model.pth"
CLASS_NAMES = ['Normal', 'COPD']

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Model ===
def load_model():
    model_instance = models.resnet18(pretrained=False)
    model_instance.fc = nn.Linear(model_instance.fc.in_features, 2)
    if os.path.exists(MODEL_PATH):
        model_instance.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model_instance = model_instance.to(device)
    model_instance.eval()
    return model_instance

model = load_model()

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Database Functions ===
def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL UNIQUE,
            patient_name TEXT NOT NULL,
            image_path TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def get_patient_names():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT patient_name FROM images")
    names = [row[0] for row in cursor.fetchall()]
    conn.close()
    return names

def get_xray_from_db(patient_name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT image_path FROM images WHERE patient_name=?", (patient_name,))
    result = cursor.fetchone()
    conn.close()
    if result and os.path.exists(result[0]):
        return Image.open(result[0])
    return None

# === Prediction Function ===
def predict_copd(patient_name):
    image = get_xray_from_db(patient_name)
    if image is None:
        return None, f"Image not found for {patient_name}"
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
    prediction = {CLASS_NAMES[0]: float(output[0][0]), CLASS_NAMES[1]: float(output[0][1])}
    return image, prediction

# === Main ===
if __name__ == "__main__":
    setup_database()
    names = get_patient_names()
    if not names:
        names = ["No patients in DB"]

    iface = gr.Interface(
        fn=predict_copd,
        inputs=gr.Dropdown(choices=names, label="Select Patient Name"),
        outputs=[gr.Image(label="X-ray Image"), gr.Label(num_top_classes=2, label="COPD Prediction")],
        title="COPD Chest X-ray Classifier",
        description="Select a patient to view X-ray and COPD prediction."
    )

    # Render requires port 10000
    iface.launch(server_name="0.0.0.0", server_port=10000)
