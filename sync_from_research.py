# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 00:44:32 2025

@author: sinethi
"""

import os
import shutil

# === Paths ===
RESEARCH_FOLDER = r"C:\Users\sinethi\OneDrive\Documents\Research"
PROJECT_IMAGES_FOLDER = "images"
PROJECT_MODEL_PATH = "copd_model.pth"

# Ensure project images folder exists
os.makedirs(PROJECT_IMAGES_FOLDER, exist_ok=True)

# === Copy latest model ===
source_model = os.path.join(RESEARCH_FOLDER, "copd_model.pth")
if os.path.exists(source_model):
    shutil.copy2(source_model, PROJECT_MODEL_PATH)
    print(f"✅ Model copied: {source_model} -> {PROJECT_MODEL_PATH}")
else:
    print(f"⚠️ Model not found at {source_model}")

# === Copy all images ===
source_images_folder = os.path.join(RESEARCH_FOLDER, "images")
if os.path.exists(source_images_folder):
    for filename in os.listdir(source_images_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            src = os.path.join(source_images_folder, filename)
            dst = os.path.join(PROJECT_IMAGES_FOLDER, filename)
            try:
                shutil.copy2(src, dst)
                print(f"✅ Copied image: {filename}")
            except Exception as e:
                print(f"⚠️ Failed to copy {filename}: {e}")
else:
    print(f"⚠️ Images folder not found at {source_images_folder}")
