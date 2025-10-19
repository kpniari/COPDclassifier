# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 21:07:22 2025

@author: sinethi
"""

import sqlite3
import os
import shutil

# === Constants ===
DB_PATH = "xrays.db"
IMAGES_FOLDER = "images"

# Make sure the images folder exists
if not os.path.exists(IMAGES_FOLDER):
    os.makedirs(IMAGES_FOLDER)

# Connect to the database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get all image paths
cursor.execute("SELECT id, image_path FROM images")
rows = cursor.fetchall()

for row in rows:
    id_, old_path = row
    # Extract just the filename
    filename = os.path.basename(old_path)
    new_path = os.path.join(IMAGES_FOLDER, filename)

    # Copy image into images folder if not already there
    if os.path.exists(old_path):
        shutil.copy2(old_path, new_path)
    else:
        print(f"Warning: {old_path} not found. Skipping.")

    # Update database with new relative path
    cursor.execute("UPDATE images SET image_path=? WHERE id=?", (new_path, id_))

conn.commit()
conn.close()

print("✅ Database image paths updated and images copied to 'images/' folder.")
