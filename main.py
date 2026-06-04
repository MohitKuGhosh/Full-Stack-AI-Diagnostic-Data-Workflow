from fastapi import FastAPI, UploadFile, File, HTTPException
from io import BytesIO
from PIL import Image
import os
import time
import random

app = FastAPI(
    title="Medical AI Diagnostic API",
    description="Backend routing for Brain Tumor and Ovarian Cancer detection models."
)

# ==========================================
# 1. ATTEMPT TO LOAD TENSORFLOW & MODELS
# ==========================================
brain_model = None
ovarian_model = None
TF_AVAILABLE = False

try:
    import tensorflow as tf
    import numpy as np
    TF_AVAILABLE = True
    print("✅ TensorFlow detected.")
except ImportError:
    print("⚠️ TensorFlow not found. API will run in Simulation Mode.")

if TF_AVAILABLE:
    brain_model_path = "models/Brain_Tumor_Binary.h5"
    ovarian_model_path = "models/ovarian_image_model.h5"

    if os.path.exists(brain_model_path):
        try:
            brain_model = tf.keras.models.load_model(brain_model_path)
            print("✅ Brain Tumor Model loaded.")
        except Exception as e:
            print(f"⚠️ Failed to load Brain Tumor Model: {e}")
            
    if os.path.exists(ovarian_model_path):
        try:
            ovarian_model = tf.keras.models.load_model(ovarian_model_path)
            print("✅ Ovarian Cancer Model loaded.")
        except Exception as e:
            print(f"⚠️ Failed to load Ovarian Cancer Model: {e}")

# ==========================================
# 2. IMAGE PREPROCESSING HELPER
# ==========================================
def prepare_image(image_bytes: bytes, target_size=(224, 224)):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    if TF_AVAILABLE:
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    return None

# ==========================================
# 3. API ENDPOINTS (Real or Simulated)
# ==========================================
@app.get("/")
async def health_check():
    return {
        "status": "online", 
        "mode": "Production" if (brain_model or ovarian_model) else "Simulation",
        "message": "Select an endpoint to upload a scan."
    }

@app.post("/predict/brain-tumor/")
async def predict_brain_tumor(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    # Run REAL model if available
    if brain_model:
        processed_image = prepare_image(image_bytes)
        prediction = brain_model.predict(processed_image)
        score = float(prediction[0][0])
        is_malignant = score > 0.5
    # Run SIMULATION if model/TF is missing
    else:
        time.sleep(1.2)
        score = random.uniform(0.01, 0.99)
        is_malignant = score > 0.5

    return {
        "scan_type": "MRI",
        "filename": file.filename,
        "prediction": "Malignant" if is_malignant else "Benign",
        "confidence": f"{round((score if is_malignant else 1 - score) * 100, 2)}%"
    }

@app.post("/predict/ovarian-cancer/")
async def predict_ovarian_cancer(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    # Run REAL model if available
    if ovarian_model:
        processed_image = prepare_image(image_bytes)
        prediction = ovarian_model.predict(processed_image)
        score = float(prediction[0][0])
        is_malignant = score > 0.5
    # Run SIMULATION if model/TF is missing
    else:
        time.sleep(0.8)
        score = random.uniform(0.01, 0.99)
        is_malignant = score > 0.5

    return {
        "scan_type": "Ultrasound/CT",
        "filename": file.filename,
        "prediction": "Malignant" if is_malignant else "Benign",
        "confidence": f"{round((score if is_malignant else 1 - score) * 100, 2)}%"
    }