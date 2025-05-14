import json
import os

import joblib
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from transformers import AutoModel, AutoTokenizer

# --- INIT APP ---
app = FastAPI()

# --- CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD MODELS ---
try:
    # Load trained multimodal model
    model = load_model("RiceSafeModel.keras")

    # Load label encoder
    label_encoder = joblib.load("label_encoder.pkl")

    # Load scaler
    scaler = joblib.load("scaler.pkl")

    # Load tokenizer and transformer model for text embedding
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    text_model = AutoModel.from_pretrained("BAAI/bge-m3").eval()

    # Load image model
    mobilenet = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    mobilenet.trainable = False

except Exception as e:
    raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")


# --- FEATURE COMBINATION ---
def preprocess(image_file, description: str):
    try:
        # Image preprocessing
        img = Image.open(image_file).convert("RGB").resize((224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        img_feat = mobilenet.predict(img_array)

        # Text preprocessing
        inputs = tokenizer(
            description, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            text_feat = text_model(**inputs).last_hidden_state.mean(dim=1).numpy()

        # Combine features
        combined_feat = np.concatenate((img_feat, text_feat), axis=1)

        # Scale features
        scaled_feat = scaler.transform(combined_feat)

        return scaled_feat
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing input: {str(e)}")


# --- LOAD REMEDY DICTIONARY FROM JSON ---
try:
    with open(os.path.join("data", "treatment.json"), "r", encoding="utf-8") as f:
        remedy_treatment_dict = json.load(f)
except Exception as e:
    raise HTTPException(
        status_code=500, detail=f"Failed to load treatment data: {str(e)}"
    )


# --- PREDICTION API ---
@app.post("/predict/")
async def predict(image: UploadFile = File(...), description: str = Form(...)):
    try:
        # Extract features
        features = preprocess(image.file, description)

        # Predict
        probs = model.predict(features)[0]
        pred_idx = np.argmax(probs)
        label = label_encoder.inverse_transform([pred_idx])[0]
        confidence = probs[pred_idx] * 100

        # Remedy & Treatment
        remedy = remedy_treatment_dict.get(label, {}).get(
            "remedy", "No remedy available"
        )
        treatment = remedy_treatment_dict.get(label, {}).get(
            "treatment", "No treatment available"
        )

        return {
            "prediction": label,
            "confidence": f"{confidence:.2f}%",
            "remedy": remedy,
            "treatment": treatment,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
