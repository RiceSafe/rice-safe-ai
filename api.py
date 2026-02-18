import io
import os
import traceback
import joblib
import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications import MobileNetV2
from transformers import AutoModel, AutoTokenizer, CLIPProcessor, CLIPModel

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env.api")
if os.path.exists(dotenv_path):
    print(f"[*] Loading environment variables from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)

os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "false")

# Initialize FastAPI application
app = FastAPI(title="RiceSafe Disease Prediction API - Local Version")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Computation Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"[INFO] API using device: {DEVICE}")

# --- CONFIGURATION: Model paths and CLIP settings ---
BASE_MODEL_PATH = os.path.join(os.path.dirname(__file__))
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_THRESHOLD = float(os.environ.get("CLIP_THRESHOLD", 0.3))

CLIP_LABELS = [
    # positive
    "a photo of a rice leaf or rice plant",
    "a photo of a person or selfie",
    "a photo of an animal",
    "a photo of a vehicle or building",
    "a photo of food or a meal",
    "a screenshot of a phone or computer screen",
    "a photo of grasses",
    # catch-all negative
    "not a photo of a rice leaf",
]
CLIP_POSITIVE_INDEX = 0

# Mapping: Thai label -> English keyword
PREDICTION_MAP = {
    "ปกติ": "normal",
    "โรคไหม้": "rice_blast",
    "โรคใบจุดสีน้ำตาล": "brown_spot",
    "โรคขอบใบแห้ง": "bacterial_leaf_blight",
    "โรคใบขีดโปร่งแสง": "bacterial_leaf_streak",
}

# Disease confidence threshold — predictions below this are returned as "not_clear"
DISEASE_CONFIDENCE_THRESHOLD = float(os.environ.get("DISEASE_CONFIDENCE_THRESHOLD", 0.80))

# Global variables
keras_model = None
label_encoder = None
scaler = None
text_tokenizer = None
text_model_embedder = None
mobilenet_extractor = None
clip_model = None
clip_processor = None


# Load models and preprocessors on startup
@app.on_event("startup")
async def load_assets_on_startup():
    global keras_model, label_encoder, scaler, text_tokenizer, text_model_embedder
    global mobilenet_extractor, clip_model, clip_processor

    print(f"[INFO] API Startup: Loading local models from {BASE_MODEL_PATH}...")

    try:
        # 1. Load Keras classification model
        keras_path = os.path.join(BASE_MODEL_PATH, "RiceSafeModel.h5")
        print(f"[*] Loading Keras model: {keras_path}")
        keras_model = load_model(keras_path)

        # 2. Load LabelEncoder
        le_path = os.path.join(BASE_MODEL_PATH, "label_encoder.pkl")
        print(f"[*] Loading LabelEncoder: {le_path}")
        label_encoder = joblib.load(le_path)

        # 3. Load feature scaler
        scaler_path = os.path.join(BASE_MODEL_PATH, "scaler.pkl")
        print(f"[*] Loading Scaler: {scaler_path}")
        scaler = joblib.load(scaler_path)

        # 4. Load text and image feature extractors
        print("[*] Initializing pre-trained helper models (BAAI/bge-m3 & MobileNetV2)...")
        text_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        text_model_embedder = AutoModel.from_pretrained("BAAI/bge-m3").eval().to(DEVICE)

        mobilenet_extractor = MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(224, 224, 3),
        )
        mobilenet_extractor.trainable = False

        # 5. Initialize CLIP Gatekeeper
        print(f"[*] Loading CLIP gatekeeper model: {CLIP_MODEL_NAME}...")
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).eval().to(DEVICE)
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        print(f"[DONE] CLIP gatekeeper loaded (threshold: {CLIP_THRESHOLD})")

        print("[DONE] All local assets loaded successfully.")

    except Exception as e:
        print(f"[FATAL ERROR] Failed to load local assets: {e}")
        traceback.print_exc()
        raise RuntimeError(f"API Startup Failed: {e}")


# --- CLIP Gatekeeper ---
def check_rice_leaf_with_clip(pil_image: Image.Image) -> dict:
    """
    Verify whether the uploaded image is a rice leaf using CLIP zero-shot classification.
    Passes only if 'rice leaf' is the top-scoring label AND its score >= threshold.
    """
    inputs = clip_processor(
        text=CLIP_LABELS,
        images=pil_image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) if hasattr(v, 'to') else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    rice_leaf_prob = float(probs[CLIP_POSITIVE_INDEX])
    best_idx = int(np.argmax(probs))
    best_label = CLIP_LABELS[best_idx]

    # Build score map for debugging
    all_scores = {label: f"{prob * 100:.1f}%" for label, prob in zip(CLIP_LABELS, probs)}

    # Pass only if rice leaf is top-scoring AND above threshold
    is_rice = (best_idx == CLIP_POSITIVE_INDEX) and (rice_leaf_prob >= CLIP_THRESHOLD)

    return {
        "is_rice_leaf": is_rice,
        "confidence": rice_leaf_prob,
        "matched_label": best_label,
        "all_scores": all_scores,
    }


# --- Preprocessing ---
def preprocess_input_for_model(image_bytes: bytes, description: str):
    if not all([keras_model, label_encoder, scaler, text_tokenizer, text_model_embedder, mobilenet_extractor]):
        raise HTTPException(status_code=503, detail="Models not fully loaded.")

    try:
        # 1. Image: Extract pooled features -> (1280,)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        img_array = keras_image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array_expanded)

        img_feat = mobilenet_extractor.predict(img_preprocessed, verbose=0)
        img_feat = img_feat.flatten()  # (1280,)

        # 2. Text: Mean pooling -> (1024,)
        inputs_text = text_tokenizer(
            description, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(DEVICE)

        with torch.no_grad():
            outputs = text_model_embedder(**inputs_text)
            text_feat = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze(0)  # (1024,)

        # 3. Concatenate + Scale -> (1, 2304)
        combined = np.concatenate([img_feat, text_feat]).reshape(1, -1)
        combined_scaled = scaler.transform(combined)

        return combined_scaled

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {e}")


# --- Prediction Endpoint ---
@app.post("/predict/")
async def predict_endpoint(image: UploadFile = File(...), description: str = Form(...)):
    if not all([keras_model, label_encoder, scaler]):
        raise HTTPException(status_code=503, detail="Core models not ready.")

    try:
        image_bytes = await image.read()

        # Step 1: CLIP Gatekeeper — reject non-rice-leaf images
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        clip_result = check_rice_leaf_with_clip(pil_image)

        if not clip_result["is_rice_leaf"]:
            return {
                "prediction": "not_rice",
                "confidence": f"{clip_result['confidence'] * 100:.2f}%",
                "clip_check": {
                    "passed": False,
                    "rice_leaf_confidence": f"{clip_result['confidence'] * 100:.2f}%",
                    "matched_label": clip_result["matched_label"],
                    "all_scores": clip_result["all_scores"],
                    "threshold": f"{CLIP_THRESHOLD * 100:.0f}%",
                    "message": "กรุณาส่งรูปใบข้าวเพื่อวิเคราะห์โรค",
                },
                "class_probabilities": None,
            }

        # Step 2: Preprocess + Predict
        input_scaled = preprocess_input_for_model(image_bytes, description)

        probs_tensor = keras_model(input_scaled, training=False)
        probs = probs_tensor.numpy()[0]

        pred_idx = np.argmax(probs)
        label = label_encoder.inverse_transform([pred_idx])[0]
        confidence = probs[pred_idx] * 100

        class_probabilities = {}
        for i, class_name in enumerate(label_encoder.classes_):
            class_probabilities[class_name] = f"{probs[i] * 100:.2f}%"

        # Map Thai label to English keyword for mobile
        prediction_key = PREDICTION_MAP.get(label, label)

        # If confidence is below threshold, return "not_clear"
        if confidence < DISEASE_CONFIDENCE_THRESHOLD * 100:
            return {
                "prediction": "not_clear",
                "confidence": f"{confidence:.2f}%",
                "clip_check": {
                    "passed": True,
                    "rice_leaf_confidence": f"{clip_result['confidence'] * 100:.2f}%",
                    "matched_label": clip_result["matched_label"],
                },
                "class_probabilities": class_probabilities,
            }

        return {
            "prediction": prediction_key,
            "confidence": f"{confidence:.2f}%",
            "clip_check": {
                "passed": True,
                "rice_leaf_confidence": f"{clip_result['confidence'] * 100:.2f}%",
                "matched_label": clip_result["matched_label"],
            },
            "class_probabilities": class_probabilities,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# Health check
@app.get("/health")
async def health_check():
    models_ready = all([keras_model, label_encoder, scaler])
    clip_ready = all([clip_model, clip_processor])
    status = "healthy" if (models_ready and clip_ready) else "unhealthy"
    return {
        "status": status,
        "models_loaded": models_ready,
        "clip_gatekeeper_loaded": clip_ready,
    }