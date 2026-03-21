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
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel

dotenv_path = os.path.join(os.path.dirname(__file__), ".env.api")
if os.path.exists(dotenv_path):
    print(f"[*] Loading environment variables from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)

os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "false")

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("[INFO] TensorFlow GPU memory growth enabled.")
    except Exception as e:
        print(f"[WARNING] Could not set memory growth: {e}")

app = FastAPI(title="RiceSafe Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"[INFO] API using device: {DEVICE}")

BASE_MODEL_PATH = os.path.join(os.path.dirname(__file__))
KERAS_MODEL_NAME = "RiceSafeModel.keras"
LABEL_ENCODER_NAME = "label_encoderV3.pkl"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_THRESHOLD  = float(os.environ.get("CLIP_THRESHOLD", 0.3))
CLIP_LABELS = [
    "a photo of a rice leaf or rice plant",
    "a photo of a person or selfie",
    "a photo of an animal",
    "a photo of a vehicle or building",
    "a photo of food or a meal",
    "a screenshot of a phone or computer screen",
]
CLIP_POSITIVE_INDEX = 0

PREDICTION_MAP = {
    "ปกติ": "normal",
    "อื่นๆ": "other_diseases",
    "โรคไหม้": "rice_blast",
    "โรคใบจุดสีน้ำตาล": "brown_spot",
    "โรคขอบใบแห้ง": "bacterial_leaf_blight",
}

DISEASE_CONFIDENCE_THRESHOLD = float(os.environ.get("DISEASE_CONFIDENCE_THRESHOLD", 0.80))

ENSEMBLE_W_MAIN = 0.60
ENSEMBLE_W_IMG  = 0.20
ENSEMBLE_W_TXT  = 0.20

keras_model    = None
label_encoder  = None
clip_model     = None
clip_processor = None
bge_tokenizer  = None
bge_model      = None

@app.on_event("startup")
async def load_assets_on_startup():
    global keras_model, label_encoder
    global clip_model, clip_processor
    global bge_tokenizer, bge_model

    print(f"[INFO] API Startup: Loading assets from {BASE_MODEL_PATH}...")

    try:
        keras_path = os.path.join(BASE_MODEL_PATH, KERAS_MODEL_NAME)
        print(f"[*] Loading Keras model: {keras_path}")
        keras_model = load_model(keras_path)

        le_path = os.path.join(BASE_MODEL_PATH, LABEL_ENCODER_NAME)
        print(f"[*] Loading LabelEncoder: {le_path}")
        label_encoder = joblib.load(le_path)

        print(f"[*] Loading BGE-M3 text encoder on {DEVICE}...")
        bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        bge_model = AutoModel.from_pretrained("BAAI/bge-m3", use_safetensors=True).to(DEVICE).eval()

        print(f"[*] Loading CLIP gatekeeper: {CLIP_MODEL_NAME}...")
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).eval().to(DEVICE)
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        print(f"[*] CLIP loaded (threshold: {CLIP_THRESHOLD})")

        print("[DONE] All assets loaded successfully.")

    except Exception as e:
        print(f"[FATAL ERROR] Failed to load assets: {e}")
        traceback.print_exc()
        raise RuntimeError(f"API Startup Failed: {e}")

def check_rice_leaf_with_clip(pil_image: Image.Image) -> dict:
    inputs = clip_processor(
        text=CLIP_LABELS, images=pil_image,
        return_tensors="pt", padding=True,
    )
    inputs = {k: v.to(DEVICE) if hasattr(v, 'to') else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

    rice_leaf_prob = float(probs[CLIP_POSITIVE_INDEX])
    best_idx = int(np.argmax(probs))
    all_scores = {label: f"{prob * 100:.1f}%" for label, prob in zip(CLIP_LABELS, probs)}
    is_rice = (best_idx == CLIP_POSITIVE_INDEX) and (rice_leaf_prob >= CLIP_THRESHOLD)

    return {
        "is_rice_leaf": is_rice,
        "confidence": rice_leaf_prob,
        "matched_label": CLIP_LABELS[best_idx],
        "all_scores": all_scores,
    }

def preprocess_input_for_model(pil_image: Image.Image, description: str):
    if not all([keras_model, bge_tokenizer, bge_model]):
        raise HTTPException(status_code=503, detail="Models not fully loaded.")

    try:
        img = pil_image.resize((224, 224))
        img_array = keras_image.img_to_array(img)
        img_preprocessed = (np.expand_dims(img_array, axis=0) / 127.5) - 1.0

        if not description or not description.strip():
            description = "ตรวจสอบสภาพใบข้าว"

        with torch.no_grad():
            inputs = bge_tokenizer(description, return_tensors="pt", truncation=True,
                                   padding=True, max_length=512).to(DEVICE)
            outputs = bge_model(**inputs)
            text_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        return {
            "image_input": img_preprocessed.astype(np.float32),
            "text_input":  text_embedding.astype(np.float32),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {e}")

@app.post("/predict/")
def predict_endpoint(image: UploadFile = File(...), description: str = Form(...)):
    if not all([keras_model, label_encoder]):
        raise HTTPException(status_code=503, detail="Core models not ready.")

    try:
        image_bytes = image.file.read()
        pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")

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

        inputs_dict = preprocess_input_for_model(pil_image, description)
        
        preds = keras_model.predict(inputs_dict)
        p_main = preds[0][0]
        p_img  = preds[1][0]
        p_txt  = preds[2][0]

        w_sum = ENSEMBLE_W_MAIN + ENSEMBLE_W_IMG + ENSEMBLE_W_TXT
        w_main = ENSEMBLE_W_MAIN / w_sum
        w_img  = ENSEMBLE_W_IMG / w_sum
        w_txt  = ENSEMBLE_W_TXT / w_sum

        ens_probs = (w_main * p_main) + (w_img * p_img) + (w_txt * p_txt)

        pred_idx   = int(np.argmax(ens_probs))
        label      = label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(ens_probs[pred_idx] * 100)

        class_probabilities = {
            cls: f"{ens_probs[i] * 100:.2f}%"
            for i, cls in enumerate(label_encoder.classes_)
        }

        prediction_key = PREDICTION_MAP.get(label, label)

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

@app.get("/health")
async def health_check():
    models_ready = all([keras_model, label_encoder, bge_model])
    clip_ready   = all([clip_model, clip_processor])
    return {
        "status": "healthy" if (models_ready and clip_ready) else "unhealthy",
        "model": KERAS_MODEL_NAME,
        "text_encoder": "BGE-M3",
        "models_loaded": models_ready,
        "clip_gatekeeper_loaded": clip_ready,
    }