import io
import json
import os
import traceback

import mlflow
import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from transformers import AutoModel, AutoTokenizer

dotenv_path = os.path.join(os.path.dirname(__file__), ".env.api")
if os.path.exists(dotenv_path):
    print(f"[*] Loading environment variables from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
else:
    print(
        f"[INFO] .env.api file not found at {dotenv_path}. Using default hyperparameter values or system environment variables if set."
    )

os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "false")


# Initialize FastAPI application
app = FastAPI(title="RiceSafe Disease Prediction API")

# Configure CORS (Cross-Origin Resource Sharing) to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the computation device (CPU or GPU) for PyTorch models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] API using device: {DEVICE}")

# Get MLFLOW_RUN_ID from environment variables (set in .env file)
MLFLOW_RUN_ID = os.environ.get("MLFLOW_RUN_ID")

# Validate that MLFLOW_RUN_ID is provided
if not MLFLOW_RUN_ID:
    print("[FATAL ERROR] API cannot start: MLFLOW_RUN_ID must be set in .env file.")
    raise RuntimeError("API Startup Error: MLFLOW_RUN_ID not configured.")

# Global variables to hold loaded models and preprocessors
keras_model = None
label_encoder = None
scaler = None
text_tokenizer = None  # For BAAI/bge-m3 text embeddings
text_model_embedder = None  # For BAAI/bge-m3 text embeddings
mobilenet_extractor = None  # For image feature extraction


# Load all necessary models and preprocessors when the API starts up
@app.on_event("startup")
async def load_assets_on_startup():
    global keras_model, label_encoder, scaler, text_tokenizer, text_model_embedder, mobilenet_extractor

    print("[INFO] API Startup: Loading models and preprocessors...")
    if not MLFLOW_RUN_ID:
        print("[ERROR] MLFLOW_RUN_ID not defined within startup. Models cannot load.")
        return

    try:
        # Construct URIs for MLflow artifacts based on the Run ID
        keras_model_uri = f"runs:/{MLFLOW_RUN_ID}/model"
        label_encoder_uri = f"runs:/{MLFLOW_RUN_ID}/label_encoder"
        scaler_uri = f"runs:/{MLFLOW_RUN_ID}/feature_scaler"

        print(f"[*] Loading Keras model from: {keras_model_uri}")
        keras_model = mlflow.keras.load_model(keras_model_uri)
        print(f"[*] Keras model loaded.")

        print(f"[*] Loading LabelEncoder from: {label_encoder_uri}")
        label_encoder = mlflow.sklearn.load_model(label_encoder_uri)
        print(f"[*] LabelEncoder loaded.")

        print(f"[*] Loading StandardScaler from: {scaler_uri}")
        scaler = mlflow.sklearn.load_model(scaler_uri)
        print(f"[*] StandardScaler loaded.")

        # Initialize pre-trained models
        print("[*] Initializing pre-trained helper models ...")
        text_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        text_model_embedder = AutoModel.from_pretrained("BAAI/bge-m3").eval().to(DEVICE)
        mobilenet_extractor = MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(224, 224, 3),
        )
        mobilenet_extractor.trainable = False
        print("[DONE] All models, preprocessors, and extractors loaded successfully.")

    except Exception as e_load:
        print(f"[FATAL ERROR] During API startup model/preprocessor loading: {e_load}")
        traceback.print_exc()
        # This error will prevent the API from starting correctly.
        raise RuntimeError(f"API Startup Failed: Asset loading error - {e_load}")


# Preprocess incoming image and text data to create features for the model
def preprocess_input_for_model(image_bytes: bytes, description: str):
    # Check all necessary components are loaded before attempting to preprocess
    if not all(
        [
            keras_model,
            label_encoder,
            scaler,
            text_tokenizer,
            text_model_embedder,
            mobilenet_extractor,
        ]
    ):
        raise HTTPException(
            status_code=503,
            detail="Models not fully loaded or API not ready. Please try again shortly.",
        )

    try:
        # Image feature extraction
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        img_array = keras_image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_array_preprocessed = preprocess_input(img_array_expanded)
        img_feat = mobilenet_extractor.predict(img_array_preprocessed)

        # Text feature extraction
        inputs_text = text_tokenizer(
            description,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(DEVICE)
        with torch.no_grad():
            text_feat = (
                text_model_embedder(**inputs_text)
                .last_hidden_state.mean(dim=1)
                .cpu()
                .numpy()
            )

        # Combine and scale features
        combined_feat_flat = np.concatenate((img_feat.flatten(), text_feat.flatten()))
        combined_feat_reshaped = combined_feat_flat.reshape(1, -1)
        scaled_feat = scaler.transform(combined_feat_reshaped)
        return scaled_feat
    except Exception as e_process:
        print(f"[ERROR] Preprocessing input: {e_process}")
        traceback.print_exc()
        raise HTTPException(
            status_code=400, detail=f"Error processing input data: {e_process}"
        )


# Load remedy and treatment information from a JSON file
remedy_treatment_dict = {}
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Attempt to construct path robustly, then fallback for simpler environments
    treatment_file_path = os.path.join(current_script_dir, "data", "treatment.json")
    if not os.path.exists(treatment_file_path):
        treatment_file_path = os.path.join("data", "treatment.json")

    with open(treatment_file_path, "r", encoding="utf-8") as f:
        remedy_treatment_dict = json.load(f)
    print(f"[*] Treatment data loaded successfully from {treatment_file_path}")
except Exception as e_remedy:
    print(
        f"[WARNING] Failed to load treatment data from {treatment_file_path}: {e_remedy}. Remedy/treatment info will be unavailable."
    )


# API endpoint for making predictions
@app.post("/predict/")
async def predict_endpoint(image: UploadFile = File(...), description: str = Form(...)):
    # Check if core predictive models are loaded
    if not all([keras_model, label_encoder, scaler]):
        raise HTTPException(
            status_code=503,
            detail="Core models not loaded or API not ready. Please check server logs.",
        )

    try:
        image_bytes = await image.read()
        features = preprocess_input_for_model(image_bytes, description)

        # Get model predictions (probabilities)
        probs = keras_model.predict(features)[0]
        pred_idx = np.argmax(probs)

        # Convert predicted index back to original string label
        label_input_for_inverse = (
            [pred_idx] if np.isscalar(pred_idx) else pred_idx.flatten()
        )
        label = label_encoder.inverse_transform(label_input_for_inverse)[0]

        confidence = probs[pred_idx] * 100

        # Retrieve remedy and treatment information
        remedy = remedy_treatment_dict.get(label, {}).get(
            "remedy", "ข้อมูลการรักษาไม่พร้อมใช้งาน"
        )
        treatment = remedy_treatment_dict.get(label, {}).get(
            "treatment", "ข้อมูลการรักษาไม่พร้อมใช้งาน"
        )

        # Prepare class probabilities for the API response
        class_probabilities = {}
        if hasattr(label_encoder, "classes_"):
            for i, class_name_original in enumerate(label_encoder.classes_):
                class_probabilities[class_name_original] = f"{probs[i]*100:.2f}%"
        else:
            class_probabilities = {
                "error": "Class names not available from label encoder."
            }

        return {
            "prediction": label,
            "confidence": f"{confidence:.2f}%",
            "remedy": remedy,
            "treatment": treatment,
            "class_probabilities": class_probabilities,
        }
    except HTTPException as e_http:
        raise e_http
    except Exception as e_predict:
        print(f"[ERROR] During prediction processing: {e_predict}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed due to an internal error: {e_predict}",
        )


# API endpoint for health check
@app.get("/health")
async def health_check():
    # Check if all critical components are loaded
    models_loaded_flags = {
        "Keras_Model": keras_model is not None,
        "Label_Encoder": label_encoder is not None,
        "Scaler": scaler is not None,
        "Text_Tokenizer": text_tokenizer is not None,
        "Text_Embedder": text_model_embedder is not None,
        "Image_Extractor": mobilenet_extractor is not None,
    }
    if all(models_loaded_flags.values()):
        return {
            "status": "healthy",
            "message": "All models and preprocessors loaded successfully.",
        }
    else:
        missing_components = [
            name for name, loaded in models_loaded_flags.items() if not loaded
        ]
        return {
            "status": "unhealthy",
            "message": f"One or more components failed to load: {', '.join(missing_components)}. Check server logs.",
        }
