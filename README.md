# RiceSafe Multimodal AI Model for Disease Classification

This project implements a multimodal model to classify common rice plant diseases using both image and textual symptom data. It leverages MLflow for experiment tracking and includes a FastAPI application for model serving.

## Table of Contents

- [Features](#features)
- [Sample Dataset](#sample-dataset)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Cloning the Repository](#cloning-the-repository)
  - [Setting up the Environment](#setting-up-the-environment)
- [Execution Instructions](#execution-instructions)
  - [Training the Model](#training-the-model)
  - [Monitoring with MLflow UI](#monitoring-with-mlflow-ui)
  - [Running the API](#running-the-api)
  - [Making Predictions via API](#making-predictions-via-api)
- [Model Versioning and Experimentation](#model-versioning-and-experimentation)

## Features

- **Multimodal Learning:** Combines image features (from MobileNetV2) and text embeddings (from BAAI/bge-m3) for classification.
- **Deep Learning Model:** Uses a Keras Sequential model with Dense, Batch Normalization, and Dropout layers.
- **Experiment Tracking:** Integrated with MLflow to log parameters, metrics, artifacts (models, plots, data), and source code for each training run.
- **Robust Training:** Implements callbacks like EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.
- **API for Serving:** Includes a FastAPI application to serve the trained model for predictions.
- **Thai Language Support:** Handles Thai class names and provides Thai font support for visualizations.

## Sample Dataset

This project utilizes a curated dataset combining image and text data to classify **four common rice plant diseases plus a healthy (normal) state**, totaling five distinct classes.

_Please note: The dataset details below describe the initial setup used for this proof-of-concept (POC). The dataset is expected to evolve and be refined in future development._

### Sample Image Data

- **Total samples**: 1,250 images
- **Classes**: 250 images per class across 5 disease types
- **Source**:
  - [Rice Leaf Diseases (Kaggle)](https://www.kaggle.com/datasets/trumanrase/rice-leaf-diseases)
  - [Rice Leaf Disease Dataset (Mendeley Data)](https://data.mendeley.com/datasets/dwtn3c6w6p/1)
- Only a subset of images was selected from these datasets

### Sample Text Data

- **Total samples**: 1,250 symptom descriptions
- **Creation**:
  - 100 manually written symptom descriptions per class
  - 150 additional samples per class generated through oversampling techniques
- Each sample is labeled to match one of the five classes.

### Classes

1. Bacterial Leaf Blight (โรคขอบใบแห้ง)
2. Brown Spot (โรคใบจุดสีน้ำตาล)
3. Bacterial Leaf Streak (โรคใบขีดโปร่งแสง)
4. Blast (โรคไหม้)
5. Healthy (ปกติ)

### Dataset References

- Truman Rase. _Rice Leaf Diseases_. Kaggle. https://www.kaggle.com/datasets/trumanrase/rice-leaf-diseases
- Lourdu Antony, Leo Prasanth (2023). Rice Leaf Diseases Dataset, Mendeley Data, V1. https://doi.org/10.17632/dwtn3c6w6p.1

## Project Structure

```
├── .env.api.example          # Example environment variables for the API
├── .env.train.example        # Example environment variables for training
├── .gitignore                # Specifies intentionally untracked files
├── api.py                    # FastAPI application for serving the model
├── data/
│   ├── train_image/          # Training images (organized by class subfolders)
│   │   ├── ปกติ/
│   │   ├── โรคขอบใบแห้ง/
│   │   └── ... (other classes)
│   ├── train_text/           # Training text data (CSV files per class)
│   │   ├── ปกติ.csv
│   │   ├── โรคขอบใบแห้ง.csv
│   │   └── ... (other classes)
│   ├── test_image/           # Test images (organized by class subfolders)
│   ├── test_text/            # Test text data (CSV files per class)
│   └── treatment.json        # Remedy and treatment information for diseases
├── eda.ipynb                 # Jupyter Notebook for Exploratory Data Analysis
├── fonts/
│   └── NotoSansThai-VariableFont_wdth,wght.ttf # Thai font file
├── mlruns/                   # MLflow tracking data
├── models.ipynb              # Jupyter Notebook for model development
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── train_ricesafe_multimodal.py # Main script for training the model with MLflow
```

## Setup Instructions

### Prerequisites

- Python 3.10
- pip (Python package installer)
- Git (for cloning the repository)
- A virtual environment manager (e.g., `venv`, `conda`)

### Cloning the Repository

```bash
git clone https://github.com/RiceSafe/rice-safe-ai.git
cd rice-safe-ai
```

### Setting up the Environment

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Execution Instructions

### Training the Model

1.  **Configure Training Environment:**
    - Copy `.env.train.example` to `.env.train`.
    - Modify hyperparameters in `.env.train` as needed for your experiment. The script will use these values.
2.  **Run the training script:**
    ```bash
    python train_ricesafe_multimodal.py
    ```
    - This script will:
      - Load data.
      - Preprocess image and text features.
      - Train the multimodal model.
      - Log parameters, metrics, and artifacts (model, plots, scaler, encoder, source code) to MLflow in the `mlruns/` directory.
      - Save some artifacts locally to `local_run_artifacts/` (this directory is cleaned up after the run by default).

### Monitoring with MLflow UI

1.  After (or during) training, open a new terminal in the project root directory.
2.  Start the MLflow UI:
    ```bash
    mlflow ui --port 5001 # You can use any available port
    ```
3.  Open your web browser and navigate to `http://localhost:5001`.
4.  You will see your experiments (e.g., "RiceSafe_MultiModal_V1") and can inspect individual runs, compare them, and view logged artifacts.

### Running the API

1.  **Configure API Environment:**
    - Copy `.env.api.example` to `.env.api`.
    - Edit `.env.api` and set `MLFLOW_RUN_ID` to the ID of a successfully trained model run that you want the API to serve. You can find the Run ID from the training script's console output or the MLflow UI.
2.  **Start the FastAPI application using Uvicorn:**
    ```bash
    uvicorn api:app --reload
    ```
    - The `--reload` flag is useful for development as it restarts the server on code changes.
    - The API will load the specified MLflow model and preprocessors on startup. Check the console for loading messages.

### Making Predictions via API

- The API exposes a `/predict/` endpoint that accepts a POST request with an image file and a text description.
- You can use tools like `curl`, Postman, or a simple Python script to send requests.
- The API will return a JSON response with the predicted disease class, confidence, remedy, and treatment information.
- You can also access the auto-generated API documentation (Swagger UI) at `http://localhost:8000/docs`.

## Model Versioning and Experimentation

To compare different versions of your model as required:

1.  **Define Variations:** For each "version" you want to test:
    - **Hyperparameters:** Modify values in your `.env.train` file (e.g., `EXP_INITIAL_LEARNING_RATE`, `EXP_BATCH_SIZE`). You can also add an `EXPERIMENT_VERSION_TAG` like "Version1", "Version2" to `.env.train` to explicitly tag runs.
    - **Architecture/Preprocessing:** Make necessary code changes directly in `train_ricesafe_multimodal.py` (e.g., change number of dense units, dropout rates, try a different scaler).
2.  **Train Each Version:** Run `python train_ricesafe_multimodal.py` after setting up each variation. Each execution will create a new run in MLflow under the same experiment (e.g., "RiceSafe_MultiModal_V1").
3.  **Compare in MLflow UI:**
    - Open the MLflow UI (`mlflow ui --port 5001`).
    - Navigate to your experiment.
    - Select the runs corresponding to your different versions.
    - Click the "Compare" button.
