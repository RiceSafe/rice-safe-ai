---
title: Rice Safe AI
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
---

# RiceSafe Multimodal AI Model for Disease Classification

This project implements a multimodal model to classify common rice plant diseases using both image and textual symptom data. It leverages MLflow for experiment tracking and includes a FastAPI application for model serving.

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
  - [Image Data](#image-data)
  - [Text Data](#text-data)
  - [Classes](#classes)
  - [Dataset References](#dataset-references)
- [Installation & Setup (AI Service)](#installation--setup-ai-service)
  - [1. Create Virtual Environment](#1-create-virtual-environment)
  - [2. Activate Virtual Environment](#2-activate-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Run the API](#4-run-the-api)

## Features

- **Multimodal Learning:** Combines image features (from MobileNetV2) and text embeddings (from BAAI/bge-m3) for classification.
- **Deep Learning Model:** Uses a Keras Sequential model with Dense, Batch Normalization, and Dropout layers.
- **Experiment Tracking:** Integrated with MLflow to log parameters, metrics, artifacts (models, plots, data), and source code for each training run.
- **Robust Training:** Implements callbacks like EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.
- **API for Serving:** Includes a FastAPI application to serve the trained model for predictions.
- **Thai Language Support:** Handles Thai class names and provides Thai font support for visualizations.

## Dataset

This project utilizes a curated dataset combining image and text data to classify **three common rice plant diseases plus other mixed classes in a single class ( Other )**, totaling five distinct classes.

_Please note: The dataset details below describe the initial setup used for this proof-of-concept (POC). The dataset is expected to evolve and be refined in future development._

### Image Data

## TBA

### Text Data

## TBA

### Classes

1. Bacterial Leaf Blight (โรคขอบใบแห้ง)
2. Brown Spot (โรคใบจุดสีน้ำตาล)
3. Blast (โรคไหม้)
4. Other (อื่นๆ = ปกติ + โรคใบขีดสีน้ำตาล + โรคใบสีส้ม + โรคกาบใบแห้ง + แมลงหนามตำข้าว)

### Dataset References

## TBA

## Installation & Setup (AI Service)

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Virtual Environment
- **Windows:** `.\venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 3. Install Dependencies
Due to PyTorch hardware specifics, it is recommended to install PyTorch based on your OS first to avoid CVE-2025-32434 issues with `transformers` loading CLIP models.

**For Windows (with NVIDIA GPU / CUDA 11.8):**
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**For Mac/Linux (CPU or Apple Silicon):**
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install -r requirements.txt
```

### 4. Run the API
```bash
uvicorn api_exp009_other:app --reload --port 8080
```
