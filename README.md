---
title: Rice Safe AI
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
---

# RiceSafe AI

Multimodal rice leaf disease classification: images plus Thai symptom text. Training is driven by `training_pipeline_V3.py` (MobileNetV2, BGE-M3, three softmax heads). Inference is served by `api.py` (FastAPI) with a CLIP pre-check before the Keras model.

## Layout

| Path | Purpose |
|------|---------|
| `training_pipeline_V3.py` | V3 pipeline: `--mode` `pipeline` \| `extract` \| `train` \| `eval`. Writes model, label encoder, `features_v3_exp009_other_text/`, `reports/`, `graphs/` (paths set at top of file). |
| `api.py` | `POST /predict/`, `GET /health`. Default weights: `RiceSafeModel.keras`, `label_encoderV3.pkl` (see constants in file). |
| `training_pipeline_V2.py` | Older pipeline; expects `data_physical/`. |
| `train_ricesafe_multimodal.py` | Older MLflow run; expects `data/`. |
| `tests/` | Pytest; sample CSV under `tests/data/dummy/`. |
| `Dockerfile` | `uvicorn api:app` on port 7860. |
| `.env.api.example` | Template for `.env.api` (thresholds). |

# Current Classes
- Bacterial Leaf Blight (โรคขอบใบแห้ง)
- Brown Spot (โรคใบจุดสีน้ำตาล)
- Blast (โรคไหม้)
- Other (อื่นๆ = ปกติ + โรคใบขีดสีน้ำตาล + โรคใบสีส้ม + โรคกาบใบแห้ง + แมลงหนามตำข้าว)x


## Pipeline V3 flow

1. **Extract** — For each split (`train` / `val` / `test_clean` / `test_noisy`), load images under `data_physical_v8_final`, pair each image with symptom text (class CSV or `other_csv` routing), run **BAAI/bge-m3**, save `*_text_feats.npy`, `*_labels.npy`, `*_img_paths.npy` into `features_v3_exp009_other_text/`.
2. **Train** — `tf.data` loads image files plus those arrays; training applies modality dropout and augments images. unfreeze MobileNetV2, lower LR. Best weights go to the `.keras` file and `LabelEncoder` to `.pkl` (paths in script).
3. **Eval** — Load the saved model and feature files; run the built-in scenario set (clean / noisy text / image-only / degraded image variants), print metrics, write `reports/` and `graphs/`.

`--mode pipeline` runs extract, then train, then eval; other modes run a single stage.

## Training (V3)

```bash
# All in one
python training_pipeline_V3.py --mode pipeline

# Separate 
python training_pipeline_V3.py --mode extract 
python training_pipeline_V3.py --mode train  
python training_pipeline_V3.py --mode eval  
```

## API

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8080
```

## Tests

```bash
pytest tests/
```

## Install

```bash
python -m venv venv
# Windows: .\venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

- CUDA (Windows, NVIDIA):

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

- CPU / Apple Silicon:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install -r requirements.txt
```

## CI/CD & Automation

This repository uses **GitHub Actions** to automate the testing and deployment lifecycle:

- **Continuous Integration (CI):** Every push or Pull Request triggers an automated test suite using `pytest` to ensure model and API integrity.
- **Continuous Deployment (CD):** Upon a successful merge to the `main` branch, the repository is automatically synchronized with the production environment.
- **Live Space:** [huggingface.co/spaces/xNatthapol/rice-safe-ai](https://huggingface.co/spaces/xNatthapol/rice-safe-ai)