# RiceSafe Multimodal AI Model

## Sample Dataset

This project uses a curated dataset that combines both image and text data to classify five types of rice diseases.

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
- Truman Rase. *Rice Leaf Diseases*. Kaggle. https://www.kaggle.com/datasets/trumanrase/rice-leaf-diseases  
- Lourdu Antony, Leo Prasanth (2023). Rice Leaf Diseases Dataset, Mendeley Data, V1. https://doi.org/10.17632/dwtn3c6w6p.1

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
