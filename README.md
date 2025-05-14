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
