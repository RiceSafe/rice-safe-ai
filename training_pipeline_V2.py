import sys
import subprocess
import os
import argparse
import time
import shutil

import joblib
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as tf_image

from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIGURATION ---
DATA_ROOT = 'data_physical'
FEATURES_DIR = 'features_stratified/'  # New directory for split features
SAVE_MODEL_PATH = 'RiceSafeModel.h5'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
SCALER_PATH = 'scaler.pkl'

DISEASE_CLASSES = ['ปกติ', 'โรคขอบใบแห้ง', 'โรคใบขีดโปร่งแสง', 'โรคใบจุดสีน้ำตาล', 'โรคไหม้']

class ModalityDropoutSequence(Sequence):
    """
    Dynamic Data Generator that applies Modality Dropout on the fly.
    """
    def __init__(self, x_set, y_set, batch_size=32, img_dim=1280, p_txt_drop=0.6, p_img_drop=0.15):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.img_dim = img_dim
        self.p_txt_drop = p_txt_drop # Probability to drop TEXT (keep image)
        self.p_img_drop = p_img_drop # Probability to drop IMAGE (keep text)
        self.indices = np.arange(len(self.x))

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds].copy() # Copy to avoid modifying original data
        batch_y = self.y[inds]
        
        # Vectorized dropout
        # r < p_txt_drop -> Drop Text (Zero out indices 1280:)
        # p_txt_drop <= r < p_txt_drop + p_img_drop -> Drop Image (Zero out indices :1280)
        # Else -> Full Modality
        
        r = np.random.rand(len(batch_x))
        
        # Drop Text Indices
        drop_text_mask = r < self.p_txt_drop
        batch_x[drop_text_mask, self.img_dim:] = 0
        
        # Drop Image Indices
        drop_img_mask = (r >= self.p_txt_drop) & (r < (self.p_txt_drop + self.p_img_drop))
        batch_x[drop_img_mask, :self.img_dim] = 0
        
        return batch_x, batch_y

def get_device():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def load_text_model():
    print(f"[{os.getpid()}] Loading Text Model (BGE-M3)...")
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    model = AutoModel.from_pretrained("BAAI/bge-m3").to(device).eval()
    return tokenizer, model, device

def load_image_model():
    print(f"[{os.getpid()}] Loading Image Model (MobileNetV2 - Avg Pool)...")
    mobilenet = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    mobilenet.trainable = False
    return mobilenet

@torch.no_grad()
def get_text_embedding(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze(0)

def get_image_feature(img_path, model):
    try:
        img = tf_image.load_img(img_path, target_size=(224, 224))
        img_array = tf_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        feats = model.predict(img_array, verbose=0)
        return feats.flatten()
    except Exception as e:
        print(f"[ERROR] Load image {img_path}: {e}")
        return None

def extract_set(split_name, tokenizer, text_model, device, img_model):
    """
    Extracts features for a specific split (train/val/test).
    Reads 'symptoms.csv' from each class folder.
    """
    print(f"\n[{os.getpid()}] Processing split: {split_name.upper()}...")
    base_dir = os.path.join(DATA_ROOT, split_name)
    features_list = []
    labels = []
    
    if not os.path.exists(base_dir):
        print(f"[ERROR] Split directory not found: {base_dir}")
        return None, None

    for cls_name in DISEASE_CLASSES:
        cls_dir = os.path.join(base_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
            
        # 1. Get Images
        image_files = sorted([
            os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        
        # 2. Get Text Pool
        csv_path = os.path.join(cls_dir, 'symptoms.csv')
        texts = []
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'symptoms' in df.columns:
                    texts = df['symptoms'].dropna().astype(str).tolist()
                elif not df.empty:
                    texts = df.iloc[:, 0].dropna().astype(str).tolist()
            except Exception as e:
                print(f"[WARN] Error reading CSV {csv_path}: {e}")
        
        if not texts:
            # Fallback text if CSV missing/empty
            texts = [f"Image of {cls_name}"] 
            
        print(f"   Class '{cls_name}': {len(image_files)} images, {len(texts)} texts pool")
        
        for i, img_path in enumerate(image_files):
            img_feat = get_image_feature(img_path, img_model)
            if img_feat is not None:
                # Random text sampling for training to increase variety? 
                # Or deterministic cycling?
                # For training: Random is better for variety.
                # For val/test: Deterministic (cycling) is better for consistency.
                if split_name == 'train':
                    text_str = np.random.choice(texts)
                else:
                    text_str = texts[i % len(texts)]
                    
                txt_feat = get_text_embedding(text_str, tokenizer, text_model, device)
                combined = np.concatenate([img_feat.flatten(), txt_feat.flatten()])
                features_list.append(combined)
                labels.append(cls_name)
                
    return np.array(features_list, dtype='float32'), np.array(labels)

def run_extraction():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    
    # Load Models once
    tok, txt_model, dev = load_text_model()
    img_model = load_image_model()
    
    for split in ['train', 'val', 'test']:
        X, y = extract_set(split, tok, txt_model, dev, img_model)
        if X is not None and len(X) > 0:
            np.save(os.path.join(FEATURES_DIR, f'{split}_features.npy'), X)
            np.save(os.path.join(FEATURES_DIR, f'{split}_labels.npy'), y)
            print(f"[{os.getpid()}] Saved {split}: {X.shape}")
        else:
            print(f"[WARN] No data found for {split}")

    print(f"[{os.getpid()}] Extraction Complete.")

def run_training():
    print(f"\n[{os.getpid()}] >>> STARTING TRAINING (PHYSICAL SPLIT)...")
    
    # 1. Load Data
    X_train = np.load(os.path.join(FEATURES_DIR, 'train_features.npy'))
    y_train_raw = np.load(os.path.join(FEATURES_DIR, 'train_labels.npy'), allow_pickle=True)
    
    X_val = np.load(os.path.join(FEATURES_DIR, 'val_features.npy'))
    y_val_raw = np.load(os.path.join(FEATURES_DIR, 'val_labels.npy'), allow_pickle=True)
    
    X_test = np.load(os.path.join(FEATURES_DIR, 'test_features.npy'))
    y_test_raw = np.load(os.path.join(FEATURES_DIR, 'test_labels.npy'), allow_pickle=True)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # 2. Encode Labels (Fit on ALL classes to ensure consistency)
    le = LabelEncoder()
    le.fit(DISEASE_CLASSES) 
    # Use fixed classes ensures mapped correctly even if train set misses a class (unlikely)
    
    y_train = le.transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)
    num_classes = len(le.classes_)
    
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"Label Encoder classes: {le.classes_}")
    
    # 3. Scale Features (Fit on TRAIN only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val) # Transform val/test using train stats
    X_test = scaler.transform(X_test)
    
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved.")
    
    # 4. Oversampling (Train Only)
    print(f"[{os.getpid()}] Performing Random Oversampling on Training Data...")
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    max_count = np.max(class_counts)
    
    X_train_bal, y_train_bal = [], []
    for cls in unique_classes:
        cls_indices = np.where(y_train == cls)[0]
        # Sample to match max_count
        choices = np.random.choice(cls_indices, size=max_count, replace=True)
        X_train_bal.append(X_train[choices])
        y_train_bal.append(y_train[choices])
        
    X_train = np.concatenate(X_train_bal)
    y_train = np.concatenate(y_train_bal)
    
    # Shuffle
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]
    print(f"Balanced Train Size: {X_train.shape}")
    
    # 5. Class Weights (Redundant due to Oversampling, but kept as None for clarity)
    class_weight_dict = None
    
    # 6. Modality Dropout Generator (Dynamic)
    print(f"[{os.getpid()}] Initializing Modality Dropout Generator...")
    print(f"   - Text Dropout Rate: 60% (Forces Image Learning)")
    print(f"   - Image Dropout Rate: 15% (Robustness)")
    
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    train_gen = ModalityDropoutSequence(
        X_train, y_train_cat, 
        batch_size=32, 
        img_dim=1280, 
        p_txt_drop=0.6, # 60% Chance to lose text
        p_img_drop=0.15 # 15% Chance to lose image
    )
    
    # 7. Model
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=CategoricalFocalCrossentropy(label_smoothing=0.0),
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=SAVE_MODEL_PATH, save_best_only=True, monitor='val_accuracy', verbose=1)
    ]
    
    history = model.fit(
        train_gen,
        validation_data=(X_val, y_val_cat),
        epochs=200,
        # batch_size=32, # Handled by generator
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    model.save(SAVE_MODEL_PATH)
    
    # 8. Evaluation
    print("\n--- Evaluation on HELD-OUT TEST SET ---")
    loss, acc = model.evaluate(X_test, y_test_cat)
    print(f"Test Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # 9. Plotting
    os.makedirs('graphs', exist_ok=True)
    try:
        plt.rcParams['font.family'] = 'Thonburi' # Thai font support
    except:
        pass
        
    # History
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.savefig('graphs/training_history.png')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Test Set Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig('graphs/confusion_matrix.png')
    
    print("[*] Training and Evaluation Complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pipeline', 'extract', 'train'], default='pipeline')
    args = parser.parse_args()
    
    if args.mode in ['extract', 'pipeline']:
        if args.mode == 'pipeline' and os.path.exists(os.path.join(FEATURES_DIR, 'train_features.npy')):
            print("[*] Features found. Skipping extraction (use --mode extract to force).")
        else:
            run_extraction()
            
    if args.mode in ['train', 'pipeline']:
        run_training()

if __name__ == "__main__":
    main()
