import os

os.environ["KERAS_BACKEND"] = "torch"

import argparse
import json
import datetime

import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Concatenate, Add, Multiply,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_ROOT          = "data_physical_v8_final"
OTHER_CSV_ROOT     = "other_csv"
TEXT_FEATURES_DIR  = "features_v3_exp009_other_text/"
SAVE_MODEL_PATH    = "RiceSafeModel_V3_exp009_other.keras"
LABEL_ENCODER_PATH = "label_encoder_V3_exp009_other.pkl"

IMG_SIZE             = (224, 224)
IMG_DIM              = 1280
TEXT_DIM             = 1024
BATCH_SIZE           = 32
NUM_FINE_TUNE_LAYERS = 60

# Modality-dropout probs during training (textboost settings)
TEXT_ZERO_PROB = 0.10
IMG_ZERO_PROB  = 0.20

# Ensemble weights for evaluation
ENSEMBLE_W_MAIN = 0.60
ENSEMBLE_W_IMG  = 0.20
ENSEMBLE_W_TXT  = 0.20

DISEASE_CLASSES = ["อื่นๆ", "โรคขอบใบแห้ง", "โรคใบจุดสีน้ำตาล", "โรคไหม้"]
NUM_CLASSES     = len(DISEASE_CLASSES)

# Other-class: image filename prefix -> folder under other_csv/
OTHER_SUBCLASS_MAP = {
    "ใบขีดสีน้ำตาล":  "ใบขีดสีน้ำตาล",
    "โรคกาบใบแห้ง":   "โรคกาบใบแห้ง",
    "โรคใบสีส้ม":     "โรคใบสีส้ม",
    "แมลงดำหนามข้าว": "แมลงดำหนามข้าว",
}
OTHER_NORMAL_FOLDER = "ปกติ"  # images with no prefix belong here


# ─── TEXT UTILITIES ───────────────────────────────────────────────────────────

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_text_model():
    device = get_device()
    print(f"Loading BGE-M3 on {device}...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    model = AutoModel.from_pretrained("BAAI/bge-m3", use_safetensors=True).to(device).eval()
    return tokenizer, model, device


@torch.no_grad()
def embed_text(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=512).to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze(0)


def _read_csv_texts(csv_path: str) -> list:
    """Read symptoms column; try several encodings."""
    for enc in ("utf-8-sig", "utf-8", "cp874", "cp1252"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            col = df["symptoms"] if "symptoms" in df.columns else df.iloc[:, 0]
            texts = col.dropna().astype(str).tolist()
            return texts if texts else []
        except (UnicodeDecodeError, pd.errors.EmptyDataError):
            continue
    return []


def load_other_csv_texts(subclass_folder: str, split: str) -> list:
    """Load symptoms.csv under other_csv/{subclass}/{split}/<first subdir>/."""
    split_dir = os.path.join(OTHER_CSV_ROOT, subclass_folder, split)
    if not os.path.isdir(split_dir):
        return ["ตรวจสอบสภาพใบข้าว"]
    subdirs = sorted(d for d in os.listdir(split_dir)
                     if os.path.isdir(os.path.join(split_dir, d)))
    if not subdirs:
        return ["ตรวจสอบสภาพใบข้าว"]
    csv_path = os.path.join(split_dir, subdirs[0], "symptoms.csv")
    texts = _read_csv_texts(csv_path)
    return texts if texts else ["ตรวจสอบสภาพใบข้าว"]


def get_other_subclass_folder(img_filename: str) -> str:
    """Return the other_csv folder name based on filename prefix."""
    for prefix, folder in OTHER_SUBCLASS_MAP.items():
        if img_filename.startswith(prefix + "_"):
            return folder
    return OTHER_NORMAL_FOLDER  # default if no prefix match


def build_cross_class_text_pool(target_split: str) -> list:
    """All-class text pool for noise; other-class pulls every other_csv subfolder."""
    pool = []
    for cls in DISEASE_CLASSES:
        if cls == "อื่นๆ":
            for folder in list(OTHER_SUBCLASS_MAP.values()) + [OTHER_NORMAL_FOLDER]:
                pool.extend(load_other_csv_texts(folder, target_split))
        else:
            csv_path = os.path.join(DATA_ROOT, target_split, cls, "symptoms.csv")
            if os.path.exists(csv_path):
                pool.extend(_read_csv_texts(csv_path))
    return pool if pool else ["ตรวจสอบสภาพใบข้าว"]


# ─── TEXT FEATURE EXTRACTION ──────────────────────────────────────────────────

def extract_text_features():
    os.makedirs(TEXT_FEATURES_DIR, exist_ok=True)
    tok, txt_model, dev = load_text_model()

    # Per-split pools for 50% cross-class noise
    train_cross_pool = build_cross_class_text_pool("train")
    val_cross_pool   = build_cross_class_text_pool("val")
    test_cross_pool  = build_cross_class_text_pool("test")

    for split in ["train", "val", "test_clean", "test_noisy"]:
        feats, labels, img_paths_list = [], [], []
        base_split = "test" if split.startswith("test") else split
        base_dir   = os.path.join(DATA_ROOT, base_split)
        use_cross  = split == "test_noisy"

        if split == "train":         cross_pool = train_cross_pool
        elif split == "val":         cross_pool = val_cross_pool
        else:                        cross_pool = test_cross_pool

        print(f"\n[Text Extraction] {split.upper()} "
              f"{'(cross-class pool)' if use_cross else '(class-specific)'}")

        for cls in DISEASE_CLASSES:
            cls_dir = os.path.join(base_dir, cls)
            if not os.path.exists(cls_dir):
                print(f"  [WARN] Missing class dir: {cls_dir}")
                continue

            img_files = sorted([
                os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ])

            # Cache class texts once per class dir
            if cls != "อื่นๆ":
                csv_path = os.path.join(cls_dir, "symptoms.csv")
                texts = _read_csv_texts(csv_path) if os.path.exists(csv_path) else []
                texts = texts or ["ตรวจสอบสภาพใบข้าว"]
            else:
                # Cache other_csv subfolders once (avoid per-image IO)
                other_cache: dict = {}
                for folder in list(OTHER_SUBCLASS_MAP.values()) + [OTHER_NORMAL_FOLDER]:
                    other_cache[folder] = load_other_csv_texts(folder, base_split)
                texts = []  # unused; per-image pool from other_cache

            print(f"  {cls}: {len(img_files)} images"
                  + (f", {len(texts)} texts" if cls != 'อื่นๆ' else " (sub-class routing)"))

            for img_path in img_files:
                fname = os.path.basename(img_path)

                # Pick text pool for this image
                if cls == "อื่นๆ":
                    subfolder = get_other_subclass_folder(fname)
                    img_texts = other_cache.get(subfolder, other_cache[OTHER_NORMAL_FOLDER])
                else:
                    img_texts = texts

                if use_cross:
                    # test_noisy: all cross-pool
                    txt = np.random.choice(cross_pool)
                elif split == "test_clean":
                    # test_clean: class-specific only
                    txt = np.random.choice(img_texts)
                else:
                    # train/val: 50% cross-pool noise
                    if np.random.rand() < 0.50:
                        txt = np.random.choice(cross_pool)
                    else:
                        txt = np.random.choice(img_texts)

                emb = embed_text(txt, tok, txt_model, dev)
                feats.append(emb)
                labels.append(cls)
                img_paths_list.append(img_path)

        feats_arr  = np.array(feats, dtype="float32")
        labels_arr = np.array(labels)
        paths_arr  = np.array(img_paths_list)

        np.save(os.path.join(TEXT_FEATURES_DIR, f"{split}_text_feats.npy"), feats_arr)
        np.save(os.path.join(TEXT_FEATURES_DIR, f"{split}_labels.npy"),     labels_arr)
        np.save(os.path.join(TEXT_FEATURES_DIR, f"{split}_img_paths.npy"),  paths_arr)
        print(f"  Saved: text_feats={feats_arr.shape}, labels={labels_arr.shape}")

    print("\n[*] Text extraction complete.")


# ─── TF.DATA PIPELINE ────────────────────────────────────────────────────────

def build_tf_dataset(img_paths, text_feats, labels, batch_size=BATCH_SIZE,
                     is_training=False, apply_dropout=False, sample_weights=None):
    if sample_weights is not None:
        ds = tf.data.Dataset.from_tensor_slices(
            (img_paths, text_feats, labels, sample_weights))
    else:
        ds = tf.data.Dataset.from_tensor_slices((img_paths, text_feats, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=len(img_paths), seed=42)

    def process_data(img_path, text_feat, label):
        try:
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize_with_pad(img, IMG_SIZE[0], IMG_SIZE[1])
            img = tf.cast(img, tf.float32)
        except Exception as e:
            # Log path; zeros image so batch keeps going
            tf.print("!!! ERROR LOADING IMAGE:", img_path, "Error:", str(e))
            img = tf.zeros((*IMG_SIZE, 3), dtype=tf.float32)

        if is_training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, max_delta=0.2)

        if apply_dropout:
            rand_val = tf.random.uniform([])
            text_feat = tf.cond(
                rand_val < TEXT_ZERO_PROB,
                true_fn=lambda: tf.zeros_like(text_feat),
                false_fn=lambda: text_feat,
            )
            img = tf.cond(
                (rand_val >= TEXT_ZERO_PROB) & (rand_val < TEXT_ZERO_PROB + IMG_ZERO_PROB),
                true_fn=lambda: tf.zeros_like(img),
                false_fn=lambda: img,
            )

        img = (img / 127.5) - 1.0
        y = {"out_main": label, "out_img": label, "out_txt": label}
        return {"image_input": img, "text_input": text_feat}, y

    def process_data_with_weight(img_path, text_feat, label, sw):
        (x, y) = process_data(img_path, text_feat, label)
        return x, y, sw

    AUTOTUNE = tf.data.AUTOTUNE
    if sample_weights is not None:
        ds = ds.map(
            lambda ip, tf_, lb, sw: process_data_with_weight(ip, tf_, lb, sw),
            num_parallel_calls=AUTOTUNE,
        )
    else:
        ds = ds.map(process_data, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=is_training)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ─── MODEL ───────────────────────────────────────────────────────────────────

def build_multimodal_model(fine_tune=False):
    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    img_input = Input(shape=(*IMG_SIZE, 3), name="image_input")
    x_img = base(img_input, training=False)
    x_img = GlobalAveragePooling2D()(x_img)
    x_img = Dense(256, activation="relu")(x_img)
    x_img = BatchNormalization()(x_img)
    x_img = Dropout(0.4)(x_img)

    txt_input = Input(shape=(TEXT_DIM,), name="text_input")
    x_txt = Dense(256, activation="relu")(txt_input)
    x_txt = BatchNormalization()(x_txt)
    x_txt = Dropout(0.4)(x_txt)

    out_img = Dense(NUM_CLASSES, activation="softmax", name="out_img")(x_img)
    out_txt = Dense(NUM_CLASSES, activation="softmax", name="out_txt")(x_txt)

    merged   = Concatenate()([x_img, x_txt])
    gate_img = Dense(256, activation="sigmoid", name="gate_img")(merged)
    gate_txt = Dense(256, activation="sigmoid", name="gate_txt")(merged)
    gated_img = Multiply()([x_img, gate_img])
    gated_txt = Multiply()([x_txt, gate_txt])
    fused = Add()([gated_img, gated_txt])

    out_main = Dense(NUM_CLASSES, activation="softmax", name="out_main")(fused)

    model = Model(inputs=[img_input, txt_input],
                  outputs=[out_main, out_img, out_txt])

    if fine_tune:
        base.trainable = True
        for layer in base.layers[:-NUM_FINE_TUNE_LAYERS]:
            layer.trainable = False
        trainable = sum(1 for l in base.layers if l.trainable)
        print(f"Fine-tuning: {trainable}/{len(base.layers)} MobileNetV2 layers unfrozen")

    return model, base


# ─── EVALUATION ──────────────────────────────────────────────────────────────

def _run_evaluation(model, test_clean_paths, test_clean_txt,
                    test_noisy_txt, test_lbl, history1=None, history2=None):
    w_sum  = float(ENSEMBLE_W_MAIN + ENSEMBLE_W_IMG + ENSEMBLE_W_TXT)
    w_main = ENSEMBLE_W_MAIN / w_sum
    w_img  = ENSEMBLE_W_IMG  / w_sum
    w_txt  = ENSEMBLE_W_TXT  / w_sum

    print("\n--- FINAL EVALUATION (EXP-009 OTHER + ENSEMBLE HEADS) ---")
    print(f"Ensemble weights: w_main={w_main:.2f}, w_img={w_img:.2f}, w_txt={w_txt:.2f}")

    def predict_with_heads(gen_):
        pred_main, pred_img, pred_txt, pred_ens, true = [], [], [], [], []
        for x_dict, y_b in gen_:
            preds     = model.predict_on_batch(x_dict)
            p_main    = preds[0]
            p_img     = preds[1]
            p_txt     = preds[2]
            p_ens     = w_main * p_main + w_img * p_img + w_txt * p_txt
            pred_main.extend(np.argmax(p_main, axis=1))
            pred_img.extend(np.argmax(p_img,  axis=1))
            pred_txt.extend(np.argmax(p_txt,  axis=1))
            pred_ens.extend(np.argmax(p_ens,  axis=1))
            y_b_ = y_b["out_main"] if isinstance(y_b, dict) else y_b
            true.extend(y_b_.numpy())
        return (np.array(pred_main), np.array(pred_img),
                np.array(pred_txt),  np.array(pred_ens), np.array(true))

    def degrade(img_paths, text_feats, labels):
        ds = tf.data.Dataset.from_tensor_slices((img_paths, text_feats, labels))
        def _deg(ip, tf_, lb):
            try:
                img = tf.io.read_file(ip)
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize_with_pad(img, IMG_SIZE[0], IMG_SIZE[1])
                img = tf.cast(img, tf.float32)
            except Exception as e:
                tf.print("!!! ERROR LOADING DEGRADED IMAGE:", ip, "Error:", str(e))
                img = tf.zeros((*IMG_SIZE, 3), dtype=tf.float32)
                
            noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=40.0)
            img = tf.clip_by_value(img + noise, 0.0, 255.0)
            img = (img / 127.5) - 1.0
            y = {"out_main": lb, "out_img": lb, "out_txt": lb}
            return {"image_input": img, "text_input": tf_}, y
        return (ds.map(_deg, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    zero_txt = np.zeros_like(test_clean_txt)

    # A – clean multimodal
    gA = build_tf_dataset(test_clean_paths, test_clean_txt, test_lbl, BATCH_SIZE)
    mA, iA, tA, eA, y_true = predict_with_heads(gA)

    # B – noisy text
    gB = build_tf_dataset(test_clean_paths, test_noisy_txt, test_lbl, BATCH_SIZE)
    mB, iB, tB, eB, _ = predict_with_heads(gB)

    # C – image only
    gC = build_tf_dataset(test_clean_paths, zero_txt, test_lbl, BATCH_SIZE)
    mC, iC, tC, eC, _ = predict_with_heads(gC)

    # D – degraded + clean text
    gD = degrade(test_clean_paths, test_clean_txt, test_lbl)
    mD, iD, tD, eD, _ = predict_with_heads(gD)

    # E – degraded + no text
    gE = degrade(test_clean_paths, zero_txt, test_lbl)
    mE, iE, tE, eE, _ = predict_with_heads(gE)

    def acc(pred): return float(np.mean(pred == y_true))

    print(f"\n[A] MAIN Clean            -> {acc(mA):.4f}  | ENS -> {acc(eA):.4f}")
    print(f"[B] MAIN Noisy            -> {acc(mB):.4f}  | ENS -> {acc(eB):.4f}")
    print(f"[C] MAIN Image-Only       -> {acc(mC):.4f}  | ENS -> {acc(eC):.4f}")
    print(f"[D] MAIN Degraded+Text    -> {acc(mD):.4f}  | ENS -> {acc(eD):.4f}")
    print(f"[E] MAIN Degraded NoText  -> {acc(mE):.4f}  | ENS -> {acc(eE):.4f}")
    text_lift = acc(mD) - acc(mE)
    print(f"\n>>> Text Lift (MAIN Degraded): {text_lift:+.4f} ({text_lift:+.1%})")

    for tag, pred in [("[A] Clean", mA), ("[B] Noisy", mB), ("[C] ImgOnly", mC)]:
        print(f"\n{tag} – classification report:")
        print(classification_report(y_true, pred, target_names=DISEASE_CLASSES))

    # ── Reports ──────────────────────────────────────────────────────────────
    os.makedirs("reports", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    report_data = {
        "timestamp": ts,
        "model_path": SAVE_MODEL_PATH,
        "ensemble_weights": {"out_main": w_main, "out_img": w_img, "out_txt": w_txt},
        "A_clean":    {"main": acc(mA), "img": acc(iA), "txt": acc(tA), "ens": acc(eA),
                       "report_main": classification_report(y_true, mA, target_names=DISEASE_CLASSES, output_dict=True)},
        "B_noisy":    {"main": acc(mB), "img": acc(iB), "txt": acc(tB), "ens": acc(eB),
                       "report_main": classification_report(y_true, mB, target_names=DISEASE_CLASSES, output_dict=True)},
        "C_imgonly":  {"main": acc(mC), "img": acc(iC), "txt": acc(tC), "ens": acc(eC),
                       "report_main": classification_report(y_true, mC, target_names=DISEASE_CLASSES, output_dict=True)},
        "D_deg_text": {"main": acc(mD), "img": acc(iD), "txt": acc(tD), "ens": acc(eD),
                       "report_main": classification_report(y_true, mD, target_names=DISEASE_CLASSES, output_dict=True)},
        "E_deg_notxt":{"main": acc(mE), "img": acc(iE), "txt": acc(tE), "ens": acc(eE),
                       "report_main": classification_report(y_true, mE, target_names=DISEASE_CLASSES, output_dict=True)},
        "text_lift_main": text_lift,
        "text_lift_ens":  acc(eD) - acc(eE),
    }
    json_path = f"reports/classification_report_exp009_other_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    txt_path = f"reports/classification_report_exp009_other_{ts}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"=== RiceSafe EXP-009 OTHER ===\nTimestamp: {ts}\nModel: {SAVE_MODEL_PATH}\n\n")
        for tag, pred in [("A Clean", mA), ("B Noisy", mB), ("C ImgOnly", mC),
                          ("D DegText", mD), ("E DegNoTxt", mE)]:
            f.write(f"[{tag}] MAIN Acc: {acc(pred):.4f}\n")
            f.write(classification_report(y_true, pred, target_names=DISEASE_CLASSES))
            f.write("\n")
        f.write(f"Text Lift (MAIN): {text_lift:+.4f} ({text_lift:+.1%})\n")
    print(f"\n[*] Reports -> {json_path}\n[*] Reports -> {txt_path}")

    # ── Graphs ───────────────────────────────────────────────────────────────
    os.makedirs("graphs", exist_ok=True)
    try:
        plt.rcParams["font.family"] = "Tahoma"
    except Exception:
        pass

    # Confusion matrices – ensemble
    fig, axes = plt.subplots(1, 5, figsize=(36, 7))
    for ax, pred, title in zip(
        axes,
        [eA, eB, eC, eD, eE],
        [f"[A] ENS Clean\n{acc(eA):.1%}", f"[B] ENS Noisy\n{acc(eB):.1%}",
         f"[C] ENS ImgOnly\n{acc(eC):.1%}", f"[D] ENS Deg+Text\n{acc(eD):.1%}",
         f"[E] ENS Deg NoTxt\n{acc(eE):.1%}"],
    ):
        cm_ = confusion_matrix(y_true, pred)
        sns.heatmap(cm_, annot=True, fmt="d", cmap="Blues",
                    xticklabels=DISEASE_CLASSES, yticklabels=DISEASE_CLASSES, ax=ax)
        ax.set_title(title); ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("graphs/confusion_matrix_V3_exp009_other_ensemble.png")
    plt.close()

    # Confusion matrices – main head
    fig, axes = plt.subplots(1, 5, figsize=(36, 7))
    for ax, pred, title in zip(
        axes,
        [mA, mB, mC, mD, mE],
        [f"[A] MAIN Clean\n{acc(mA):.1%}", f"[B] MAIN Noisy\n{acc(mB):.1%}",
         f"[C] MAIN ImgOnly\n{acc(mC):.1%}", f"[D] MAIN Deg+Text\n{acc(mD):.1%}",
         f"[E] MAIN Deg NoTxt\n{acc(mE):.1%}"],
    ):
        cm_ = confusion_matrix(y_true, pred)
        sns.heatmap(cm_, annot=True, fmt="d", cmap="Oranges",
                    xticklabels=DISEASE_CLASSES, yticklabels=DISEASE_CLASSES, ax=ax)
        ax.set_title(title); ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("graphs/confusion_matrix_V3_exp009_other_main.png")
    plt.close()

    # Training history
    if history1 is not None and history2 is not None:
        acc_key  = "out_main_accuracy" if "out_main_accuracy" in history1.history else "accuracy"
        loss_key = "out_main_loss"     if "out_main_loss"     in history1.history else "loss"
        acc_  = history1.history[acc_key]           + history2.history[acc_key]
        val_a = history1.history[f"val_{acc_key}"]  + history2.history[f"val_{acc_key}"]
        loss_ = history1.history[loss_key]           + history2.history[loss_key]
        val_l = history1.history[f"val_{loss_key}"]  + history2.history[f"val_{loss_key}"]
        p1_end = len(history1.history[acc_key])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, tr, vl, title in zip(axes, [acc_, loss_], [val_a, val_l],
                                     ["Accuracy (out_main)", "Loss (out_main)"]):
            ax.axvline(p1_end, color="gray", linestyle="--", label="Fine-tune start")
            ax.plot(tr, label="Train"); ax.plot(vl, label="Val")
            ax.set_title(title); ax.legend()
        plt.tight_layout()
        plt.savefig("graphs/training_history_V3_exp009_other.png")
        plt.close()

    print("[*] Done! Graphs saved to graphs/")


# ─── TRAINING ────────────────────────────────────────────────────────────────

def run_training():
    print("\n>>> EXP-009 OTHER: TEXTBOOST – SUB-CLASS ROUTED TEXT")
    print(f"Data Root      : {DATA_ROOT}")
    print(f"Other CSV Root : {OTHER_CSV_ROOT}")
    print(f"Save Model     : {SAVE_MODEL_PATH}")
    print(f"TEXT_ZERO_PROB : {TEXT_ZERO_PROB}  |  IMG_ZERO_PROB : {IMG_ZERO_PROB}")

    def load_split(sp):
        txt   = np.load(os.path.join(TEXT_FEATURES_DIR, f"{sp}_text_feats.npy"))
        lbls  = np.load(os.path.join(TEXT_FEATURES_DIR, f"{sp}_labels.npy"), allow_pickle=True)
        paths = np.load(os.path.join(TEXT_FEATURES_DIR, f"{sp}_img_paths.npy"), allow_pickle=True)
        return paths, txt, lbls

    train_paths, train_txt, train_lbl_raw = load_split("train")
    val_paths,   val_txt,   val_lbl_raw   = load_split("val")
    test_clean_paths, test_clean_txt, test_clean_lbl_raw = load_split("test_clean")
    test_noisy_paths, test_noisy_txt, _                  = load_split("test_noisy")

    le = LabelEncoder()
    le.fit(DISEASE_CLASSES)
    joblib.dump(le, LABEL_ENCODER_PATH)

    train_lbl = le.transform(train_lbl_raw)
    val_lbl   = le.transform(val_lbl_raw)
    test_lbl  = le.transform(test_clean_lbl_raw)

    unique, counts = np.unique(train_lbl, return_counts=True)
    total    = counts.sum()
    cw_dict  = {int(c): total / (NUM_CLASSES * cnt) for c, cnt in zip(unique, counts)}
    sample_w = np.array([cw_dict[lbl] for lbl in train_lbl], dtype="float32")

    print(f"Train: {len(train_paths)}  Val: {len(val_paths)}  Test: {len(test_clean_paths)}")

    train_gen = build_tf_dataset(train_paths, train_txt, train_lbl, BATCH_SIZE,
                                 is_training=True, apply_dropout=True, sample_weights=sample_w)
    val_gen   = build_tf_dataset(val_paths, val_txt, val_lbl, BATCH_SIZE,
                                 is_training=False, apply_dropout=False)

    model, base_model = build_multimodal_model(fine_tune=False)

    loss_weights = {"out_main": 1.0, "out_img": 0.3, "out_txt": 0.8}

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss={"out_main": "sparse_categorical_crossentropy",
              "out_img":  "sparse_categorical_crossentropy",
              "out_txt":  "sparse_categorical_crossentropy"},
        loss_weights=loss_weights,
        metrics={"out_main": "accuracy", "out_img": "accuracy", "out_txt": "accuracy"},
    )

    cbs_p1 = [
        EarlyStopping(patience=5, restore_best_weights=True,
                      monitor="val_out_main_loss", mode="min", verbose=1),
        ModelCheckpoint(SAVE_MODEL_PATH, save_best_only=True,
                        monitor="val_out_main_accuracy", mode="max", verbose=1),
    ]
    history1 = model.fit(train_gen, validation_data=val_gen,
                         epochs=20, callbacks=cbs_p1, verbose=2)

    print("\n[Phase 2] Fine-tuning MobileNetV2 ...")
    base_model.trainable = True
    for layer in base_model.layers[:-NUM_FINE_TUNE_LAYERS]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss={"out_main": "sparse_categorical_crossentropy",
              "out_img":  "sparse_categorical_crossentropy",
              "out_txt":  "sparse_categorical_crossentropy"},
        loss_weights=loss_weights,
        metrics={"out_main": "accuracy", "out_img": "accuracy", "out_txt": "accuracy"},
    )

    cbs_p2 = [
        EarlyStopping(patience=10, restore_best_weights=True,
                      monitor="val_out_main_loss", mode="min", verbose=1),
        ReduceLROnPlateau(monitor="val_out_main_loss", mode="min",
                          factor=0.2, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(SAVE_MODEL_PATH, save_best_only=True,
                        monitor="val_out_main_accuracy", mode="max", verbose=1),
    ]
    history2 = model.fit(train_gen, validation_data=val_gen,
                         epochs=100, callbacks=cbs_p2, verbose=2)

    print(f"\n[*] Training finished. Saved: {SAVE_MODEL_PATH}")
    _run_evaluation(model, test_clean_paths, test_clean_txt,
                    test_noisy_txt, test_lbl, history1, history2)


# ─── EVAL ONLY ───────────────────────────────────────────────────────────────

def run_eval_only():
    from tensorflow.keras.models import load_model as keras_load

    def load_split(sp):
        txt   = np.load(os.path.join(TEXT_FEATURES_DIR, f"{sp}_text_feats.npy"))
        lbls  = np.load(os.path.join(TEXT_FEATURES_DIR, f"{sp}_labels.npy"), allow_pickle=True)
        paths = np.load(os.path.join(TEXT_FEATURES_DIR, f"{sp}_img_paths.npy"), allow_pickle=True)
        return paths, txt, lbls

    test_clean_paths, test_clean_txt, test_clean_lbl_raw = load_split("test_clean")
    _, test_noisy_txt, _ = load_split("test_noisy")

    le = LabelEncoder()
    le.fit(DISEASE_CLASSES)
    test_lbl = le.transform(test_clean_lbl_raw)

    model = keras_load(SAVE_MODEL_PATH)
    _run_evaluation(model, test_clean_paths, test_clean_txt,
                    test_noisy_txt, test_lbl)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EXP-009 OTHER training pipeline")
    parser.add_argument(
        "--mode",
        choices=["pipeline", "extract", "train", "eval"],
        default="pipeline",
        help="pipeline=extract+train+eval | extract | train | eval (load saved model)",
    )
    args = parser.parse_args()

    if args.mode in ["extract", "pipeline"]:
        extract_text_features()
    if args.mode in ["train", "pipeline"]:
        run_training()
    if args.mode == "eval":
        run_eval_only()


if __name__ == "__main__":
    main()
