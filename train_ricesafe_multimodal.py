import gc
import os

import joblib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.utils import to_categorical
from transformers import AutoModel, AutoTokenizer

# --- Environment and Global Configuration ---
dotenv_path = os.path.join(os.path.dirname(__file__), ".env.train")
if os.path.exists(dotenv_path):
    print(f"[*] Loading environment variables from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
else:
    print(
        f"[INFO] .env.train file not found at {dotenv_path}. Using default hyperparameter values or system environment variables if set."
    )

os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "false")

# Font setup for Matplotlib
FONT_FILE_PATH = "fonts/NotoSansThai-VariableFont_wdth,wght.ttf"
THAI_FONT_PROPERTIES = None
if os.path.exists(FONT_FILE_PATH):
    try:
        THAI_FONT_PROPERTIES = fm.FontProperties(fname=FONT_FILE_PATH)
        plt.rcParams["font.family"] = THAI_FONT_PROPERTIES.get_name()
        plt.rcParams["font.sans-serif"] = [
            THAI_FONT_PROPERTIES.get_name()
        ] + plt.rcParams["font.sans-serif"]
        print(
            f"[*] Matplotlib font set: {FONT_FILE_PATH} (Family: {THAI_FONT_PROPERTIES.get_name()})"
        )
    except Exception as e:
        print(f"[WARNING] Font load error: {e}")
else:
    print(f"[WARNING] Thai font file not found: {FONT_FILE_PATH} (CWD: {os.getcwd()})")

# PyTorch device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data and artifact paths
DATA_BASE_DIR = "data/"
TRAIN_IMG_DIR = os.path.join(DATA_BASE_DIR, "train_image/")
TEST_IMG_DIR = os.path.join(DATA_BASE_DIR, "test_image/")
TRAIN_TEXT_DIR = os.path.join(DATA_BASE_DIR, "train_text/")
TEST_TEXT_DIR = os.path.join(DATA_BASE_DIR, "test_text/")
ARTIFACTS_DIR = "local_run_artifacts/"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
LABEL_ENCODER_PATH_LOCAL = os.path.join(ARTIFACTS_DIR, "label_encoder.joblib")
SCALER_PATH_LOCAL = os.path.join(ARTIFACTS_DIR, "scaler.joblib")
MODEL_CHECKPOINT_PATH_LOCAL = os.path.join(ARTIFACTS_DIR, "best_model_checkpoint.keras")
TRAINING_HISTORY_PLOT_PATH = os.path.join(ARTIFACTS_DIR, "training_history_plot.png")
CONFUSION_MATRIX_PLOT_PATH = os.path.join(ARTIFACTS_DIR, "confusion_matrix_plot.png")
TEST_PREDICTIONS_CSV_PATH = os.path.join(ARTIFACTS_DIR, "test_predictions.csv")
CLASSIFICATION_REPORT_TEXT_PATH = os.path.join(
    ARTIFACTS_DIR, "classification_report.txt"
)

# Class definitions and mappings
DISEASE_CLASSES = ["ปกติ", "โรคขอบใบแห้ง", "โรคใบขีดโปร่งแสง", "โรคใบจุดสีน้ำตาล", "โรคไหม้"]
NUM_CLASSES = len(DISEASE_CLASSES)
DISEASE_CLASSES_MLFLOW_SAFE = {
    "ปกติ": "Normal",
    "โรคขอบใบแห้ง": "BacterialLeafBlight",
    "โรคใบขีดโปร่งแสง": "BacterialLeafStreak",
    "โรคใบจุดสีน้ำตาล": "BrownSpot",
    "โรคไหม้": "Blast",
}
assert all(cls_name in DISEASE_CLASSES_MLFLOW_SAFE for cls_name in DISEASE_CLASSES)

# Training hyperparameters
EPOCHS_MAX = int(os.environ.get("EXP_EPOCHS_MAX", 100))
BATCH_SIZE = int(os.environ.get("EXP_BATCH_SIZE", 32))
VALIDATION_SET_SIZE = float(os.environ.get("EXP_VALIDATION_SET_SIZE", 0.2))
RANDOM_STATE = int(os.environ.get("EXP_RANDOM_STATE", 42))
INITIAL_LEARNING_RATE = float(os.environ.get("EXP_INITIAL_LEARNING_RATE", 0.0001))
OPTIMIZER_TYPE = os.environ.get("EXP_OPTIMIZER_TYPE", "Adam")
EARLY_STOPPING_PATIENCE = int(os.environ.get("EXP_EARLY_STOPPING_PATIENCE", 10))
REDUCE_LR_PATIENCE = int(os.environ.get("EXP_REDUCE_LR_PATIENCE", 5))
REDUCE_LR_FACTOR = float(os.environ.get("EXP_REDUCE_LR_FACTOR", 0.2))
MIN_LR_ON_PLATEAU = float(os.environ.get("EXP_MIN_LR_ON_PLATEAU", 1e-6))

# MLflow experiment name
MLFLOW_EXPERIMENT_NAME = "RiceSafe_MultiModal_V1"


# --- Image and Text Feature Extraction Functions ---
def load_single_image(img_path, target_size=(224, 224)):
    try:
        img = tf_image.load_img(img_path, target_size=target_size)
        img_array = tf_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)
    except Exception as e:
        print(f"[ERROR] Load image {img_path}: {e}")
        return None


print("[*] Initializing text model (BAAI/bge-m3) and tokenizer...")
try:
    text_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    text_embedding_model = AutoModel.from_pretrained("BAAI/bge-m3").to(DEVICE).eval()
    print("[DONE] Text model/tokenizer initialized.")
except Exception as e:
    print(f"[FATAL] Text model/tokenizer load error: {e}. Exiting.")
    exit(1)


@torch.no_grad()
def embed_text_batch(texts_list):
    if not isinstance(texts_list, list):
        texts_list = [str(texts_list)]
    inputs = text_tokenizer(
        texts_list, return_tensors="pt", truncation=True, padding=True, max_length=512
    ).to(DEVICE)
    outputs = text_embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


print("[*] Initializing image feature extractor (MobileNetV2)...")
try:
    image_feature_extractor = MobileNetV2(
        weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3)
    )
    image_feature_extractor.trainable = False
    print("[DONE] Image feature extractor initialized.")
except Exception as e:
    print(f"[FATAL] MobileNetV2 load error: {e}. Exiting.")
    exit(1)


def extract_image_features_batch(image_arrays_batch):
    if isinstance(image_arrays_batch, list):
        image_arrays_batch = np.vstack(image_arrays_batch)
    return image_feature_extractor.predict(image_arrays_batch)


# --- Data Pairing and Loading ---
def pair_images_and_texts(mode="train", batch_size=32):
    paired_data_list = []
    base_img_dir = TRAIN_IMG_DIR if mode == "train" else TEST_IMG_DIR
    base_text_dir = TRAIN_TEXT_DIR if mode == "train" else TEST_TEXT_DIR
    data_type_str = "Training" if mode == "train" else "Test"
    print(f"[*] Pairing {data_type_str} data from {base_img_dir} & {base_text_dir}...")
    available_classes = [
        d
        for d in os.listdir(base_img_dir)
        if os.path.isdir(os.path.join(base_img_dir, d))
    ]
    if not available_classes:
        print(f"[WARNING] No class subdirs in {base_img_dir}")
        return []

    for disease_class in available_classes:
        img_folder = os.path.join(base_img_dir, disease_class)
        text_file = os.path.join(base_text_dir, f"{disease_class}.csv")
        if not os.path.isfile(text_file):
            print(f"[SKIP] No CSV: {text_file}")
            continue
        image_files = sorted(
            [
                f
                for f in os.listdir(img_folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        if not image_files:
            print(f"[INFO] No images for {disease_class}")
            continue
        try:
            df_text = pd.read_csv(text_file)
            if df_text.empty or df_text.iloc[:, 0].isnull().all():
                print(f"[WARNING] Empty CSV: {text_file}")
                continue
            text_lines = df_text.iloc[:, 0].astype(str).tolist()
        except Exception as e:
            print(f"[ERROR] Reading CSV {text_file}: {e}")
            continue
        if len(image_files) != len(text_lines):
            print(
                f"[WARNING] Mismatch {disease_class}: {len(image_files)}img/{len(text_lines)}txt"
            )
            continue

        for i in range(0, len(image_files), batch_size):
            current_image_files, current_texts = (
                image_files[i : i + batch_size],
                text_lines[i : i + batch_size],
            )
            batch_loaded_images, valid_texts_for_batch = [], []
            for img_name, text_content in zip(current_image_files, current_texts):
                img_array = load_single_image(os.path.join(img_folder, img_name))
                if img_array is not None:
                    batch_loaded_images.append(img_array)
                    valid_texts_for_batch.append(text_content)
            if batch_loaded_images and valid_texts_for_batch:
                img_feats = extract_image_features_batch(batch_loaded_images)
                text_feats = embed_text_batch(valid_texts_for_batch)
                paired_data_list.extend(
                    [
                        (
                            np.concatenate((img_feat.flatten(), text_feat.flatten())),
                            disease_class,
                        )
                        for img_feat, text_feat in zip(img_feats, text_feats)
                    ]
                )
            gc.collect()
    print(f"[DONE] Paired {len(paired_data_list)} {data_type_str} entries.")
    return paired_data_list


# --- Plotting Utility Functions ---
def plot_training_history_to_file(history, file_path):
    plt.figure(figsize=(12, 5))
    font_prop = THAI_FONT_PROPERTIES if THAI_FONT_PROPERTIES else None
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy", fontproperties=font_prop)
    plt.ylabel("Accuracy", fontproperties=font_prop)
    plt.xlabel("Epoch", fontproperties=font_prop)
    plt.legend(loc="lower right", prop=font_prop)
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss", fontproperties=font_prop)
    plt.ylabel("Loss", fontproperties=font_prop)
    plt.xlabel("Epoch", fontproperties=font_prop)
    plt.legend(loc="upper right", prop=font_prop)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    print(f"[*] Training history plot: {file_path}")


def plot_confusion_matrix_to_file(
    y_true_encoded, y_pred_encoded, class_names_ordered, file_path, normalize=False
):
    cm = confusion_matrix(y_true_encoded, y_pred_encoded)
    plt.figure(figsize=(10, 8))
    font_prop = THAI_FONT_PROPERTIES if THAI_FONT_PROPERTIES else None
    sns.heatmap(
        cm,
        annot=True,
        fmt="d" if not normalize else ".2f",
        cmap=plt.cm.Blues,
        xticklabels=class_names_ordered,
        yticklabels=class_names_ordered,
        annot_kws={"fontproperties": font_prop},
    )
    title_text = "Confusion Matrix"
    if normalize:
        title_text = "Normalized " + title_text
    plt.title(title_text, fontproperties=font_prop)
    plt.ylabel("True Label", fontproperties=font_prop)
    plt.xlabel("Predicted Label", fontproperties=font_prop)
    ax = plt.gca()
    ax.set_xticklabels(
        ax.get_xticklabels(), fontproperties=font_prop, rotation=45, ha="right"
    )
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    print(f"[*] Confusion matrix plot: {file_path}")


# --- Main Training and Evaluation Logic with MLflow ---
def train_and_evaluate():
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(
            f"[*] MLflow Run Started (ID: {run_id}) for experiment '{MLFLOW_EXPERIMENT_NAME}'"
        )
        mlflow.log_param("mlflow_run_id", run_id)
        mlflow.log_param(
            "version", os.environ.get("EXPERIMENT_VERSION_TAG", "Version1")
        )

        # Log parameters
        params_to_log = {
            "epochs_max": EPOCHS_MAX,
            "batch_size": BATCH_SIZE,
            "validation_set_size": VALIDATION_SET_SIZE,
            "random_state": RANDOM_STATE,
            "optimizer_type": OPTIMIZER_TYPE,
            "initial_learning_rate": INITIAL_LEARNING_RATE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "reduce_lr_patience": REDUCE_LR_PATIENCE,
            "reduce_lr_factor": REDUCE_LR_FACTOR,
            "min_lr_on_plateau": MIN_LR_ON_PLATEAU,
            "num_disease_classes": NUM_CLASSES,
            "image_feature_extractor": "MobileNetV2_avg_pool",
            "text_embedding_model": "BAAI/bge-m3",
            "pytorch_device": str(DEVICE),
            "font_file_path_for_plots": (
                FONT_FILE_PATH if THAI_FONT_PROPERTIES else "Not Used/Found"
            ),
        }
        mlflow.log_params(params_to_log)
        mlflow.log_param("disease_classes_original_list", ", ".join(DISEASE_CLASSES))
        mlflow.log_param(
            "disease_classes_mlflow_safe_map", str(DISEASE_CLASSES_MLFLOW_SAFE)
        )

        # Data collection and initial processing
        all_training_data = pair_images_and_texts(mode="train", batch_size=BATCH_SIZE)
        if not all_training_data:
            print("[FATAL] No training data. Exiting.")
            mlflow.log_metric("total_collected_train_samples", 0)
            return
        features_all_raw, labels_all_str = zip(*all_training_data)
        X_all_features, y_all_labels_str = np.array(
            [f.flatten() for f in features_all_raw]
        ), np.array(labels_all_str)
        mlflow.log_metric("total_collected_train_samples", len(X_all_features))

        # Train/validation split
        print(f"[*] Splitting data (validation_size={VALIDATION_SET_SIZE})...")
        X_train, X_val, y_train_str, y_val_str = train_test_split(
            X_all_features,
            y_all_labels_str,
            test_size=VALIDATION_SET_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_all_labels_str,
        )
        mlflow.log_metrics(
            {
                "train_samples_count": len(X_train),
                "validation_samples_count": len(X_val),
            }
        )
        print(f"[DONE] Train: {len(X_train)}, Validation: {len(X_val)}")

        # Feature scaling
        print("[*] Scaling features...")
        scaler = StandardScaler()
        X_train_scaled, X_val_scaled = scaler.fit_transform(X_train), scaler.transform(
            X_val
        )
        joblib.dump(scaler, SCALER_PATH_LOCAL)
        if X_train.shape[0] > 0:
            ex_scl_unscaled = X_train[: min(5, X_train.shape[0])]
            if ex_scl_unscaled.ndim == 1:
                ex_scl_unscaled = ex_scl_unscaled.reshape(1, -1)
            if ex_scl_unscaled.ndim == 2 and ex_scl_unscaled.shape[0] > 0:
                try:
                    sig_scl = infer_signature(
                        ex_scl_unscaled, scaler.transform(ex_scl_unscaled)
                    )
                    mlflow.sklearn.log_model(
                        scaler,
                        "feature_scaler",
                        signature=sig_scl,
                        input_example=ex_scl_unscaled,
                    )
                except Exception as e:
                    mlflow.sklearn.log_model(scaler, "feature_scaler")
                    print(f"[WARN] Scaler sig err: {e}")
            else:
                mlflow.sklearn.log_model(scaler, "feature_scaler")
                print("[WARN] Scaler ex unsuitable for sig.")
        else:
            mlflow.sklearn.log_model(scaler, "feature_scaler")
        print("[DONE] Feature scaling.")

        # Label encoding
        print("[*] Encoding labels...")
        label_encoder = LabelEncoder()
        y_train_encoded, y_val_encoded = label_encoder.fit_transform(
            y_train_str
        ), label_encoder.transform(y_val_str)
        y_train_categorical = to_categorical(y_train_encoded, NUM_CLASSES)
        y_val_categorical = to_categorical(y_val_encoded, NUM_CLASSES)
        joblib.dump(label_encoder, LABEL_ENCODER_PATH_LOCAL)
        if y_train_str.size > 0:
            ex_le_raw = y_train_str[: min(5, y_train_str.size)]
            try:
                in_df, out_df = pd.DataFrame(ex_le_raw, columns=["l_in"]), pd.DataFrame(
                    label_encoder.transform(ex_le_raw), columns=["l_out"]
                )
                sig_le = infer_signature(in_df, out_df)
                mlflow.sklearn.log_model(
                    label_encoder,
                    "label_encoder",
                    signature=sig_le,
                    input_example=in_df,
                )
            except Exception as e:
                mlflow.sklearn.log_model(label_encoder, "label_encoder")
                print(f"[WARN] LabelEnc sig err: {e}")
        else:
            mlflow.sklearn.log_model(label_encoder, "label_encoder")
        ordered_class_names_from_le = label_encoder.classes_
        print("[DONE] Label encoding.")
        mlflow.log_param("input_feature_dimension", X_train_scaled.shape[1])

        # Model definition and compilation
        print("[*] Defining Keras model...")
        model = Sequential(
            [
                Input(shape=(X_train_scaled.shape[1],)),
                Dense(256, activation="relu"),
                BatchNormalization(),
                Dropout(0.3),
                Dense(128, activation="relu"),
                BatchNormalization(),
                Dropout(0.3),
                Dense(NUM_CLASSES, activation="softmax"),
            ]
        )
        optimizer_instance = Adam(learning_rate=INITIAL_LEARNING_RATE)
        model.compile(
            optimizer=optimizer_instance,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        mlflow.log_text("\n".join(summary_list), "model_summary.txt")
        print("[DONE] Model defined.")

        # Keras autologging and callbacks setup
        print("[*] Setting up Keras autologging & callbacks...")
        try:
            mlflow.keras.autolog(
                log_model_format="tf", registered_model_name=None, log_models=True
            )
            print("[INFO] Keras autolog: 'tf' format.")
        except TypeError:
            print("[WARN] Keras autolog: 'tf' format failed. Trying default.")
            try:
                mlflow.keras.autolog(registered_model_name=None, log_models=True)
                print("[INFO] Keras autolog: default format.")
            except Exception as e:
                print(f"[ERROR] Keras autolog entirely failed: {e}")
        except Exception as e:
            print(f"[ERROR] Keras autolog unexpected error: {e}")
        callbacks_list = [
            EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=REDUCE_LR_FACTOR,
                patience=REDUCE_LR_PATIENCE,
                min_lr=MIN_LR_ON_PLATEAU,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=MODEL_CHECKPOINT_PATH_LOCAL,
                save_best_only=True,
                monitor="val_accuracy",
                verbose=1,
            ),
        ]
        print("[DONE] Keras setup.")

        # Model training
        print("[*] Training model...")
        history = model.fit(
            X_train_scaled,
            y_train_categorical,
            validation_data=(X_val_scaled, y_val_categorical),
            epochs=EPOCHS_MAX,
            batch_size=BATCH_SIZE,
            callbacks=callbacks_list,
            verbose=1,
        )
        print("[DONE] Model training.")
        if os.path.exists(MODEL_CHECKPOINT_PATH_LOCAL):
            mlflow.log_artifact(
                MODEL_CHECKPOINT_PATH_LOCAL, "best_model_by_val_accuracy"
            )
        plot_training_history_to_file(history, TRAINING_HISTORY_PLOT_PATH)
        mlflow.log_artifact(TRAINING_HISTORY_PLOT_PATH, "evaluation_plots")

        # Test set evaluation
        print("[*] Preparing test data...")
        test_data_paired = pair_images_and_texts(mode="test", batch_size=BATCH_SIZE)
        if not test_data_paired:
            print("[WARNING] No test data. Skipping test eval.")
            mlflow.log_metric("test_samples_count", 0)
        else:
            test_features_raw, test_labels_str_raw = zip(*test_data_paired)
            X_test, y_test_str = np.array(
                [f.flatten() for f in test_features_raw]
            ), np.array(test_labels_str_raw)
            mlflow.log_metric("test_samples_count", len(X_test))
            X_test_scaled, y_test_encoded = scaler.transform(
                X_test
            ), label_encoder.transform(y_test_str)
            final_model_to_evaluate = model
            print("[*] Evaluating on test set...")
            predictions_probs_test = final_model_to_evaluate.predict(X_test_scaled)
            predicted_labels_encoded_test = np.argmax(predictions_probs_test, axis=1)

            report_dict_test = classification_report(
                y_test_encoded,
                predicted_labels_encoded_test,
                labels=np.arange(len(ordered_class_names_from_le)),
                target_names=ordered_class_names_from_le,
                output_dict=True,
                zero_division=0,
            )
            report_str_test = classification_report(
                y_test_encoded,
                predicted_labels_encoded_test,
                labels=np.arange(len(ordered_class_names_from_le)),
                target_names=ordered_class_names_from_le,
                zero_division=0,
            )
            print("\nClassification Report (Test Set):\n", report_str_test)
            with open(CLASSIFICATION_REPORT_TEXT_PATH, "w", encoding="utf-8") as f:
                f.write(f"Test Set Report:\n{report_str_test}")
            mlflow.log_artifact(CLASSIFICATION_REPORT_TEXT_PATH, "evaluation_results")
            mlflow.log_metric("test_set_accuracy", report_dict_test["accuracy"])
            for mt in ["precision", "recall", "f1-score"]:
                mlflow.log_metric(
                    f"test_set_macro_avg_{mt}", report_dict_test["macro avg"][mt]
                )
                mlflow.log_metric(
                    f"test_set_weighted_avg_{mt}", report_dict_test["weighted avg"][mt]
                )
            for ocn in ordered_class_names_from_le:
                msn = DISEASE_CLASSES_MLFLOW_SAFE[ocn]
                if ocn in report_dict_test:
                    for mt in ["precision", "recall", "f1-score"]:
                        mlflow.log_metric(
                            f"test_set_{msn}_{mt}", report_dict_test[ocn][mt]
                        )

            plot_confusion_matrix_to_file(
                y_test_encoded,
                predicted_labels_encoded_test,
                ordered_class_names_from_le,
                CONFUSION_MATRIX_PLOT_PATH,
            )
            mlflow.log_artifact(CONFUSION_MATRIX_PLOT_PATH, "evaluation_plots")

            print("[*] Saving test predictions to CSV...")
            results_df = pd.DataFrame(
                {
                    "True_Label": y_test_str,
                    "Predicted_Label": label_encoder.inverse_transform(
                        predicted_labels_encoded_test
                    ),
                    "Confidence": np.max(predictions_probs_test, axis=1),
                }
            )
            for i, cn in enumerate(ordered_class_names_from_le):
                results_df[f"Prob_{DISEASE_CLASSES_MLFLOW_SAFE[cn]}"] = (
                    predictions_probs_test[:, i]
                )
            results_df.to_csv(TEST_PREDICTIONS_CSV_PATH, index=False, encoding="utf-8")
            mlflow.log_artifact(TEST_PREDICTIONS_CSV_PATH, "evaluation_results")
            print(f"[DONE] Test predictions saved to {TEST_PREDICTIONS_CSV_PATH}")

        # Log source code
        mlflow.log_artifact(__file__, "source_code")
        print(f"--- MLflow Run '{run.info.run_name}' (ID: {run_id}) completed. ---")


# --- Script Entry Point and Error Handling ---
if __name__ == "__main__":
    try:
        train_and_evaluate()
    except Exception as e:
        print(f"[FATAL ERROR] Uncaught exception: {e}")
        import traceback

        traceback.print_exc()
        if mlflow.active_run():
            mlflow.log_params(
                {
                    "script_execution_status": "FAILED",
                    "script_error_message": str(e)[:2000],
                }
            )
            mlflow.end_run(status="FAILED")
    finally:
        import shutil

        if os.path.exists(ARTIFACTS_DIR):
            shutil.rmtree(ARTIFACTS_DIR)
            print(f"[*] Cleaned up: {ARTIFACTS_DIR}")
        print("--- Training script completed successfully. ---")
        print("--- View MLflow UI with: mlflow ui --port 5001 ---")
