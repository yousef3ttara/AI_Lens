"""
AI Lens - Plant Leaf Disease Detection Model Training Pipeline
Transfer Learning with MobileNetV2 on PlantVillage Dataset (38 Classes)

Features:
- Programmatic dataset loading via Hugging Face `datasets` or local directory
- Preserves leaf-level train/validation/test splits
- MobileNetV2 preprocessing (-1 to 1) & ImageNet transfer learning
- Training-only data augmentation (rotation, zoom, flip, contrast)
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- Optional two-phase fine-tuning
- Comprehensive evaluation: Accuracy, Precision, Recall, Macro F1, Confusion Matrix
- Saves trained model to models/ and visual reports to reports/
"""

import os
import sys
import json
import argparse
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress noisy TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


# Standard configuration defaults
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 38
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-3


def check_compute_devices() -> None:
    """Detect and report available CPU/GPU hardware."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[*] GPU Acceleration Enabled: Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"    - {gpu.name}")
        # Enable dynamic memory growth to prevent TF from hogging full VRAM
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                print(f"    [!] Could not set memory growth for {gpu.name}: {e}")
    else:
        print("[*] No GPU detected. Proceeding with CPU computation.")
        print(f"    Available CPU cores: {os.cpu_count()}")


def get_data_augmentation() -> keras.Sequential:
    """Create data augmentation pipeline to be applied exclusively to training set."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", name="aug_flip"),
            layers.RandomRotation(0.15, name="aug_rotation"),
            layers.RandomZoom(0.1, name="aug_zoom"),
            layers.RandomTranslation(0.08, 0.08, name="aug_translation"),
            layers.RandomContrast(0.1, name="aug_contrast"),
        ],
        name="training_augmentation",
    )


def build_mobilenetv2_model(num_classes: int = NUM_CLASSES, learning_rate: float = DEFAULT_LR) -> keras.Model:
    """
    Construct transfer learning model using MobileNetV2 pretrained on ImageNet.
    Preprocesses inputs to [-1, 1] range inside the model graph.
    """
    inputs = keras.Input(shape=INPUT_SHAPE, name="leaf_image_input")

    # MobileNetV2 expects [-1, 1] input normalization
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenetv2_scaling")(inputs)

    # Base pretrained model (ImageNet weights)
    base_model = keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze base during phase 1

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_features")(x)
    x = layers.BatchNormalization(name="batch_norm_2")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="AI_Lens_MobileNetV2")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_dataset_from_directory(
    data_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Load dataset from a directory containing 38 class subdirectories.
    Splits data into Train (70%), Validation (15%), and Test (15%) splits
    without data leakage.
    """
    print(f"[*] Scanning local dataset directory: {data_dir}")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Directory '{data_dir}' does not exist. Please place the PlantVillage dataset "
            f"in this directory or pass --data_dir with the correct path."
        )

    # First load train+val split
    full_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    class_names = full_ds.class_names
    total_batches = len(full_ds)

    if len(class_names) != NUM_CLASSES:
        print(f"[!] Warning: Expected {NUM_CLASSES} classes, found {len(class_names)} in directory.")

    test_batches = max(1, int(total_batches * test_split))
    val_batches = max(1, int(total_batches * val_split))
    train_batches = total_batches - test_batches - val_batches

    print(f"[*] Total batches: {total_batches} | Train: {train_batches} | Val: {val_batches} | Test: {test_batches}")

    # Unbatch & re-split to guarantee clean disjoint partitions
    test_ds = full_ds.take(test_batches)
    remaining_ds = full_ds.skip(test_batches)
    val_ds = remaining_ds.take(val_batches)
    train_ds = remaining_ds.skip(val_batches)

    # Apply data augmentation ONLY to training data
    aug_pipeline = get_data_augmentation()
    train_ds = train_ds.map(lambda x, y: (aug_pipeline(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def load_dataset_from_huggingface(
    dataset_name: str = "mohanty/PlantVillage",
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    seed: int = 42,
    target_data_dir: str = "data/plantvillage_color",
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Downloads and extracts the official PlantVillage dataset archive directly from the
    Hugging Face repository without relying on deprecated dataset loading scripts
    (which were removed in Hugging Face datasets >= 3.0).
    """
    import urllib.request
    import zipfile

    # If already extracted, load directly
    if os.path.exists(target_data_dir) and len(os.listdir(target_data_dir)) >= NUM_CLASSES:
        print(f"[*] Found existing extracted dataset at: {target_data_dir}")
        return load_dataset_from_directory(
            target_data_dir, batch_size=batch_size, image_size=image_size, seed=seed
        )

    archive_url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/data.zip"
    os.makedirs("data", exist_ok=True)
    zip_dest = os.path.join("data", "plantvillage_data.zip")

    if not os.path.exists(zip_dest):
        print(f"[*] Downloading official PlantVillage archive from Hugging Face:")
        print(f"    URL: {archive_url}")
        print("    Size: ~2.0 GB (this will take a few moments depending on connection)...")

        try:
            # Download with progress report
            def _progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = downloaded / total_size * 100
                    mb_down = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    sys.stdout.write(f"\r    Progress: {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
                else:
                    sys.stdout.write(f"\r    Downloaded: {downloaded / (1024 * 1024):.1f} MB")
                sys.stdout.flush()

            urllib.request.urlretrieve(archive_url, zip_dest, reporthook=_progress_hook)
            print("\n[*] Archive download complete!")
        except Exception as e:
            raise RuntimeError(
                f"\n[ERROR] Failed to download PlantVillage archive from Hugging Face.\n"
                f"Reason: {e}\n\n"
                "Instructions to resolve:\n"
                "1. Verify internet connectivity.\n"
                "2. Alternatively, manually download data.zip from:\n"
                f"   {archive_url}\n"
                "   and place it at 'data/plantvillage_data.zip', or extract color images to 'data/plantvillage_color'.\n"
            )

    print(f"[*] Extracting color dataset into '{target_data_dir}'...")
    try:
        with zipfile.ZipFile(zip_dest, "r") as zf:
            # Extract only the color images folder (raw/color/...)
            color_members = [m for m in zf.namelist() if m.startswith("raw/color/") and not m.endswith("/")]
            print(f"[*] Unpacking {len(color_members)} images...")
            for member in color_members:
                # Remove 'raw/color/' prefix to save cleanly into target_data_dir/<class>/<file>
                rel_path = member[len("raw/color/"):]
                target_path = os.path.join(target_data_dir, rel_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with zf.open(member) as source, open(target_path, "wb") as target:
                    target.write(source.read())
        print(f"[*] Extraction complete! Dataset ready at: {target_data_dir}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract '{zip_dest}': {e}")

    return load_dataset_from_directory(
        target_data_dir, batch_size=batch_size, image_size=image_size, seed=seed
    )


def plot_training_history(history_dict: Dict[str, List[float]], output_path: str) -> None:
    """Plot and save training/validation accuracy and loss curves."""
    epochs_range = range(1, len(history_dict["loss"]) + 1)

    plt.figure(figsize=(14, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_dict["loss"], "o-", label="Training Loss", color="#d9534f")
    if "val_loss" in history_dict:
        plt.plot(epochs_range, history_dict["val_loss"], "s--", label="Validation Loss", color="#0275d8")
    plt.title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_dict["accuracy"], "o-", label="Training Accuracy", color="#5cb85c")
    if "val_accuracy" in history_dict:
        plt.plot(epochs_range, history_dict["val_accuracy"], "s--", label="Validation Accuracy", color="#f0ad4e")
    plt.title("Classification Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[*] Training history plot saved to: {output_path}")


def evaluate_and_generate_reports(
    model: keras.Model,
    test_ds: tf.data.Dataset,
    class_names: List[str],
    reports_dir: str,
) -> Dict[str, Any]:
    """
    Evaluate trained model on the held-out test partition.
    Computes Accuracy, Precision, Recall, Macro F1, and saves Confusion Matrix.
    """
    print("\n" + "=" * 60)
    print("[*] Evaluating Model on Test Dataset...")
    print("=" * 60)

    y_true_list = []
    y_pred_list = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        y_true_list.extend(labels.numpy())
        y_pred_list.extend(pred_labels)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    # Core evaluation metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    print(f"[*] Test Accuracy:    {acc * 100:.2f}%")
    print(f"[*] Test Precision:   {macro_prec * 100:.2f}% (Macro)")
    print(f"[*] Test Recall:      {macro_rec * 100:.2f}% (Macro)")
    print(f"[*] Test Macro F1:    {macro_f1 * 100:.2f}%")

    # Detailed classification report
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print("\nDetailed Classification Report:\n")
    print(report_text)

    # Save classification report text
    report_file = os.path.join(reports_dir, "classification_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[*] Classification report written to: {report_file}")

    # Metrics JSON
    metrics_summary = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "total_test_samples": int(len(y_true)),
    }
    metrics_json_file = os.path.join(reports_dir, "metrics.json")
    with open(metrics_json_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[*] Summary metrics written to: {metrics_json_file}")

    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(18, 16))
    sns.heatmap(
        cm,
        annot=False,  # 38x38 is too crowded for text numbers; use clean color mapping
        cmap="Greens",
        xticklabels=[c.split("___")[-1][:12] for c in class_names],
        yticklabels=[c.split("___")[-1][:12] for c in class_names],
    )
    plt.title("Confusion Matrix - PlantVillage 38 Classes (MobileNetV2)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[*] Confusion matrix plot saved to: {cm_path}")

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="AI Lens - Plant Leaf Disease Detection Model Training")
    parser.add_argument("--data_dir", type=str, default="", help="Path to local PlantVillage directory")
    parser.add_argument("--use_hf", action="store_true", help="Download dataset programmatically from Hugging Face")
    parser.add_argument("--hf_dataset", type=str, default="mohanty/PlantVillage", help="Hugging Face dataset identifier")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Initial Adam learning rate")
    parser.add_argument("--fine_tune", action="store_true", help="Perform second stage fine-tuning of base model layers")
    parser.add_argument("--fine_tune_epochs", type=int, default=10, help="Epochs for fine-tuning stage")
    parser.add_argument("--output_model", type=str, default="models/plant_disease_mobilenetv2.keras", help="Path to save model")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Directory to save evaluation reports and plots")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("      AI LENS - PLANT DISEASE CLASSIFICATION TRAINING")
    print("=" * 60)

    # 1. Hardware verification
    check_compute_devices()

    os.makedirs(os.path.dirname(args.output_model) or ".", exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    # 2. Dataset loading
    if args.use_hf or not args.data_dir:
        if args.data_dir and os.path.exists(args.data_dir):
            train_ds, val_ds, test_ds, class_names = load_dataset_from_directory(
                args.data_dir, batch_size=args.batch_size
            )
        else:
            print("[*] No local data directory provided. Attempting Hugging Face loading...")
            train_ds, val_ds, test_ds, class_names = load_dataset_from_huggingface(
                args.hf_dataset, batch_size=args.batch_size
            )
    else:
        train_ds, val_ds, test_ds, class_names = load_dataset_from_directory(
            args.data_dir, batch_size=args.batch_size
        )

    # Save canonical class names mapping
    class_names_path = os.path.join(os.path.dirname(args.output_model), "class_names.json")
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"[*] Class names mapping saved to: {class_names_path}")

    # 3. Model construction
    print("\n[*] Initializing MobileNetV2 Transfer Learning Architecture...")
    model = build_mobilenetv2_model(num_classes=len(class_names), learning_rate=args.lr)
    model.summary()

    # 4. Training Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=args.output_model,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 5. Phase 1: Train classification head
    print(f"\n[*] Commencing Phase 1 Training (Frozen Base) for up to {args.epochs} epochs...")
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    combined_history = {k: list(v) for k, v in history_phase1.history.items()}

    # 6. Optional Phase 2: Fine-Tuning
    if args.fine_tune:
        print("\n[*] Commencing Phase 2 Fine-Tuning: Unfreezing top MobileNetV2 layers...")
        base_mobilenet = model.get_layer("mobilenetv2_1.00_224") if "mobilenetv2_1.00_224" in [l.name for l in model.layers] else model.layers[2]
        base_mobilenet.trainable = True

        # Freeze the first 100 layers and fine-tune remaining top layers
        for layer in base_mobilenet.layers[:100]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.lr / 10.0),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
        )
        for k, v in history_phase2.history.items():
            if k in combined_history:
                combined_history[k].extend(v)

    # 7. Plot and record training curves
    history_plot_path = os.path.join(args.reports_dir, "training_history.png")
    plot_training_history(combined_history, history_plot_path)

    # 8. Evaluation on Held-Out Test Set
    evaluate_and_generate_reports(model, test_ds, class_names, args.reports_dir)

    # Ensure best model is saved
    model.save(args.output_model)
    print(f"\n[✓] Final model successfully saved to: {args.output_model}")
    print("[✓] AI Lens training pipeline completed successfully!\n")


if __name__ == "__main__":
    main()
