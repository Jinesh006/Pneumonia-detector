"""
train.py
Trains an EfficientNetB0-based pneumonia classifier on chest X-ray images.

Dataset expected structure (Kaggle: paultimothymooney/chest-xray-pneumonia):
chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/

Run: python train.py --data_dir /path/to/chest_xray --epochs 10
"""

import argparse
import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def build_model():
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False  # freeze base for initial training

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation="sigmoid")(x)  # binary: NORMAL vs PNEUMONIA

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def get_generators(data_dir):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,  # X-rays shouldn't be flipped horizontally (orientation matters)
    )
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=["NORMAL", "PNEUMONIA"],
    )
    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=["NORMAL", "PNEUMONIA"],
    )
    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=["NORMAL", "PNEUMONIA"],
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to chest_xray dataset folder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out", type=str, default="pneumonia_model.h5")
    args = parser.parse_args()

    train_gen, val_gen, test_gen = get_generators(args.data_dir)
    model = build_model()
    model.summary()

    callbacks = [
        ModelCheckpoint(args.out, monitor="val_auc", mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print("\nEvaluating on test set...")
    results = model.evaluate(test_gen)
    print(dict(zip(model.metrics_names, results)))

    model.save(args.out)
    print(f"\nModel saved to {args.out}")


if __name__ == "__main__":
    main()
