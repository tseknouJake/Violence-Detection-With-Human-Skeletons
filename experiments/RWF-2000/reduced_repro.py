import argparse
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


ORIGINAL_FRAMES_PER_VIDEO = 150


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reduced-compute reproduction for RWF-2000 experiments."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("experiments/RWF-2000/datasets/RWF-2000"),
    )
    parser.add_argument(
        "--variant",
        choices=["original", "frame_diff"],
        default="original",
    )
    parser.add_argument("--train-fraction", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experiments/RWF-2000/reduced_repro_results.json"),
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def list_split(split_root: Path):
    video_ids = sorted(str(path) for path in split_root.glob("*/*"))
    labels = {video_id: 0 if "NonFight" in video_id else 1 for video_id in video_ids}
    return video_ids, labels


def stratified_subset(video_ids, labels, fraction: float, seed: int):
    if fraction >= 1.0:
        return list(video_ids)
    rng = random.Random(seed)
    negatives = [video_id for video_id in video_ids if labels[video_id] == 0]
    positives = [video_id for video_id in video_ids if labels[video_id] == 1]
    take_negatives = max(1, math.floor(len(negatives) * fraction))
    take_positives = max(1, math.floor(len(positives) * fraction))
    return sorted(
        rng.sample(negatives, take_negatives) + rng.sample(positives, take_positives)
    )


def load_videos(
    video_ids,
    video_labels,
    video_frames: int,
    video_width: int,
    video_height: int,
    video_channels: int = 3,
    dtype=np.float32,
    normalize: bool = False,
):
    videos = np.empty(
        (len(video_ids), video_frames, video_height, video_width, video_channels),
        dtype=dtype,
    )
    labels = np.empty((len(video_ids),), dtype=np.int8)
    frames_idx = set(
        np.round(np.linspace(0, ORIGINAL_FRAMES_PER_VIDEO - 1, video_frames)).astype(int)
    )

    for i, video_id in enumerate(video_ids):
        cap = cv2.VideoCapture(video_id)
        frames = []
        index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if index in frames_idx:
                frame = cv2.resize(frame, (video_width, video_height)).astype(dtype)
                if normalize:
                    frame /= 255.0
                frames.append(frame)
            index += 1
        cap.release()
        videos[i] = np.array(frames)
        labels[i] = video_labels[video_id]

    return videos, labels


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        video_ids,
        video_labels,
        batch_size,
        video_frames,
        video_width,
        video_height,
        video_channels=3,
        dtype=np.float32,
        normalize=False,
        shuffle=True,
    ):
        self.video_ids = list(video_ids)
        self.video_labels = video_labels
        self.batch_size = batch_size
        self.video_frames = video_frames
        self.video_width = video_width
        self.video_height = video_height
        self.video_channels = video_channels
        self.dtype = dtype
        self.normalize = normalize
        self.shuffle = shuffle

    def __len__(self):
        return len(self.video_ids) // self.batch_size

    def __getitem__(self, idx):
        batch_ids = self.video_ids[idx * self.batch_size : (idx + 1) * self.batch_size]
        return load_videos(
            batch_ids,
            self.video_labels,
            self.video_frames,
            self.video_width,
            self.video_height,
            self.video_channels,
            self.dtype,
            self.normalize,
        )

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.video_ids)


def tf_frame_diff(video):
    return video[1:] - video[:-1]


def build_model(variant: str, frames: int, width: int, height: int, channels: int = 3):
    inputs = tf.keras.layers.Input(shape=(frames, height, width, channels))
    x = inputs

    if variant == "frame_diff":
        x = tf.keras.layers.Lambda(lambda video: tf.map_fn(tf_frame_diff, video))(inputs)

    x = tf.keras.layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        return_sequences=False,
        data_format="channels_last",
        activation="tanh",
    )(x)
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        depth_multiplier=2,
        activation="relu",
        data_format="channels_last",
    )(x)
    x = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")(x)
    x = tf.keras.layers.Dense(units=128, activation="relu")(x)
    x = tf.keras.layers.Dense(units=16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    args = parse_args()
    set_seed(args.seed)

    train_ids, train_labels = list_split(args.dataset_root / "train")
    val_ids, val_labels = list_split(args.dataset_root / "val")

    train_ids = stratified_subset(train_ids, train_labels, args.train_fraction, args.seed)
    val_ids = stratified_subset(val_ids, val_labels, args.val_fraction, args.seed + 1)

    if args.variant == "frame_diff":
        generator_frames = args.frames + 1
    else:
        generator_frames = args.frames

    train_generator = DataGenerator(
        train_ids,
        train_labels,
        batch_size=args.batch_size,
        video_frames=generator_frames,
        video_width=args.width,
        video_height=args.height,
        shuffle=True,
    )
    val_generator = DataGenerator(
        val_ids,
        val_labels,
        batch_size=args.batch_size,
        video_frames=generator_frames,
        video_width=args.width,
        video_height=args.height,
        shuffle=False,
    )

    model = build_model(args.variant, generator_frames, args.width, args.height)

    start_time = time.time()
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=val_generator,
        verbose=2,
    )
    elapsed = time.time() - start_time

    best_val_accuracy = float(max(history.history["val_accuracy"]))
    best_epoch = int(np.argmax(history.history["val_accuracy"]) + 1)

    result = {
        "variant": args.variant,
        "dataset_root": str(args.dataset_root),
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "train_videos": len(train_ids),
        "val_videos": len(val_ids),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "frames": args.frames,
        "generator_frames": generator_frames,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_train_loss": float(history.history["loss"][-1]),
        "trainable_params": int(
            np.sum([np.prod(variable.shape) for variable in model.trainable_weights])
        ),
        "elapsed_seconds": elapsed,
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "physical_devices": [device.name for device in tf.config.list_physical_devices()],
        "gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        "notes": (
            "Reduced-compute run on a stratified subset. "
            "This is not directly comparable to the paper's full-data 90.25% result."
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
