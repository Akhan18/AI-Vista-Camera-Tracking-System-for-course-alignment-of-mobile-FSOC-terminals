"""
generate_yolo_dataset.py

Generates a synthetic YOLO dataset for the AI-VISTA optical beacon.

The dataset contains:
- dark backgrounds
- randomly positioned beacons
- different beacon sizes
- brightness variation
- Gaussian noise
- low-light examples
- blur
- small camera-like disturbances

YOLO annotation format:

class x_center y_center width height

All coordinates are normalized to 0..1.
"""

from pathlib import Path
import random

import cv2
import numpy as np


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

TRAIN_IMAGES = 1200
VAL_IMAGES = 300

SEED = 42

OUTPUT_DIR = Path("data/yolo_beacon")


def make_directories():

    for split in ["train", "val"]:

        (OUTPUT_DIR / "images" / split).mkdir(
            parents=True,
            exist_ok=True,
        )

        (OUTPUT_DIR / "labels" / split).mkdir(
            parents=True,
            exist_ok=True,
        )


def add_gaussian_noise(image):

    noise = np.random.normal(
        0,
        random.uniform(3, 20),
        image.shape,
    )

    noisy = image.astype(np.float32) + noise

    return np.clip(
        noisy,
        0,
        255,
    ).astype(np.uint8)


def create_image():

    background_value = random.randint(
        0,
        20,
    )

    image = np.full(
        (
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            3,
        ),
        background_value,
        dtype=np.uint8,
    )

    beacon_width = random.randint(
        6,
        20,
    )

    beacon_height = random.randint(
        6,
        20,
    )

    center_x = random.randint(
        beacon_width,
        IMAGE_WIDTH - beacon_width,
    )

    center_y = random.randint(
        beacon_height,
        IMAGE_HEIGHT - beacon_height,
    )

    brightness = random.randint(
        150,
        255,
    )

    x1 = int(
        center_x - beacon_width / 2
    )

    y1 = int(
        center_y - beacon_height / 2
    )

    x2 = int(
        center_x + beacon_width / 2
    )

    y2 = int(
        center_y + beacon_height / 2
    )

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (
            brightness,
            brightness,
            brightness,
        ),
        thickness=-1,
    )

    # Optional blur
    if random.random() < 0.25:

        image = cv2.GaussianBlur(
            image,
            (3, 3),
            0,
        )

    # Optional low-light condition
    if random.random() < 0.25:

        factor = random.uniform(
            0.25,
            0.7,
        )

        image = np.clip(
            image.astype(np.float32)
            * factor,
            0,
            255,
        ).astype(np.uint8)

    # Optional Gaussian noise
    if random.random() < 0.5:

        image = add_gaussian_noise(
            image
        )

    return image, (
        center_x,
        center_y,
        beacon_width,
        beacon_height,
    )


def write_label(
    label_path,
    bbox,
):

    center_x, center_y, width, height = bbox

    normalized_x = center_x / IMAGE_WIDTH
    normalized_y = center_y / IMAGE_HEIGHT

    normalized_width = width / IMAGE_WIDTH
    normalized_height = height / IMAGE_HEIGHT

    with open(
        label_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "0 "
            f"{normalized_x:.6f} "
            f"{normalized_y:.6f} "
            f"{normalized_width:.6f} "
            f"{normalized_height:.6f}\n"
        )


def generate_split(
    split,
    count,
):

    print(
        f"Generating {count} {split} images..."
    )

    for index in range(count):

        image, bbox = create_image()

        image_path = (
            OUTPUT_DIR
            / "images"
            / split
            / f"{split}_{index:05d}.png"
        )

        label_path = (
            OUTPUT_DIR
            / "labels"
            / split
            / f"{split}_{index:05d}.txt"
        )

        cv2.imwrite(
            str(image_path),
            image,
        )

        write_label(
            label_path,
            bbox,
        )

    print(
        f"{split} generation complete."
    )


def write_dataset_yaml():

    yaml_text = f"""path: {OUTPUT_DIR.resolve()}
train: images/train
val: images/val

names:
  0: beacon
"""

    yaml_path = (
        OUTPUT_DIR
        / "beacon.yaml"
    )

    yaml_path.write_text(
        yaml_text,
        encoding="utf-8",
    )

    print(
        f"Dataset configuration written to: "
        f"{yaml_path}"
    )


def main():

    random.seed(SEED)
    np.random.seed(SEED)

    make_directories()

    generate_split(
        "train",
        TRAIN_IMAGES,
    )

    generate_split(
        "val",
        VAL_IMAGES,
    )

    write_dataset_yaml()

    print()
    print(
        "YOLO beacon dataset generation complete."
    )


if __name__ == "__main__":
    main()