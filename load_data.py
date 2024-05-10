# load_data.py
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 224
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image)
    return image


def load_data(dataset_path):
    class_names = sorted(os.listdir(dataset_path))
    classes = {i: class_name for i, class_name in enumerate(class_names)}

    images = []
    labels = []
    for class_idx, class_name in classes.items():
        class_dir = os.path.join(dataset_path, class_name)
        for image_name in os.listdir(class_dir):
            if image_name.lower().endswith(IMAGE_EXTENSIONS):
                image_path = os.path.join(class_dir, image_name)
                image = load_image(image_path)
                images.append(image)
                labels.append(class_idx)

    images = np.array(images)
    labels = np.array(labels)
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

    return X_train, X_val, y_train, y_val, classes
