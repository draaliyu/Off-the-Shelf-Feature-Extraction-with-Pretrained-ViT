# feature_extraction.py
from vit_keras import vit
import numpy as np


IMAGE_SIZE = 224


def extract_features(model, dataset):
    features = []
    labels = []
    for images, lbls in dataset:
        feats = model.predict(images)
        features.append(feats)
        labels.append(lbls)
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels


def build_feature_extractor(num_classes):
    vit_model = vit.vit_b32(
        image_size=IMAGE_SIZE,
        activation='softmax',
        pretrained=True,
        include_top=False,
        pretrained_top=False,
        classes=num_classes
    )
    return vit_model


