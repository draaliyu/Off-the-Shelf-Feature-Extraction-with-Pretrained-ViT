# inference.py
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import joblib
from vit_keras import vit
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

IMAGE_SIZE = 224
MODEL_SAVE_PATH = './model/logistic_regression_model.pkl'
CLASSES = {0: "Class 0", 1: "Class 1", 2: "Class 2", 3: "Class 3"}  # Update this based on your actual class names


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


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


def extract_features(model, image):
    features = model.predict(image)
    return features


def load_model(model_path):
    return joblib.load(model_path)


def make_prediction(image_path, model_path, classes):
    # Load and preprocess the image
    image = load_image(image_path)

    # Build feature extractor
    feature_extractor = build_feature_extractor(len(classes))

    # Extract features
    features = extract_features(feature_extractor, image)

    # Load the classifier model
    classifier = load_model(model_path)

    # Make prediction
    prediction = classifier.predict(features)
    prediction_prob = classifier.predict_proba(features)

    predicted_class = classes[prediction[0]]
    confidence = prediction_prob[0][prediction[0]]

    return predicted_class, confidence


def main():
    # Create a Tkinter root window
    root = Tk()
    root.withdraw()  # Hide the root window

    # Open a file dialog to select an image
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )

    if image_path:
        predicted_class, confidence = make_prediction(image_path, MODEL_SAVE_PATH, CLASSES)
        print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")

        # Display the image
        image = Image.open(image_path)
        plt.imshow(image)
        plt.title(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")
        plt.axis('off')
        plt.show()
    else:
        print("No image selected.")


if __name__ == '__main__':
    main()
