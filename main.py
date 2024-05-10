# main.py
import os
import pickle
from load_data import load_data
from processing import create_datasets
from feature_extraction import build_feature_extractor, extract_features
from train_model import train_classifier
from evaluate_model import evaluate_classifier
from train_model import save_model

DATASET_PATH = 'dataset'
SAVE_PATH = './results'
MODEL_SAVE_PATH = './model/logistic_regression_model.pkl'


def main(dataset_path, save_path, model_save_path):
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Load data
    X_train, X_val, y_train, y_val, classes = load_data(dataset_path)

    # Create datasets
    train_dataset, val_dataset = create_datasets(X_train, X_val, y_train, y_val, len(classes))

    # Build feature extractor
    feature_extractor = build_feature_extractor(len(classes))

    # Extract features
    train_features, train_labels = extract_features(feature_extractor, train_dataset)
    val_features, val_labels = extract_features(feature_extractor, val_dataset)

    # Train classifier
    is_multiclass = len(classes) > 2
    classifier = train_classifier(train_features, train_labels, is_multiclass)

    # Save the trained classifier model
    save_model(classifier, model_save_path)

    # Evaluate classifier
    evaluate_classifier(classifier, val_features, val_labels, classes, save_path)


if __name__ == '__main__':
    main(DATASET_PATH, SAVE_PATH, MODEL_SAVE_PATH)
