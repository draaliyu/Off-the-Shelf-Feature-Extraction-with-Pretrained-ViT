
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression


def train_classifier(features, labels, is_multiclass=True):
    labels = np.argmax(labels, axis=1)  # Convert one-hot to single class labels
    clf = LogisticRegression(multi_class='ovr' if is_multiclass else 'auto', max_iter=1000)
    clf.fit(features, labels)
    return clf


def save_model(clf, path):
    joblib.dump(clf, path)