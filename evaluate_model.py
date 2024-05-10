
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def evaluate_classifier(clf, features, labels, classes, save_path):
    labels = np.argmax(labels, axis=1)  # Convert one-hot to single class labels
    predictions = clf.predict(features)
    pred_probs = clf.predict_proba(features)

    # Confusion Matrix
    conf_matrix = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes.values(),
                yticklabels=classes.values())
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

    # ROC Curve and AUC
    if len(classes) == 2:  # Binary classification
        fpr, tpr, _ = roc_curve(labels, pred_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(save_path, 'roc_curve.png'))
        plt.close()

    # Classification Report
    report = classification_report(labels, predictions, target_names=classes.values(), output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Precision, Recall, F1-score, Specificity
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=None,
                                                               labels=np.arange(len(classes)))

    metrics = pd.DataFrame({
        'Class': list(classes.values()),
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

    # Save metrics to CSV
    metrics.to_csv(os.path.join(save_path, 'metrics.csv'), index=False)
    report_df.to_csv(os.path.join(save_path, 'classification_report.csv'))


