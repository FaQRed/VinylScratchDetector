import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


CSV_PATH = '../data/extracted_features.csv'
RF_MODEL_PATH = '../models/random_forest_model/rf_model.pkl'
SVM_MODEL_PATH = '../models/svm_model/svm_model.pkl'
SCALER_PATH = '../models/random_forest_model/scaler.pkl'


def plot_performance():
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: File {CSV_PATH} not found.")
        return

    X = df.select_dtypes(include=[np.number]).drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    try:
        rf_model = joblib.load(RF_MODEL_PATH)
        svm_model = joblib.load(SVM_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError as e:
        print(f"Error while loading models: {e}")
        return


    X_test_scaled = scaler.transform(X_test)


    rf_preds = rf_model.predict(X_test_scaled)
    svm_preds = svm_model.predict(X_test_scaled)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion Matrix for Random Forest
    cm_rf = confusion_matrix(y_test, rf_preds)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Grays', ax=ax[0])
    ax[0].set_title('Confusion Matrix: Random Forest')
    ax[0].set_xlabel('Predicted Label (0=Clean, 1=Scratch)')
    ax[0].set_ylabel('True Label')

    # Confusion Matrix for SVM
    cm_svm = confusion_matrix(y_test, svm_preds)
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Purples', ax=ax[1])
    ax[1].set_title('Confusion Matrix: SVM')
    ax[1].set_xlabel('Predicted Label (0=Clean, 1=Scratch)')
    ax[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig('confusion_matrix_comparison(rf_svm).png')


    print("-" * 30)
    print("RANDOM FOREST")
    print("-" * 30)
    print(classification_report(y_test, rf_preds))

    print("-" * 30)
    print("SVM")
    print("-" * 30)
    print(classification_report(y_test, svm_preds))

    print("Visualization saved!")


if __name__ == "__main__":
    plot_performance()