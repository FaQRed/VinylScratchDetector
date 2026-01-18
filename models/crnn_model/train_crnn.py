import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

CNN_MAP_PATH = '../../cnn_dataset_map.csv'
MODEL_NAME = 'vinyl_crnn_model.keras'
BATCH_SIZE = 32
EPOCHS = 50


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = [np.load(path) for path in df['npy_path']]
    return np.expand_dims(np.array(X), axis=-1), df['label'].values.astype('float32')


def build_crnn(input_shape):
    reg = regularizers.l2(0.001)

    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Start 96x188
        # CNN block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.2),
        # 96/2 = 48, 188/2 = 94
        layers.MaxPooling2D((2, 2)),

        # CNN block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.2),
        # 48/2 = 24, 94/2 = 47
        layers.MaxPooling2D((2, 2)),

        # Reshape for RNN
        layers.Reshape((47, 24 * 64)),

        # RNN block
        layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.4)),
        layers.Bidirectional(layers.LSTM(32, dropout=0.4)),

        # Dense classifier
        layers.Dense(64, activation='relu', kernel_regularizer=reg),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    return model


def train():
    print("Loading data...")
    X, y = load_data(CNN_MAP_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    # Rozmiar spektrogram
    model = build_crnn(input_shape=(96, 188, 1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    model_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint(MODEL_NAME, monitor='val_accuracy', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]

    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=model_callbacks,
        verbose=1
    )


    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.savefig('crnn_training_plots.png')

    print("\nClassification Report:")
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    report = classification_report(y_test, y_pred, target_names=['Clean', 'Scratch'])
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Clean', 'Scratch'],
                yticklabels=['Clean', 'Scratch'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('crnn_confusion_matrix.png')
    plt.show()
    print("Confusion matrix saved.")


if __name__ == "__main__":
    train()