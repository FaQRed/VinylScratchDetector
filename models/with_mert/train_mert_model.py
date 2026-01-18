import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import class_weight


MERT_MAP_PATH = '../../mert_dataset_map.csv'
MODEL_NAME = 'vinyl_mert_classifier.keras'
BATCH_SIZE = 32
EPOCHS = 60


def load_mert_data(csv_path):
    df = pd.read_csv(csv_path)
    X = []
    for path in df['mert_path']:
        X.append(np.load(path))
    return np.array(X).astype('float32'), df['label'].values.astype('float32')


def train():
    X, y = load_mert_data(MERT_MAP_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1536 input size from MERT (768 from the hidden layer and 768 from the last layer)
    model = models.Sequential([
        layers.Input(shape=(1536,)),

        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )

    weights = class_weight.compute_class_weight('balanced', y=y_train)
    class_weights = dict(enumerate(weights))

    сallbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),

        callbacks.ModelCheckpoint(MODEL_NAME, monitor='val_accuracy', save_best_only=True)
    ]


    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=сallbacks,
        verbose=1
    )


    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Training Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train()