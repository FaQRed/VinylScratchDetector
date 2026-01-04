import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import keras_tuner as kt
import os
from data_loader import get_data


MAP_PATH = '../../cnn_dataset_map.csv'

def build_tuner_model(hp):
    model = models.Sequential()

    model.add(layers.Input(shape=(96, 188, 1)))

    model.add(layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=64, step=32),
        kernel_size=(3, 3),
        activation='relu',
        padding='same'
    ))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())

    if hp.Boolean('extra_conv_layer'):
        model.add(layers.Conv2D(
            filters=hp.Int('conv_2_filter', min_value=64, max_value=128, step=64),
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        ))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())

    model.add(layers.Flatten())

    model.add(layers.Dense(
        units=hp.Int('dense_units', min_value=64, max_value=256, step=64),
        activation='relu'
    ))

    model.add(layers.Dropout())
    model.add(layers.Dense(1, activation='sigmoid'))


    lr = hp.Choice('learning_rate', values=[1e-4, 1e-5])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision')
        ]
    )
    return model


def run_tuner():



    X_train, _, X_val, y_train, _, y_val = get_data(MAP_PATH)


    tuner = kt.Hyperband(
        build_tuner_model,
        objective=kt.Objective("val_recall", direction="max"),
        max_epochs=15,
        factor=3,
        directory='tuner_results',
        project_name='vinyl_scratch_detector',
    )

    stop_early = callbacks.EarlyStopping(monitor='val_recall', patience=5)

    print("Starting hyperparameter search...")
    tuner.search(
        X_train, y_train,
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=[stop_early]
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("-" * 30)
    print("Search complete. Best hyperparameters found:")


    for param, value in best_hps.values.items():
        print(f"  > {param}: {value}")
    print("-" * 30)

    print("Saving the best model from the trials...")
    best_model = tuner.get_best_models(num_models=1)[0]


    model_save_path = 'vinyl_cnn_tuner.keras'
    best_model.save(model_save_path)

    print(f"Model successfully saved to: {model_save_path}")


if __name__ == "__main__":
    run_tuner()