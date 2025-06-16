import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os

# Define the directories to read and save data
PREPARED_DATA_DIR = "./data/prepared"
MODEL_DIR = "./models"
PREDICTIONS_DIR = "./predictions"


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_data():

    # Check if the prepared data directory exists
    if not os.path.exists(PREPARED_DATA_DIR):
        print("Prepared data directory not found.")
        return

    # Load the prepared data from CSV file
    if not os.path.exists(os.path.join(PREPARED_DATA_DIR, "X_train_scaled.csv")) or \
       not os.path.exists(os.path.join(PREPARED_DATA_DIR, "y_train_scaled.csv")) or \
       not os.path.exists(os.path.join(PREPARED_DATA_DIR, "X_test_scaled.csv")) or \
       not os.path.exists(os.path.join(PREPARED_DATA_DIR, "y_test_scaled.csv")):
        print("Prepared data files not found.")
        return

    X_train = pd.read_csv(os.path.join(
        PREPARED_DATA_DIR, "X_train_scaled.csv")).drop("Date", axis=1).values
    y_train = pd.read_csv(os.path.join(
        PREPARED_DATA_DIR, "y_train_scaled.csv")).drop("Date", axis=1).values
    X_test = pd.read_csv(os.path.join(
        PREPARED_DATA_DIR, "X_test_scaled.csv")).drop("Date", axis=1).values
    y_test = pd.read_csv(os.path.join(
        PREPARED_DATA_DIR, "y_test_scaled.csv")).drop("Date", axis=1).values

    # Replace 0s y with NaN
    y_train[y_train == 0] = np.nan
    y_test[y_test == 0] = np.nan

    print("Loaded prepared data for training and testing.")

    return X_train, y_train, X_test, y_test


def create_model(train_shape, output_shape):
    # Define a custom masked MSE loss function for stocks with non trading days

    def masked_mse(y_true, y_pred):
        mask = ~tf.math.is_nan(y_true)
        y_true = tf.where(mask, y_true, 0.0)
        y_pred = tf.where(mask, y_pred, 0.0)
        mse = tf.reduce_sum(tf.square(y_true - y_pred) * tf.cast(mask,
                            tf.float32)) / tf.reduce_sum(tf.cast(mask, tf.float32))
        return mse

    # Define a simple Keras model

    model = keras.Sequential([
        keras.Input(shape=(train_shape,)),  # Explicit Input layer
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(32, activation='relu'),
        # Output layer with neurons equal to the number of stocks
        keras.layers.Dense(output_shape)
    ])

    model.compile(optimizer='adam', loss=masked_mse, metrics=[masked_mse])

    print("Model created")

    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=32):

    print("Starting model training...")

    # Train the model
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test)
    )

    print("Model training completed")

    return model


def save_model(model):

    # Save the model
    create_directory(MODEL_DIR)

    try:
        model.save(os.path.join(MODEL_DIR, "stock_price_predictor.keras"))
        print("Model saved in keras format.")
    except Exception as e:
        print(f"Failed to save model: {e}")


def main():
    # Load the prepared data
    X_train, y_train, X_test, y_test = load_data()

    if X_train is None or y_train is None or X_test is None or y_test is None:
        print("Data loading failed. Exiting.")
        return

    # Create the model
    model = create_model(X_train.shape[1], y_train.shape[1])

    # Train the model
    model = train_model(model, X_train, y_train, X_test, y_test)

    # Save the trained model
    save_model(model)

    print("Model training and saving completed.")


if __name__ == "__main__":
    main()
