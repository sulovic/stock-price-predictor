import pandas as pd
from tensorflow import keras
import tensorflow as tf
import os
import numpy as np


# Paths
MODEL_PATH = './models/stock_price_predictor_model.keras'
SCALER_X_PATH = './models/scaler_X.pkl'
SCALER_Y_PATH = './models/scaler_y.pkl'
PREDICTIONS_DIR = './data/predictions'


def load_prepare_data():
    # Check if the prepared data directory exists
    if not os.path.exists(os.path.join(PREDICTIONS_DIR, "X_predictions.csv")):
        print("File with input data for predictions not found.")
        return None

    # Load the new input data for predictions
    X_predictions = pd.read_csv(os.path.join(PREDICTIONS_DIR, "X_predictions.csv"), parse_dates=[
        'Date'], index_col='Date')

    # Map sentiment strings to numerical values
    sentiment_map = {'Bearish': -1, 'Neutral': 0, 'Bullish': 1}
    X_predictions['SentimentNum'] = X_predictions['Sentiment'].map(
        sentiment_map)

    # Extract date features from the index
    X_predictions['date'] = X_predictions.index
    X_predictions['year'] = X_predictions['date'].dt.year
    X_predictions['month'] = X_predictions['date'].dt.month
    X_predictions['day'] = X_predictions['date'].dt.day
    X_predictions['day_of_week'] = X_predictions['date'].dt.dayofweek

    # Cyclical encoding of month and day of week
    X_predictions['month_sin'] = np.sin(
        2 * np.pi * X_predictions['month'] / 12)
    X_predictions['month_cos'] = np.cos(
        2 * np.pi * X_predictions['month'] / 12)
    X_predictions['day_of_week_sin'] = np.sin(
        2 * np.pi * X_predictions['day_of_week'] / 7)
    X_predictions['day_of_week_cos'] = np.cos(
        2 * np.pi * X_predictions['day_of_week'] / 7)

    # Recalculate days starting from 1.1.2000
    X_predictions['days_since_start'] = (
        X_predictions['date'] - pd.Timestamp('2000-01-01')).dt.days

    # Select features from Processed_data

    X_predictions_processed = X_predictions[["month_sin", "month_cos", "day_of_week_sin", "day_of_week_cos", "days_since_start", "GDP growth rate (%)", "Unemployment rate (%)", "Real interest rate (%)",
                                             "Inflation rate (%)", "Population growth (%)", "Export growth (%)", "Import growth (%)", "SentimentNum"]]

    print("Loaded new input data for predictions.")
    return X_predictions, X_predictions_processed


def load_model():
    def masked_mse(y_true, y_pred):
        mask = ~tf.math.is_nan(y_true)
        y_true = tf.where(mask, y_true, 0.0)
        y_pred = tf.where(mask, y_pred, 0.0)
        mse = tf.reduce_sum(tf.square(y_true - y_pred) * tf.cast(mask,
                            tf.float32)) / tf.reduce_sum(tf.cast(mask, tf.float32))
        return mse

    # Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        print("Model file not found.")
        return None

    model = keras.models.load_model(MODEL_PATH, custom_objects={
                                    'masked_mse': masked_mse})

    print("Model loaded successfully.")
    return model


def load_scalers():
    import joblib

    if not os.path.exists(SCALER_X_PATH) or not os.path.exists(SCALER_Y_PATH):
        print("Scaler files not found.")
        return None, None

    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    print("Scalers loaded successfully.")
    return scaler_X, scaler_y


def predict(model, scaler_X, scaler_y, X_predictions_processed):

    # Check if the model and scalers are loaded
    if model is None or scaler_X is None or scaler_y is None:
        print("Model or scalers not loaded.")
        return None

    # Scale the new input data
    X_predictions_scaled = scaler_X.transform(X_predictions_processed)

    # Make predictions
    y_predictions_scaled = model.predict(X_predictions_scaled)

    # Inverse transform the predictions
    y_predictions = scaler_y.inverse_transform(y_predictions_scaled)

    return y_predictions


def save_predictions(predictions, X_predictions, filename='Predictions.csv'):

    # Save the predictions to a CSV file
    predictions_df = pd.DataFrame(predictions, index=X_predictions.index, columns=[
                                  'AAPL', 'BTC-USD', 'GOOGL', 'MSFT'])
    predictions_df.to_csv(os.path.join(PREDICTIONS_DIR, filename), index=False)
    print(f"Predictions saved to {filename}.")


def main():
    # Load the new input data for predictions
    X_predictions, X_predictions_processed = load_prepare_data()
    if X_predictions is None or X_predictions_processed is None:
        print("No data to make predictions.")
        return

    # Load the model
    model = load_model()

    # Load the scalers
    scaler_X, scaler_y = load_scalers()

    # Make predictions
    predictions = predict(
        model, scaler_X, scaler_y, X_predictions_processed)

    if predictions is not None:
        # Save the predictions
        save_predictions(predictions, X_predictions)


if __name__ == "__main__":
    main()
