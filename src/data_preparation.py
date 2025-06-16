import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Define the directories to read and save data
PROCESSED_DATA_DIR = "./data/processed"
PREPARED_DATA_DIR = "./data/prepared"


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def prepare_data():

    # Check if the processed data directory exists
    if not os.path.exists(PROCESSED_DATA_DIR):
        print("Processed data directory not found.")
        return

    # Check if the processed data file exists
    processed_data_path = os.path.join(
        PROCESSED_DATA_DIR, "Processed_data.csv")
    if not os.path.exists(processed_data_path):
        print("Processed data file not found.")
        return

    # Load the processed data from CSV file

    df = pd.read_csv('data/processed/Processed_data.csv',
                     index_col='Date', parse_dates=True)

    print("Loaded processed data, preparing for training and testing...")

    # Check if the DataFrame is empty
    if df.empty:
        print("Processed data is empty.")
        return

    # Map sentiment strings to numerical values
    sentiment_map = {'Bearish': -1, 'Neutral': 0, 'Bullish': 1}
    df['SentimentNum'] = df['Sentiment'].map(sentiment_map)

    # New column for date with Epoch time in seconds
    df['Date'] = df.index.astype('int64') / 10**9

    # Select features from Processed_data

    X = df[["Date", "GDP growth rate (%)", "Unemployment rate (%)", "Real interest rate (%)",
            "Inflation rate (%)", "Population growth (%)", "Export growth (%)", "SentimentScore", "SentimentNum"]]

    # Select target variables from Processed_data and replace 0 with NaN for training and testing

    y = df[['AAPL', 'BTC-USD', 'GOOGL', 'MSFT']].replace(0, np.nan)

    # Split the data into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42)

    # Scale the features using MinMaxScaler

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_scaled = scaler.fit_transform(y_train)
    y_test_scaled = scaler.transform(y_test)

   # Ensure the prepared data directory exists
    create_directory(PREPARED_DATA_DIR)

    # Convert numpy arrays to pandas dataframes and Save the training and testing sets to CSV files
    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=X.columns, index=X_test.index)
    y_train_scaled = pd.DataFrame(
        y_train_scaled, columns=y.columns, index=y_train.index)
    y_test_scaled = pd.DataFrame(
        y_test_scaled, columns=y.columns, index=y_test.index)

    # Sort and save the data to CSV files
    try:
        X_train_scaled.sort_index().to_csv(
            os.path.join(PREPARED_DATA_DIR, "X_train_scaled.csv"))
        X_test_scaled.sort_index().to_csv(
            os.path.join(PREPARED_DATA_DIR, "X_test_scaled.csv"))
        y_train_scaled.sort_index().to_csv(
            os.path.join(PREPARED_DATA_DIR, "y_train_scaled.csv"))
        y_test_scaled.sort_index().to_csv(
            os.path.join(PREPARED_DATA_DIR, "y_test_scaled.csv"))
        y_test.sort_index().to_csv(os.path.join(PREPARED_DATA_DIR, "y_test_original.csv"))

        # Save the scaler to a file for later use
        joblib.dump(scaler, os.path.join(PREPARED_DATA_DIR, "scaler.pkl"))

    except Exception as e:
        print(f"An error occurred while saving data: {e}")

    print("Data preparation completed successfully. Data saved to prepared directory.")
    print("Scaler saved successfully.")


def main():

    prepare_data()


if __name__ == "__main__":
    main()
