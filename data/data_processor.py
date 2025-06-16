import os
import pandas as pd


# Define the directories to fetch and save data
RAW_DATA_DIR = "./data/raw"
PROCESSED_DATA_DIR = "./data/processed"


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_raw_data():

    # Process raw data by merging stock and economics data.

    if not os.path.exists(RAW_DATA_DIR):
        print("Raw data directory not found.")
        return

    if not os.path.exists(os.path.join(RAW_DATA_DIR, "Stock_data.csv")):
        print("Stock data not found.")
        return

    if not os.path.exists(os.path.join(RAW_DATA_DIR, "Economics_data.csv")):
        print("Economics data not found.")
        return

    if not os.path.exists(os.path.join(RAW_DATA_DIR, "Market_sentiment.csv")):
        print("Sentiment data not found.")
        return

    # Load the data from CSV files.

    stock_data = pd.read_csv(os.path.join(
        RAW_DATA_DIR, "Stock_data.csv"), index_col="Date")
    economics_data = pd.read_csv(os.path.join(
        RAW_DATA_DIR, "Economics_data.csv"), index_col="date")
    sentiment_data = pd.read_csv(os.path.join(
        RAW_DATA_DIR, "Market_sentiment.csv"), index_col="Date")

    if stock_data.empty or economics_data.empty or sentiment_data.empty:
        print("No data found.")
        return

    economics_data.index.name = stock_data.index.name
    economics_data.index = pd.to_datetime(economics_data.index, format='%Y')
    stock_data.index = pd.to_datetime(stock_data.index, format='%Y-%m-%d')
    sentiment_data.index = pd.to_datetime(
        sentiment_data.index, format='%Y-%m-%d')

    economics_daily = economics_data.resample(
        'D').interpolate(method='linear').ffill()
    stocks_daily = stock_data.resample('D').last().ffill()

    merged_data = pd.merge(economics_daily, stocks_daily,
                           left_index=True, right_index=True, how='left')

    # Interpolate missing values, fill with 0 for periods with no trading

    merged_data['BTC-USD'] = merged_data['BTC-USD'].fillna(0)
    merged_data['GOOGL'] = merged_data['GOOGL'].fillna(0)
    merged_data['CL=F'] = merged_data['CL=F'].interpolate(method='linear')
    merged_data['GC=F'] = merged_data['GC=F'].interpolate(method='linear')

    # Drop rows with any remaining NaN values

    # Merge sentiment data with the merged stock and economics data
    merged_data = pd.merge(merged_data, sentiment_data,
                           left_index=True, right_index=True, how='left')
    merged_data = merged_data.fillna(method='ffill')

    merged_data = merged_data.dropna()

    if merged_data.isnull().any().any():
        print("Warning: Missing data detected after processing.")

    return merged_data


def save_data_to_csv(dataframe, file_name):

    # Save the processed data to a CSV file.

    try:
        file_name = f"{file_name}.csv"
        file_path = os.path.join(PROCESSED_DATA_DIR, file_name)
        dataframe.to_csv(file_path)
        print(f"Data saved to {file_path}.")
    except Exception as e:
        print(f"An error occurred while saving data: {e}")


def main():

    # Ensure processed data directory exists
    create_directory(PROCESSED_DATA_DIR)

    processed_data = process_raw_data()

    save_data_to_csv(processed_data, "Processed_data")


if __name__ == "__main__":
    main()
