import os
import yfinance as yf
import wbdata as wbd
import pandas as pd

# Define the directory to save the fetched data
RAW_DATA_DIR = "./data/raw"


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def fetch_data(tickers, start_date, end_date, interval="1mo"):

    # Fetch stock data for a given ticker from Yahoo Finance.

    try:
        print(
            f"Fetching data for {tickers} from {start_date} to {end_date}...")
        stock_data = yf.download(
            tickers, start=start_date, end=end_date, interval=interval)['Close']

        if stock_data.empty:
            print(f"No data found for {tickers}.")
            return None

        print(f"Successfully fetched data for {tickers}.")
        return stock_data
    except Exception as e:
        print(f"An error occurred while fetching data for {tickers}: {e}")
        return None


def fetch_economics_data(tickers, start_date, end_date, country="USA", freq="M"):

    # Fetch economics data from the World Bank Data API.

    try:
        print(
            f"Fetching data for {tickers.values()} from {start_date} to {end_date}...")
        economics_data: pd.DataFrame = wbd.get_dataframe(
            tickers, country, date=(start_date[:4], end_date[:4]))

        if economics_data.empty:
            print(f"No data found for {tickers.values()}.")
            return None

        print(f"Successfully fetched data for {tickers.values()}.")
        return economics_data
    except Exception as e:
        print(
            f"An error occurred while fetching data for {tickers.values()}: {e}")
        return None


def save_data_to_csv(dataframe, file_name):

    # Save the fetched data to a CSV file.

    try:
        file_name = f"{file_name}.csv"
        file_path = os.path.join(RAW_DATA_DIR, file_name)
        dataframe.to_csv(file_path)
        print(f"Data saved to {file_path}.")
    except Exception as e:
        print(f"An error occurred while saving data: {e}")


def main():

    # Ensure raw data directory exists
    create_directory(RAW_DATA_DIR)

    # Define tickers and date range
    stock_tickers = ["^GSPC", "^TNX", "^DJI", "GC=F",
                     "CL=F", "BTC-USD", "AAPL", "GOOGL", "MSFT"]
    economy_tickers = {"NY.GDP.MKTP.CD": "GDP (current US$)",
                       "NY.GDP.MKTP.KD.ZG": "GDP growth rate (%)",
                       "SL.UEM.TOTL.ZS": "Unemployment rate (%)",
                       "FR.INR.RINR": "Real interest rate (%)",
                       "NY.GDP.DEFL.KD.ZG": "Inflation rate (%)",
                       "SP.POP.TOTL": "Population",
                       "NE.EXP.GNFS.CD": "Exports (current US$)",
                       "NE.IMP.GNFS.CD": "Imports (current US$)",
                       }
    start_date = "2001-01-01"
    end_date = "2024-01-01"
    interval = "1mo"

    stock_data = fetch_data(stock_tickers, start_date, end_date, interval)

    economics_data = fetch_economics_data(
        economy_tickers, start_date, end_date)

    if stock_data is None:
        print("Cannot fetch stock data...")
        return

    if economics_data is None:
        print("Cannot fetch economics data...")
        return

    save_data_to_csv(stock_data, "Stock_data")

    save_data_to_csv(economics_data, "Economics_data")


if __name__ == "__main__":
    main()
