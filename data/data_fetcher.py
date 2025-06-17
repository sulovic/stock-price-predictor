import os
import yfinance as yf
import wbdata as wbd
import pandas as pd

# Define the directory to save the fetched data
RAW_DATA_DIR = "./data/raw"


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

        economics_data = economics_data.sort_index(ascending=True)

        economics_data["Population growth (%)"] = economics_data["Population"].pct_change(
            fill_method=None) * 100

        economics_data["Export growth (%)"] = economics_data["Exports (current US$)"].pct_change(
            fill_method=None) * 100

        economics_data["Import growth (%)"] = economics_data["Imports (current US$)"].pct_change(
            fill_method=None) * 100

        print(f"Successfully fetched data for {tickers.values()}.")
        return economics_data
    except Exception as e:
        print(
            f"An error occurred while fetching data for {tickers.values()}: {e}")
        return None


def fetch_market_sentiment(start_date, end_date, interval="1mo", ticker="^GSPC"):

    # Fetch data for preparing market sentiment data from Yahoo Finance.

    try:
        print(
            f"Fetching  data for market sentiment {ticker} from {start_date} to {end_date}...")
        market_sentiment_data = yf.download(
            "^GSPC", start_date, end_date, interval)

        # Calculate price change percentage
        market_sentiment_data['PriceChange%'] = (
            market_sentiment_data['Close'] - market_sentiment_data['Open']) / market_sentiment_data['Open'] * 100

        # Add a moving average of volume (7-days in this example)
        market_sentiment_data['AvgVolume'] = market_sentiment_data['Volume'].rolling(
            window=7, min_periods=1).mean()

        # Calculate volume change percentage
        market_sentiment_data[('VolumeChange%', '')] = (market_sentiment_data[(
            'Volume', '^GSPC')] - market_sentiment_data[('AvgVolume', '')]) / market_sentiment_data[('AvgVolume', '')] * 100

        # Drop rows with missing values
        market_sentiment_data.dropna(inplace=True)

        # Calculate Sentiment score based on volume and price chanege - Price change more weight
        market_sentiment_data['SentimentScore'] = market_sentiment_data['PriceChange%'] * \
            3 + market_sentiment_data['VolumeChange%']

        # Determine sentiment label
        market_sentiment_data['Sentiment'] = pd.cut(market_sentiment_data['SentimentScore'],
                                                    bins=[-float('inf'), -5,
                                                          8, float('inf')],
                                                    labels=['Bearish', 'Neutral', 'Bullish'])

        market_sentiment_data.columns = market_sentiment_data.columns.get_level_values(
            0)

        return market_sentiment_data
    except Exception as e:
        print(f"An error occurred while fetching market sentiment data: {e}")
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
    os.makedirs(os.path.dirname(RAW_DATA_DIR), exist_ok=True)

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
    interval = "1d"

    stock_data = fetch_data(stock_tickers, start_date, end_date, interval)

    economics_data = fetch_economics_data(
        economy_tickers, start_date, end_date)

    market_sentiment = fetch_market_sentiment(start_date, end_date, interval)

    if stock_data is None:
        print("Cannot fetch stock data...")
        return

    if economics_data is None:
        print("Cannot fetch economics data...")
        return

    if market_sentiment is None:
        print("Cannot fetch market sentiment data...")
        return

    save_data_to_csv(stock_data, "Stock_data")

    save_data_to_csv(economics_data, "Economics_data")

    save_data_to_csv(market_sentiment, "Market_sentiment")


if __name__ == "__main__":
    main()
