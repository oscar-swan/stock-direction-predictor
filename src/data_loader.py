import yfinance as yf
import os


def download_stock_data(ticker, start_date, end_date):
    """Downloads historical stock data from Yahoo Finance and saves it as a CSV."""
    print(f"Downloading {ticker} data...")
    df = yf.download(ticker, start=start_date, end=end_date)
    os.makedirs("../data", exist_ok=True)
    filepath = f"../data/{ticker}_{start_date}_{end_date}.csv"
    df.to_csv(filepath)
    print(f"Saved to {filepath}")
    return df


if __name__ == "__main__":
    df = download_stock_data("AMZN", "2015-01-01", "2024-01-01")
    print(df.head())
