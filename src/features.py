import ta

def engineer_features(df):
    #Add new features
    df['MA7'] = df['Close'].rolling(7).mean()
    df['MA14'] = df['Close'].rolling(14).mean()
    df['Momentum7'] = df['Close'].diff(7)
    df['Volatility'] = df['Close'].rolling(7).std()
    df['VolumeChange'] = df['Volume'].pct_change()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'].squeeze(), window=14).rsi()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    #Drop old features and entries without complete data
    df.dropna(inplace=True)
    df.drop(columns=["Open", "High", "Low", "Close", "Volume"], inplace=True)

    return df

if __name__ == "__main__":
    #Test function works using data_loader.py
    from data_loader import download_stock_data
    df = download_stock_data("AMZN", "2015-01-01", "2024-01-01")
    df = engineer_features(df)
    print(df.head())
    print(df.shape)