# Stock Direction Predictor
Binary classification model to predict stock price direction from historical data using sklearn and Weights &amp; Biases

## What this project predicts
Whether a stock's closing price will be higher or lower at the close of the next market day as a binary classification (Up or down from previous day)

## Why this project is interesting
I am interested in financial decisions and how data can be used to identify market trends. Stock prices are hard to predict as they can be influenced by unpredictable real world factors making machine learning more difficult.

## How success is measured
Model accuracy, precision and recall, AUC-ROC. Model will be compared against a baseline of always predicting a stock to rise since a stock price is more likely to rise than fall.

## Dataset
Yahoo Finance historical price data sourced via the yfinance Python library.
Ticker and date range to be confirmed.

## Design decisions
I am going to begin by building a model based around the prediction of amazon's stock price, my reasoning is that since it is a large diverse company that has been Mag 7 for a long time, it should have a more stable and more predictable stock price compared to a company like Nvidia which may be unstable and influenced a lot by real world trends.
I will use stock price data from 2015 to now, capturing the last 10 years as it includes a variety of market conditions such as a normal market, COVID crash and recovery, AI boom and crashes due to tariff announcements and trade tensions.
I am going to create moving averages of stock prices to filter out the noise and capture an overall trend. I will use both 7 and 14 day moving averages as that will capture an entire week rather than splitting weeks up which may result in some moving averages including a larger amount of closed market days such as the weekend. This ensures each moving average contains the same amount of open days. I will also capture a 7 day momentum change for the same reasons. Calculating volatility, volume change and RSI will also give the model more data to work with to improve accuracy. 
Finally a boolean target variable will be 0 or 1 if the next day's price is higher or not.