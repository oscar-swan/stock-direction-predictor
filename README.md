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