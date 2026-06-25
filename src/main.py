from data_loader import download_stock_data
from features import engineer_features
from model import create_model
from evaluate import evaluate, evaluate_generalisation
from config import XGB_PARAMS_SEARCH, RF_PARAMS_SEARCH
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def main():
    """Main function. Handles all user input and orchestrates the execution."""
    #Gather user input
    ticker = input("Enter ticker of stock you want to train model on: ")
    default_dates = input("Do you want to select a specific date range for the dataset? (y/n) ")
    if default_dates.upper() == "Y":
        start_date = input("Enter start date (YYYY-MM-DD): ")
        end_date = input("Enter end date (YYYY-MM-DD): ")
    else:
        start_date = "2015-01-01"
        end_date = "2024-01-01"
    wandb_choice = input("Use wandb to track performance? (y/n) ")
    if wandb_choice.upper() == "Y":
        wandb_choice = True
    else:
        wandb_choice = False
    model_eval_choice = input("Evaluate model performance? (y/n) ")
    model_general_choice = input("Test generalisation across other stocks? (y/n) ")

    #Process user input
    df = download_stock_data(ticker.upper(), start_date, end_date)
    df = engineer_features(df)
    xgb_result, X_test_XGB, y_test_XGB = create_model(df, XGBClassifier(random_state=42, eval_metric="logloss"),XGB_PARAMS_SEARCH, "XGBoost", use_wandb=wandb_choice)
    rf_result, X_test_RF, y_test_RF = create_model(df, RandomForestClassifier(random_state=42),RF_PARAMS_SEARCH, "Random Forest", use_wandb=wandb_choice)
    if model_eval_choice.upper() == "Y":
        evaluate(xgb_result, X_test_XGB, y_test_XGB, "XGBoost")
        evaluate(rf_result, X_test_RF, y_test_RF, "Random Forest")
    if model_general_choice.upper() == "Y":
        evaluate_generalisation(xgb_result, "XGBoost", start_date, end_date)
        evaluate_generalisation(rf_result, "Random Forest", start_date, end_date)

if __name__ == "__main__":
    main()