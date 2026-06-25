import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as np
from data_loader import download_stock_data
from features import engineer_features

def evaluate(model, X_test, y_test, modelname):
    """Evaluates model performance by confusion matrix, classification report, AUC-ROC, baseline comparison and feature importance."""
    #Creates confusion matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f"{modelname} Confusion Matrix")
    plt.savefig(f"../outputs/{modelname}_confusion.png")
    plt.close()

    #Prints classification report
    print(classification_report(y_test, y_pred))

    #Finds and prints baseline accuracy for always predicting price rise
    baseline = [1] * len(y_test)
    baseline_accuracy = accuracy_score(y_test, baseline)
    print(f"Baseline accuracy: {baseline_accuracy}")

    #Prints AUC-ROC score
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"AUC-ROC: {auc}")

    # Feature importance
    if hasattr(model.best_estimator_.named_steps['model'], 'feature_importances_'):
        importances = model.best_estimator_.named_steps['model'].feature_importances_
        features = [col[0] if isinstance(col, tuple) else col for col in X_test.columns]
        indices = np.argsort(importances)[::-1]

        plt.figure()
        plt.bar(range(len(features)), importances[indices])
        plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
        plt.title(f"{modelname} Feature Importance")
        plt.tight_layout()
        plt.savefig(f"../outputs/{modelname}_feature_importance.png")
        plt.close()

def evaluate_generalisation(model, modelname, start_date, end_date):
    """Tests model accuracy across a few large companies to assess generalisation beyond the training ticker."""
    stocks = ["AMZN", "AAPL", "NVDA", "MSFT"]
    for ticker in stocks:
        df = download_stock_data(ticker, start_date, end_date)
        df = engineer_features(df)
        X = df.drop(columns=["Target"])
        y = df["Target"]
        y_pred = model.predict(X)
        score = accuracy_score(y, y_pred)
        print(f"{modelname} achieved {round(score * 100, 2)}% accuracy on {ticker} stock")

if __name__ == "__main__":
    from data_loader import download_stock_data
    from features import engineer_features
    from model import create_model
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from config import XGB_PARAMS_AMZN, RF_PARAMS_AMZN

    df = download_stock_data("AMZN", "2015-01-01", "2024-01-01")
    df = engineer_features(df)

    xgb_result, X_test, y_test = create_model(df,XGBClassifier(random_state=42, eval_metric="logloss"), XGB_PARAMS_AMZN, "XGBoost", use_wandb=False)
    evaluate(xgb_result, X_test, y_test, "XGBoost")

    evaluate_generalisation(xgb_result, "XGBoost")

    rf_result, X_test, y_test = create_model(df, RandomForestClassifier(random_state=42), RF_PARAMS_AMZN,"Random Forest", use_wandb=False)
    evaluate(rf_result, X_test, y_test, "Random Forest")

    evaluate_generalisation(rf_result, "Random Forest")