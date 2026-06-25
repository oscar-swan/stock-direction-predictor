from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import wandb
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit

def create_model(df, model, param_grid, modelname, use_wandb=True):
    """Creates a model using a date ordered train and test split and GridSearchCV with TimeSeriesSplit, logs results to W&B if enabled."""
    #Adds model to wandb with a timestamp for identification
    if use_wandb:
        run_name = f"{modelname}_{datetime.now().strftime('%m%d_%H%M')}"
        wandb.init(project="stock-direction-predictor", name=run_name)
    #Creates a model
    print("Fitting model...")
    X = df.drop(columns=["Target"])
    y = df["Target"]

    #Splits data so training data includes oldest 80% of data so only future dates are predicted
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    #Preprocessor
    preprocessor = StandardScaler()

    #Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    #Grid search and cv value
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring="accuracy")
    grid_search.fit(X_train, y_train)

    #Display best parameters and model scores
    print(f"{modelname} model:")
    print("Best params:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    print("CV score:", grid_search.score(X_test, y_test))

    #Adds to wandb log if enabled and closes the run.
    if use_wandb:
        wandb.log({
            "best_score": grid_search.best_score_,
            "test_score": grid_search.score(X_test, y_test),
            "best_params": grid_search.best_params_
        })

        wandb.finish()

    return grid_search, X_test, y_test



if __name__ == "__main__":
    from data_loader import download_stock_data
    from features import engineer_features
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    df = download_stock_data("AMZN", "2015-01-01", "2024-01-01")
    df = engineer_features(df)

    #Parameter selections
    xgb_params_long = {
        "model__n_estimators": [100, 125, 150, 175, 200],
        "model__max_depth": [2, 3, 4, 5, 6, 7, 8],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.015]
    }

    rf_params_long = {
        "model__n_estimators": [100, 125, 150, 175, 200],
        "model__max_depth": [2, 3, 4, 5, 6, 7, 8]
    }

    xgb_result, X_test, y_test = create_model(df, XGBClassifier(random_state=42, eval_metric="logloss"), xgb_params_long, "XGBoost", use_wandb=False)
    rf_result, X_test, y_test = create_model(df, RandomForestClassifier(random_state=42), rf_params_long, "Random Forest", use_wandb=False)
