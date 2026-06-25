#Parameters for finding the best parameters on a new stock
XGB_PARAMS_SEARCH = {
    "model__n_estimators": [100, 125, 150, 175, 200],
    "model__max_depth": [2, 3, 4, 5, 6],
    "model__learning_rate": [0.01, 0.05, 0.1]
}

RF_PARAMS_SEARCH = {
    "model__n_estimators": [100, 125, 150, 175, 200],
    "model__max_depth": [2, 3, 4, 5, 6],
}

#Best parameters for AMZN
XGB_PARAMS_AMZN = {
    "model__n_estimators": [125],
    "model__max_depth": [2],
    "model__learning_rate": [0.01]
}

RF_PARAMS_AMZN = {
    "model__n_estimators": [125],
    "model__max_depth": [2]
}

