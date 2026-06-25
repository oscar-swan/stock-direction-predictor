#Parameters for finding the best parameters on a new stock
XGB_PARAMS_SEARCH = {
    "n_estimators": [100, 125, 150, 175, 200],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1]
}

RF_PARAMS_SEARCH = {
    "n_estimators": [100, 125, 150, 175, 200],
    "max_depth": [2, 3, 4, 5, 6],
}

#Best parameters for AMZN
XGB_PARAMS_AMZN = {
    "n_estimators": [100],
    "max_depth": [4],
    "learning_rate": [0.01]
}

RF_PARAMS_AMZN = {
    "n_estimators": [150],
    "max_depth": [3]
}

