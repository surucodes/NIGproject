import os
import sys
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model
class HyperParameterConifg:
    hyperparameters_file_path = os.path.join("artifacts", "hyperparameters.pkl")

class HyperParameter:
    def __init__(self):
        self.HyperParameter_config = HyperParameterConifg()
    def initiate_hyperparametertuning(self, X_train, y_train, X_test, y_test, models):
        try:
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.001, 0.01, 0.05, 0.1],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [3, 5, 7, 10],
                    'min_samples_split': [2, 5, 10]
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'leaf_size': [20, 30, 40],
                    'p': [1, 2]
                },
                "XGBRegressor": {
                    'learning_rate': [0.001, 0.01, 0.05, 0.1],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [3, 5, 7, 10],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.6, 0.8, 1.0]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],
                    'l2_leaf_reg': [1, 3, 5]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.001, 0.01, 0.5, 0.1],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "MLP Regressor": {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'learning_rate_init': [0.001, 0.01, 0.1],
                    'max_iter': [1000],
                    'alpha': [0.0001, 0.001, 0.01]
                },
                "Stacking Regressor": {
                    # Tune base estimators
                    'rf__n_estimators': [50, 100, 200],
                    'rf__max_depth': [None, 10, 20],
                    'xgb__n_estimators': [50, 100, 200],
                    'xgb__learning_rate': [0.01, 0.1],
                    'cat__depth': [6, 8],
                    'cat__learning_rate': [0.01, 0.05]
                }
            }
            model_report, best_models =evaluate_model(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test, models=models,param=params)
            save_object(
                file_path= self.HyperParameter_config.hyperparameters_file_path , 
                obj = model_report
            )
            return model_report , best_models
        except Exception as e:
            raise CustomException(e,sys)