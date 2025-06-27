import os
import sys
from dataclasses import dataclass
import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object , evaluate_model
from src.components.HyperParameterTuning import HyperParameter
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array ,test_array ,metric = 'mae'):
        try:
            logging.info("Splitting Training and Testing input data")
            x_train , y_train , x_test , y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(random_state=42),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
                "MLP Regressor": MLPRegressor(max_iter=1000, random_state=42),
                "Stacking Regressor": StackingRegressor(
                    estimators=[
                        ('rf', RandomForestRegressor(random_state=42)),
                        ('xgb', XGBRegressor(random_state=42)),
                        ('cat', CatBoostRegressor(verbose=False, random_state=42))
                    ],
                    final_estimator=LinearRegression()
                )
            }
            hyper = HyperParameter()
            model_report , best_models = hyper.initiate_hyperparametertuning(X_train = x_train ,y_train = y_train ,X_test = x_test,y_test = y_test , models = models)
            # Select best model based on R2 score
            if metric == 'mae':
                best_model_score = min(model_report[model_name]['mae'] for model_name in model_report)
                best_model_name = next(model_name for model_name in model_report if model_report[model_name]['mae'] == best_model_score)
                threshold = 15  # MAE threshold in mg/dL
                if best_model_score > threshold:
                    raise CustomException(f"No best model found with MAE <= {threshold}, best MAE: {best_model_score}", sys)
            else:  # Default to r2
                best_model_score = max(model_report[model_name]['r2'] for model_name in model_report)
                best_model_name = next(model_name for model_name in model_report if model_report[model_name]['r2'] == best_model_score)
                if best_model_score < 0.6:
                    raise CustomException(f"No best model found with R2 score >= 0.6, best R2: {best_model_score}", sys)
                
            best_model = best_models[best_model_name]
            logging.info(f"Best model: {best_model_name} with R2: {model_report[best_model_name]['r2']}, MAE: {model_report[best_model_name]['mae']}, MSE: {model_report[best_model_name]['mse']}")
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Evaluate additional metrics
            r2 = model_report[best_model_name]['r2']
            mae = model_report[best_model_name]['mae']
            mse = model_report[best_model_name]['mse']
            logging.info(f"Test set metrics - R2: {r2}, MAE: {mae}, MSE: {mse}")
            return [r2,mae,mse]

        except Exception as e:
            raise CustomException(e, sys)