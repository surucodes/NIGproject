import os 
import sys 
import dill
import numpy as np 
import pandas as pd 
from src.exception import CustomException
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.model_selection import RandomizedSearchCV
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path , exist_ok= True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(x_train, y_train, x_test, y_test, models, param, feature_names=None):
    try:
        report = {}
        best_models = {}

        for model_name, model in models.items():
            logging.info(f"Evaluating {model_name}")
            if model_name in param and param[model_name]:  # Check if model has hyperparameters
                # Use RandomizedSearchCV for efficiency
                search = RandomizedSearchCV(
                    model,
                    param_distributions=param[model_name],
                    n_iter=5 if model_name == "Stacking Regressor" else 10,
                    scoring='neg_mean_absolute_error',  # Prioritize MAE for clinical accuracy
                    cv=5,  # 5-fold cross-validation
                    n_jobs=-1,
                    random_state=42
                )
                search.fit(x_train, y_train)
                best_model = search.best_estimator_
                best_params = search.best_params_
                logging.info(f"Best parameters for {model_name}: {best_params}")
            else:
                # Fit models without hyperparameters directly
                best_model = model
                best_model.fit(x_train, y_train)
                best_params = {}
                logging.info(f"No hyperparameter tuning for {model_name}")

            # Evaluate on test set
            y_test_pred = best_model.predict(x_test)
            r2 = r2_score(y_test, y_test_pred)
            mae = mean_absolute_error(y_test, y_test_pred)
            mse = mean_squared_error(y_test, y_test_pred)
            report[model_name] = {'r2': r2, 'mae': mae, 'mse': mse}
            best_models[model_name] = best_model 
            logging.info(f"{model_name} - Test R2: {r2}, MAE: {mae}, MSE: {mse}")

            # Log feature importance for tree-based models
            # if hasattr(best_model, 'feature_importances_'):
            #     importance = best_model.feature_importances_
            #     feature_names_list = feature_names if feature_names else range(x_train.shape[1])
            #     logging.info(f"Feature Importance for {model_name}: {dict(zip(feature_names_list, importance))}")

        return report, best_models

    except Exception as e:
        raise CustomException(e, sys)