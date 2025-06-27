import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  
from sklearn.impute import SimpleImputer  
from sklearn.pipeline import Pipeline  
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from src.utils import save_object  
from src.exception import CustomException  
from src.logger import logging 
@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):

        try :

            numerical_columns = ['age', 'bmi', 'sleep_duration', 'blood_pressure_systolic',
                'blood_pressure_diastolic', 'heart_rate', 'skin_temperature',
                'sensor_current_ua']
            nominal_columns= ['sex', 'ethnicity', 'socioeconomic_status','physical_activity', 'alcohol_consumption', 'smoking_status', 'family_history_diabetes', 'prediabetes', 'gestational_diabetes','hypertension', 'pcos', 'cardiovascular_disease', 'kidney_problems', 'medications']
            ordinal_columns = {
                'sleep_quality':['Poor','Average', 'Good'],
                'stress_levels':['Low', 'Moderate', 'High'],
                'bmi_category':['Underweight', 'Normal', 'Overweight', 'Obese']
            }

# Custom transformer to convert categorical to string as ordinal encoding requires the data type to be string.
            class CategoricalToString(BaseEstimator, TransformerMixin):
                def __init__(self, columns):
                    self.columns = columns

                def fit(self, X, y=None):
                    return self

                def transform(self, X):
                    X_copy = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.columns)
                    for col in self.columns:
                        if col in X_copy.columns:
                            X_copy[col] = X_copy[col].astype(str)
                    return X_copy


            class RareLabelGrouper(BaseEstimator, TransformerMixin):
                def __init__(self, threshold=0.01, columns=None):
                    self.threshold = threshold
                    self.columns = columns
                    self.rare_labels_ = {}

                def fit(self, X, y=None):
                    X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.columns)
                    for col in self.columns:
                        freq = X_df[col].value_counts(normalize=True)
                        self.rare_labels_[col] = [label for label in freq[freq < self.threshold].index if label != 'Other']
                        if self.rare_labels_[col]:
                            logging.info(f"Rare labels for {col}: {self.rare_labels_[col]}")
                    return self

                def transform(self, X):
                    X_copy = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.columns)
                    for col in self.columns:
                        X_copy[col] = X_copy[col].apply(lambda x: 'Other' if x in self.rare_labels_[col] else x)
                        if pd.api.types.is_categorical_dtype(X_copy[col]):
                            new_categories = [cat for cat in X_copy[col].cat.categories
                                            if cat not in self.rare_labels_[col]] + ['Other']
                            new_categories = list(dict.fromkeys(new_categories))
                            X_copy[col] = X_copy[col].cat.set_categories(new_categories)
                    return X_copy

                
                #Pipeline to transform numerical feratures
            num_pipeline = Pipeline(steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ])
            ordinal_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('to_string', CategoricalToString(columns=list(ordinal_columns.keys()))),
                        # Ordinal encoder requires the dtype to be string
                    ('ordinal', OrdinalEncoder(
                         categories = [ordinal_columns[col] for col in ordinal_columns],
                         handle_unknown= 'use_encoded_value',
                        unknown_value=-1
                            # impute missing ordinal values and encodes them as integers, handling category data type.
                    ))
                ]
            )
            nominal_pipeline= Pipeline(
                 steps= [
                    ('imputer',SimpleImputer(strategy = 'most_frequent')),
                    ('to_string', CategoricalToString(columns=nominal_columns)),
                    ('rare_label_grouper', RareLabelGrouper(threshold = 0.01,columns=nominal_columns)),
                    ('onehot', OneHotEncoder(handle_unknown= 'ignore'))
                 ])
            
            preprocessor = ColumnTransformer(transformers=[
                ('num',num_pipeline, numerical_columns),
                ('ord', ordinal_pipeline, list(ordinal_columns.keys())),
                ('cat', nominal_pipeline, nominal_columns)
            ])

            logging.info("Preprocessing object created successfully")
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read the train and test data from ingestion module successfully.")
            train_df[train_df.select_dtypes(include='object').columns] = train_df.select_dtypes(include='object').astype('category')
            test_df[test_df.select_dtypes(include='object').columns] = test_df.select_dtypes(include='object').astype('category')
            # Convert to category dtype as it is memory efficient.
            train_df.columns = train_df.columns.str.lower()
            test_df.columns = test_df.columns.str.lower()
            logging.info("Converted categorical columns to category dtype")
            logging.info("Obtaining the preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'blood_glucose'
            input_feature_train_df = train_df.drop(columns = [target_column_name], axis= 1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("features and target separated")
            logging.info("Applying preprocessing object on training and testing dataset:")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr , np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr , np.array(target_feature_test_df)]
            logging.info("Combined transformed features with target")

            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path ,
                obj = preprocessing_obj
            )

            logging.info("Data transformation completed")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path
            )
    
        except Exception as e:
                raise CustomException(e,sys)