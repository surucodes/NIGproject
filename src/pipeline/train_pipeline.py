import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("Starting training pipeline...")

            # Step 1: Data ingestion
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()

            # Step 2: Data transformation
            train_arr, test_arr, _ = self.data_transformation.initiate_data_transformation(train_data_path, test_data_path)

            # Step 3: Model training
            r2_score = self.model_trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info(f"Training pipeline completed successfully with R2 score: {r2_score}")

        except Exception as e:
            raise CustomException(e, sys)