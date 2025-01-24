from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

if __name__ == "__main__":
    # Perform data ingestion
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    # Perform data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = (
        data_transformation.initiate_data_transformation(train_data, test_data)
    )

    logging.info("Data ingestion and transformation completed.")
    logging.info(f"Preprocessor saved at: {preprocessor_path}")

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr)
