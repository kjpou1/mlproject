import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging
from src.utils.file_utils import save_object


@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation.
    Defines the file path for saving the preprocessor object.
    """

    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    Handles data transformation, including preprocessing, feature scaling,
    and combining input and target arrays.
    """

    def __init__(self):
        """
        Initializes the DataTransformation class with the configuration.
        """
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Creates a preprocessing pipeline for both numerical and categorical columns.

        Numerical columns:
            - Imputes missing values using the median strategy.
            - Scales the data using StandardScaler.

        Categorical columns:
            - Imputes missing values using the most frequent value strategy.
            - Encodes the data using OneHotEncoder.
            - Scales the encoded data using StandardScaler (without centering).

        Returns:
            ColumnTransformer: A transformer object that applies the preprocessing steps.
        """
        try:
            # Define numerical and categorical columns for preprocessing
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Define the pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),  # Handle missing values
                    ("scaler", StandardScaler()),  # Scale numerical data
                ]
            )

            # Define the pipeline for categorical features
            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),  # Handle missing values
                    ("one_hot_encoder", OneHotEncoder()),  # Encode categorical data
                    ("scaler", StandardScaler(with_mean=False)),  # Scale encoded data
                ]
            )

            # Log the preprocessing setup
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine pipelines for numerical and categorical features
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            logging.error("Error in get_data_transformer_object: %s", e)
            # Handle exceptions with a custom error class
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Reads train and test datasets, applies preprocessing, and saves the preprocessor object.

        Args:
            train_path (str): Path to the training dataset.
            test_path (str): Path to the testing dataset.

        Returns:
            tuple: Transformed training array, testing array, and the path to the saved preprocessor object.
        """
        try:
            # Read train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Get the preprocessing object
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Define target column and input features
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Split input features and target variable for train and test datasets
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Transform input features using the preprocessing pipeline
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine input features and target variable into single arrays
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            # Return the transformed datasets and preprocessor file path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.error("Error in get_data_transformer_object: %s", e)
            # Handle exceptions with a custom error class
            raise CustomException(e, sys) from e


# if __name__ == "__main__":
#     """
#     Entry point for data ingestion and transformation.
#     """
#     # Perform data ingestion
#     obj = DataIngestion()
#     train_data, test_data = obj.initiate_data_ingestion()

#     # Perform data transformation
#     data_transformation = DataTransformation()
#     train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
#         train_data, test_data
#     )
