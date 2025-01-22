import os
import sys
from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils.file_utils import save_object
from src.utils.ml_utils import evaluate_models

# from xgboost.sklearn import XGBRegressor


@dataclass
class ModelTrainerConfig:
    """
    Configuration class for ModelTrainer.
    Defines paths and directories used during model training and evaluation.
    """

    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    catboost_training_dir: str = os.path.join("logs", "catboost_logs")


class ModelTrainer:
    """
    Handles the training, evaluation, and saving of the best-performing machine learning model.
    """

    def __init__(self):
        """
        Initializes the ModelTrainer class with the configuration.
        Ensures required directories are created.
        """
        self.model_trainer_config = ModelTrainerConfig()
        os.makedirs(self.model_trainer_config.catboost_training_dir, exist_ok=True)

    def initiate_model_trainer(
        self, train_array: np.ndarray, test_array: np.ndarray
    ) -> float:
        """
        Trains multiple models, evaluates their performance, and saves the best model.

        Args:
            train_array (np.ndarray): Training data (features and target combined).
            test_array (np.ndarray): Testing data (features and target combined).

        Returns:
            float: R2 score of the best-performing model on the test dataset.

        Raises:
            CustomException: If no model achieves a minimum performance threshold.
        """
        try:
            logging.info("Splitting training and testing input data.")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features for training
                train_array[:, -1],  # Target for training
                test_array[:, :-1],  # Features for testing
                test_array[:, -1],  # Target for testing
            )

            # Define models and their hyperparameters
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(
                    verbose=False,
                    train_dir=self.model_trainer_config.catboost_training_dir,
                ),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            # Evaluate all models
            logging.info(
                "Evaluating models with the provided training and testing data."
            )
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            # Get the best model and its performance
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logging.info(
                f"Best model: {best_model_name} with R2 score: {best_model_score:.4f}"
            )

            # Check if the best model meets the performance threshold
            if best_model_score < 0.6:
                logging.error("No model met the minimum performance threshold.")
                raise CustomException("No best model found", sys)

            # Save the best model
            logging.info("Saving the best model.")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            # Evaluate the best model on the test set
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"R2 score of the best model on the test set: {r2_square:.4f}")

            return r2_square

        except Exception as e:
            logging.error("Error in model training: %s", e)
            raise CustomException(e, sys) from e
