import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomeException
from src.loggers import logging

from src.utils import save_object,evaluate_models


@dataclass
class Modeltrainerconfig:

    trained_model_file_path = os.path.join('artifacts','model.pkl')

class Modeltrainer:

    def __init__(self):
        self.model_trainer_config= Modeltrainerconfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                # Parameter dictionary remains unchanged
            }

            # Evaluate models
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                models=models, param=params)

            # Filter valid scores
            valid_scores = {key: value for key, value in model_report.items() if isinstance(value, (int, float)) and value is not None}

            if not valid_scores:
                raise CustomeException("No valid model scores found")

            # Get the best model and its score
            best_model_score = max(valid_scores.values())
            best_model_name = max(valid_scores, key=valid_scores.get)
            best_model = models[best_model_name]

            logging.info(f"Best model selected: {best_model_name} with R2 score: {best_model_score}")

            # Refit the best model to ensure it's trained
            logging.info(f"Fitting the best model: {best_model_name}")
            best_model.fit(X_train, y_train)

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict and calculate R2 score on test data
            try:
                predicted = best_model.predict(X_test)
            except Exception as e:
                raise CustomeException(f"Model '{best_model_name}' is not fitted: {str(e)}")

            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomeException(e, sys)

