import os 
import sys 
import dill
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomeException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomeException(e,sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            try:
                para = param.get(model_name, {})

                # Perform Grid Search
                gs = GridSearchCV(estimator=model, param_grid=para, cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)

                # Use the best model from GridSearchCV
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train) 

                # Train the model
                model.fit(X_train, y_train)

                # Predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Calculate scores
                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)

                # Store the test score in the report
                report[model_name] = test_model_score

            except Exception as model_error:
                report[model_name] = f"Error: {model_error}"
                continue

        return report

    except Exception as e:
        raise CustomeException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomeException(e, sys)
