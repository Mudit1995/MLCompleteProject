import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
from dataclasses import dataclass


from utils import save_obj, evaluate_models
from exception import CustomeException
from logger import logging

import os
import sys

@dataclass 
class  ModelTrainerConfig:
    trained_model_file_path = os.path.join("artificats","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split traning and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )
            models = { "LinearRegression":LinearRegression(),
                        "Lasso":Lasso(),
                        "Ridge":Ridge(),
                        "KNN":KNeighborsRegressor(),
                        "DecisionTree":DecisionTreeRegressor(),
                        "RandomForest":RandomForestRegressor(),
                        "XGBRegressor":XGBRegressor(),
                        "CatBoosting Regressor":CatBoostRegressor(verbose=0),
                        "AdaBoost Regressor":AdaBoostRegressor()  
                        
                    }

            model_report:dict=evaluate_models(x_train,y_train,x_test,y_test,models=models)
            ## To get best model score from dict
            best_model_score = max(sorted (model_report.values()))
            ## To get best model name from dict
            best_model_name = list (model_report.keys() ) [
            list (model_report.values()).index(best_model_score)
              ]


            best_model = models[best_model_name]


            if best_model_score<0.6:
                raise Exception("Model score is less than 0.6")
            logging.info("best model found ")
            logging.info(f"best model name is {best_model_name}")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2 = r2_score(y_test,predicted)
            return r2  

        except Exception as e:
            raise CustomeException(e,sys)


