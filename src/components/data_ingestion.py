# this is the Data Ingestion 

# Read the data from the data source split the data into train and test 


import os
import sys 
from exception import CustomeException
from logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# used for class variable at higher level 
from dataclasses import dataclass

from components import data_transformation
from components.data_transformation import DataTransformationConfig
from components.data_transformation import DataTransformation


from components.model_trainer import ModelTrainer
from components.model_trainer import ModelTrainerConfig
from components import model_trainer

# directly define the class variable withput using init 
@dataclass
class DataIngestionConfig:
    
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"data.csv")



class DataIngestion:
    def __init__(self) -> None:
        # this is the object with the above clas we need to save these path variables 
        self.ingestion_config = DataIngestionConfig()

    # read the data from the database 
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try :
            # read the data from the database
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read the data set as data frame")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data ingestion completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                )
        except Exception as e:
            raise CustomeException(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)   

    ModelTrainer = ModelTrainer()
    print(ModelTrainer.initiate_model_trainer(train_arr,test_arr))
   









