import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

#intialize the Data Ingestion configuration
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')


# Create the data ingestion class
class DataIngestion:
    def __init__(self):
        self.config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')
        
        try:
            df = pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))
            logging.info('Dataset read as pandas Dataframe')
            
            os.makedirs(os.path.dirname(self.config.raw_data_path),exist_ok=True)
            df.to_csv(self.config.raw_data_path,index=False)
            
            logging.info('Raw data is created')
            
            train_set,test_set = train_test_split(df,test_size=0.3,random_state=42)
            
            train_set.to_csv(self.config.train_data_path,index=False,header=True)
            test_set.to_csv(self.config.test_data_path,index=False,header=True)
            
            logging.info('ingestion of data is completed')
            
            return(
                self.config.train_data_path,
                self.config.test_data_path
            )
        
        except Exception as e:
            logging.info('Exception occured at Data ingestion stage')
            raise CustomException(e,sys)
        
    