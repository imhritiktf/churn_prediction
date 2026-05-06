import sys
import os
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split    
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")
    raw_data_path:str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or commponent")
        try:
            df = pd.read_csv(os.path.join("notebooks/data", "WA_Fn-UseC_-Telco-Customer-Churn.csv"))
            logging.info("Read the dataset as dataframe")

            # TotalCharges mein ' ' spaces hain — NaN mein badlo
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            logging.info(f"TotalCharges nulls after fix: {df['TotalCharges'].isna().sum()}")

            df.drop(columns=["customerID"], inplace=True, errors="ignore")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise(CustomException(e, sys))
        
