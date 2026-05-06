import sys
import os
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path        = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model        = load_object(model_path)

            # Student project jaisa — transform karo phir predict
            data_scaled = preprocessor.transform(features)
            pred        = model.predict(data_scaled)

            return pred

        except Exception as e:
            raise CustomException(e, sys)
        
    def predict_proba(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path        = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model        = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            # [:, 1] = Churn=Yes ki probability
            proba = model.predict_proba(data_scaled)[:, 1]
            return proba

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 gender: str,
                 SeniorCitizen: int,
                 Partner: str,
                 Dependents: str,
                 tenure: int,
                 PhoneService: str,
                 MultipleLines: str,
                 InternetService: str,
                 OnlineSecurity: str,
                 OnlineBackup: str,
                 DeviceProtection: str,
                 TechSupport: str,
                 StreamingTV: str,
                 StreamingMovies: str,
                 Contract: str,
                 PaperlessBilling: str,
                 PaymentMethod: str,
                 MonthlyCharges: float,
                 TotalCharges: float):

        self.gender           = gender
        self.SeniorCitizen    = SeniorCitizen
        self.Partner          = Partner
        self.Dependents       = Dependents
        self.tenure           = tenure
        self.PhoneService     = PhoneService
        self.MultipleLines    = MultipleLines
        self.InternetService  = InternetService
        self.OnlineSecurity   = OnlineSecurity
        self.OnlineBackup     = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport      = TechSupport
        self.StreamingTV      = StreamingTV
        self.StreamingMovies  = StreamingMovies
        self.Contract         = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod    = PaymentMethod
        self.MonthlyCharges   = MonthlyCharges
        self.TotalCharges     = TotalCharges

    def get_data_as_dataframe(self):
        try:
            # dict → DataFrame
            custom_data_input_dict = {
                "gender":           [self.gender],
                "SeniorCitizen":    [self.SeniorCitizen],
                "Partner":          [self.Partner],
                "Dependents":       [self.Dependents],
                "tenure":           [self.tenure],
                "PhoneService":     [self.PhoneService],
                "MultipleLines":    [self.MultipleLines],
                "InternetService":  [self.InternetService],
                "OnlineSecurity":   [self.OnlineSecurity],
                "OnlineBackup":     [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport":      [self.TechSupport],
                "StreamingTV":      [self.StreamingTV],
                "StreamingMovies":  [self.StreamingMovies],
                "Contract":         [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod":    [self.PaymentMethod],
                "MonthlyCharges":   [self.MonthlyCharges],
                "TotalCharges":     [self.TotalCharges],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        
