import sys
import os

from imblearn.over_sampling import SMOTE
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        try:
            # SeniorCitizen — already 0/1, kuch nahi karna
            passthrough_cols = ["SeniorCitizen"]

            # Yes/No binary cols — OrdinalEncoder se handle karenge pipeline mein
            binary_cols = [
                "Partner", "Dependents", "PhoneService", "PaperlessBilling"
            ]

            # Gender — Male/Female
            gender_cols = ["gender"]

            # 3+ values — OHE
            multi_cat_cols = [
                "MultipleLines", "InternetService", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
            ]

            # Continuous — impute + scale
            numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

            # Numeric pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  
                    ("scaler", StandardScaler())
                ]
            )

            # Binary pipeline — Yes/No → 0/1 pipeline ke andar hi
            binary_pipeline = Pipeline(
                steps=[
                    ("encoder",OrdinalEncoder(
                    categories=[["No", "Yes"]] * len(binary_cols)
                )),
                ("scaler", StandardScaler())
                    
                ]
            )

            # Gender pipeline — Male/Female → 0/1
            gender_pipeline = Pipeline(steps=[
                ("encoder", OrdinalEncoder(
                    categories=[["Male", "Female"]]
                )),
                ("scaler", StandardScaler())
            ])
            
            # Multi-category pipeline
            cat_pipeline = Pipeline(steps=[
                ("one_hot_encoder", OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False
                ))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num",         num_pipeline,      numerical_cols),
                ("binary",      binary_pipeline,   binary_cols),
                ("gender",      gender_pipeline,   gender_cols),
                ("cat",         cat_pipeline,      multi_cat_cols),
                ("passthrough", "passthrough",     passthrough_cols),
            ])

            logging.info(f"Numerical cols   : {numerical_cols}")
            logging.info(f"Binary cols      : {binary_cols}")
            logging.info(f"Multi-cat cols   : {multi_cat_cols}")
            logging.info(f"Passthrough cols : {passthrough_cols}")

            return preprocessor

        except Exception as e:
            raise(CustomException(e, sys))

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            for df in [train_df, test_df]:
                df['Churn'] = df["Churn"].map({'Yes': 1, 'No': 0})

            logging.info("Target variable encoding completed")

            target_column = "Churn"    

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test  = test_df.drop(columns=[target_column])
            y_test  = test_df[target_column]
            
            logging.info("Splitting input and target feature completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing object on training and testing datasets")

            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr  = preprocessing_obj.transform(X_test)

             # SMOTE — sirf train pe, test pe kabhi nahi
            logging.info(f"Before SMOTE: {X_train_arr.shape} | "
                         f"Churn rate: {y_train.mean():.2%}")
            
            smote = SMOTE(random_state=42)
            X_train_arr, y_train = smote.fit_resample(X_train_arr, y_train)

            logging.info(f"After SMOTE: {X_train_arr.shape} | "
                         f"Churn rate: {y_train.mean():.2%}")
            
            train_arr = np.c_[X_train_arr, y_train]
            test_arr  = np.c_[X_test_arr, y_test]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor artifact file saved")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise(CustomException(e, sys))

