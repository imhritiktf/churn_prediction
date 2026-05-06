import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            # neg/pos ratio — XGBoost ke liye
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()

            models = {
                "Logistic Regression": LogisticRegression(
                    class_weight="balanced", max_iter=1000
                ),
                "Decision Tree": DecisionTreeClassifier(
                    class_weight="balanced"
                ),
                "Random Forest": RandomForestClassifier(
                    class_weight="balanced", random_state=42
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    random_state=42
                ),
                "K-Neighbors": KNeighborsClassifier(),
                "XGBoost": XGBClassifier(
                    scale_pos_weight=neg / pos,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1
                ),
                "LightGBM": LGBMClassifier(
                    class_weight="balanced",
                    random_state=42,
                    verbose=-1
                ),
                "AdaBoost": AdaBoostClassifier(
                    random_state=42
                )
            }

            params = {
                "Logistic Regression": {
                    "C":        [0.01, 0.1, 1, 10],
                    "l1_ratio": [0, 1],    # 0 = l2, 1 = l1
                    "solver":   ["saga"]   # saga dono support karta hai
                },
                
                "Decision Tree": {
                    "max_depth":        [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "criterion":        ["gini", "entropy"]
                },
                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth":    [5, 10, None],
                    "max_features": ["sqrt", "log2"]
                },
                "Gradient Boosting": {
                    "n_estimators":  [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth":     [3, 5, 7]
                },
                "K-Neighbors": {
                    "n_neighbors": [3, 5, 7, 11],
                    "weights":     ["uniform", "distance"]
                },
                "XGBoost": {
                    "n_estimators":  [100, 200, 300],
                    "max_depth":     [3, 5, 7],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "subsample":     [0.7, 0.8, 1.0],
                    "colsample_bytree": [0.7, 0.8, 1.0]
                },
                "LightGBM": {
                    "n_estimators":  [100, 200, 300],
                    "max_depth":     [3, 5, -1],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "num_leaves":    [31, 63]
                },
                "AdaBoost": {
                    "n_estimators":  [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.5, 1.0]
                }
            }

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train,
                                               X_test=X_test, y_test=y_test,
                                                  models=models, params=params)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name  = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if(best_model_score < 0.75):
                raise CustomException("No best model found — ROC-AUC below 0.75", sys)
            
            logging.info(
                f"Best model: {best_model_name} | ROC-AUC: {best_model_score:.4f}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Trained model artifact file saved")
            
            # Final test score
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            final_roc_auc  = roc_auc_score(y_test, y_pred_proba)
            logging.info(f"Final test ROC-AUC score: {final_roc_auc :.4f}")

            return final_roc_auc 

        except Exception as e:
            raise CustomException(e, sys)