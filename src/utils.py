import os
import sys
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            joblib.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            return joblib.load(f)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            param_dist = params[name]

            if param_dist:  # params hain toh tune karo

                # Total combinations count karo
                total_combinations = 1
                for v in param_dist.values():
                    total_combinations *= len(v)

                # n_iter kabhi total combinations se zyada nahi hoga
                n_iter = min(20, total_combinations)

                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_dist,
                    n_iter=n_iter,
                    scoring="roc_auc",
                    cv=cv,
                    random_state=42,
                    n_jobs=-1,
                    verbose=0
                )
                search.fit(X_train, y_train)
                model.set_params(**search.best_params_)
                logging.info(f"{name} best params: {search.best_params_}")

            # Poore train pe fit karo
            model.fit(X_train, y_train)

            y_test_proba = model.predict_proba(X_test)[:, 1]
            test_score   = roc_auc_score(y_test, y_test_proba)
            report[name] = test_score

            logging.info(f"{name} → ROC-AUC: {test_score:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)