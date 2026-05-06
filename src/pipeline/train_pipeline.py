# src/pipeline/train_pipeline.py
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    # Step 1
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Step 2
    transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
        train_path, test_path
    )

    # Step 3
    trainer = ModelTrainer()
    score = trainer.initiate_model_trainer(train_arr, test_arr)

    print(f"\nFinal Best Model ROC-AUC: {score:.4f}")