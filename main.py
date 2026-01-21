import yaml
import logging
from pathlib import Path
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from src.feature_engineering import FeatureEngineering
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Starting ML Pipeline...")
    
    # 1. Data Ingestion
    data_ingestion = DataIngestion(config)
    df = data_ingestion.load_data()
    train_df, test_df = data_ingestion.split_data(df, config['preprocessing']['test_size'])
    
    # 2. Data Preprocessing
    preprocessor = DataPreprocessing()
    train_df = preprocessor.handle_missing_values(train_df)
    test_df = preprocessor.handle_missing_values(test_df)
    
    target_col = config['target_column']
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    X_train = preprocessor.encode_categorical(X_train, categorical_cols, fit=True)
    X_test = preprocessor.encode_categorical(X_test, categorical_cols, fit=False)
    
    # 3. Feature Engineering
    feature_eng = FeatureEngineering()
    X_train = feature_eng.create_features(X_train)
    X_test = feature_eng.create_features(X_test)
    
    X_train_scaled = preprocessor.scale_features(X_train, fit=True)
    X_test_scaled = preprocessor.scale_features(X_test, fit=False)
    
    # 4. Model Training
    trainer = ModelTrainer(config)
    model = trainer.train(X_train_scaled, y_train, model_name='random_forest')
    
    Path(config['model_path']).mkdir(exist_ok=True)
    trainer.save_model(model, f"{config['model_path']}/model.pkl")
    
    # 5. Model Evaluation
    evaluator = ModelEvaluator()
    metrics, y_pred = evaluator.evaluate(model, X_test_scaled, y_test)
    
    logger.info(f"Model Metrics: {metrics}")
    evaluator.plot_confusion_matrix(y_test, y_pred, 
                                   save_path=f"{config['model_path']}/confusion_matrix.png")
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()
