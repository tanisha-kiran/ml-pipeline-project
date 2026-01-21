from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import logging

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {
            'random_forest': RandomForestClassifier(**config.get('rf_params', {})),
            'logistic_regression': LogisticRegression(**config.get('lr_params', {}))
        }
    
    def train(self, X_train, y_train, model_name='random_forest'):
        """Train the model"""
        self.logger.info(f"Training {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.logger.info("Training complete")
        return model
    
    def save_model(self, model, path):
        """Save trained model"""
        joblib.dump(model, path)
        self.logger.info(f"Model saved to {path}")
