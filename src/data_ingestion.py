import pandas as pd
import logging
from pathlib import Path

class DataIngestion:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def load_data(self):
        """Load data from source"""
        try:
            df = pd.read_csv(self.config['data_path'])
            self.logger.info(f"Data loaded: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def split_data(self, df, test_size=0.2):
        """Split into train and test"""
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df, test_size=test_size, random_state=42)
        return train, test
