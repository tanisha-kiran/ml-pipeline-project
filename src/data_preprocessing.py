import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

class DataPreprocessing:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def encode_categorical(self, df, categorical_cols, fit=True):
        """Encode categorical variables"""
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        return df
    
    def scale_features(self, X, fit=True):
        """Scale numerical features"""
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
from imblearn.over_sampling import SMOTE

def balance_classes(self, X, y):
    """Balance classes using SMOTE"""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled