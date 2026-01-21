import pandas as pd

class FeatureEngineering:
    def create_features(self, df):
        """Create new features"""
        if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
            df['total_charges_estimate'] = df['tenure'] * df['MonthlyCharges']
        return df
    
    def select_features(self, X, y, method='correlation', threshold=0.1):
        """Feature selection"""
        from sklearn.feature_selection import SelectKBest, f_classif
        
        if method == 'kbest':
            selector = SelectKBest(f_classif, k=10)
            X_selected = selector.fit_transform(X, y)
            return X_selected, selector
        return X, None
