import pandas as pd
import numpy as np

# Generate sample customer churn data
np.random.seed(42)
n_samples = 1000

data = {
    'CustomerID': range(1, n_samples + 1),
    'Age': np.random.randint(18, 70, n_samples),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'tenure': np.random.randint(0, 72, n_samples),
    'MonthlyCharges': np.random.uniform(20, 120, n_samples),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
    'Churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
}

df = pd.DataFrame(data)
df.to_csv('data/raw/customer_churn.csv', index=False)
print("Sample data created successfully!")
print(f"Shape: {df.shape}")
print(df.head())
