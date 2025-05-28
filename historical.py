import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv("historical_data.csv")
df['date'] = pd.to_datetime(df['date'])

# Sort by date just in case
df = df.sort_values('date')

# Feature engineering
df['return'] = df['close'].pct_change()
df['next_day_return'] = df['return'].shift(-1)
df['target'] = (df['next_day_return'] > 0).astype(int)

# Technical indicators
df['ma_5'] = df['close'].rolling(window=5).mean()
df['ma_10'] = df['close'].rolling(window=10).mean()
df['volatility'] = df['return'].rolling(window=5).std()

# Drop NaN rows
df.dropna(inplace=True)

# Features and labels
X = df[['ma_5', 'ma_10', 'volatility']]
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions and evaluation
y_pred = clf.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


