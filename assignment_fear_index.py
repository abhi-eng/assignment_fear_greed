import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Load data
df = pd.read_csv("fear_greed_index.csv")
df['date'] = pd.to_datetime(df['date'])

# Simulate trader performance based on sentiment
np.random.seed(42)
df['daily_return'] = np.random.normal(loc=0.001, scale=0.02, size=len(df)) + (df['value'] - 50) / 10000

# Create lagged sentiment value
df['value_lag1'] = df['value'].shift(1)

# Create classification target: profitable (1) or not (0)
df['profitable'] = (df['daily_return'] > 0).astype(int)

# Create additional features
df['sentiment_change'] = df['value'] - df['value_lag1']
df['rolling_avg'] = df['value'].rolling(window=3).mean().shift(1)

# Drop rows with missing values
df.dropna(subset=['value_lag1', 'sentiment_change', 'rolling_avg'], inplace=True)

# Features and target
X = df[['value_lag1', 'sentiment_change', 'rolling_avg']]
y = df['profitable']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

importances = clf.feature_importances_
feature_names = X.columns
model = ols("daily_return ~ value_lag1", data=df).fit()
print("\nOLS Regression Summary:")
print(model.summary())
