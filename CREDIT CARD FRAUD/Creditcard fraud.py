# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Data preprocessing
X = data.drop('Class', axis=1)
y = data['Class']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building the isolation forest model
model = IsolationForest(contamination=0.01, random_state=42)  # Contamination is the expected proportion of outliers in the data
model.fit(X_train)

# Making predictions
y_pred = model.predict(X_test)

# Converting predictions to 0 (normal) and 1 (fraudulent)
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

# Evaluating the model
print(classification_report(y_test, y_pred))
