#importing libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

#loading data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Separate features and target
X = train_df.drop(columns=['CustomerID', 'Churn'])
y = train_df['Churn']

# Identifying categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Applying one-hot encoding to both train and test datasets
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(test_df.drop(columns=['CustomerID']), columns=categorical_cols, drop_first=True)

# Aligning the train and test datasets to ensure they have the same columns
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# Spliting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Validating the model
val_preds = rf.predict_proba(X_val)[:, 1]
roc_score = roc_auc_score(y_val, val_preds)
print(f'Validation ROC AUC Score: {roc_score}')

# Making predictions on the test set
test_preds = rf.predict_proba(X_test)[:, 1]

# Prepareing submission
prediction_df = pd.DataFrame({'CustomerID': test_df['CustomerID'], 'predicted_probability': test_preds})

# Saving the predictions to a CSV file
prediction_df.to_csv("prediction_submission.csv", index=False)
