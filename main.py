import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
warnings.filterwarnings('ignore')

# df_raw= pd.read_csv('BankCustomerChurnPrediction.csv')
df_raw= pd.read_csv('split_part1.csv') # rows 5000
# print(df_raw.shape)

df_raw.head()
df = df_raw.copy()

# check data
columns_to_check = ['credit_score', 'age', 'credit_card', 'estimated_salary', 'churn']
missing_values = {col: df[col].isna().values.any() for col in columns_to_check}
# print(missing_values)

features_col = ['credit_score', 'estimated_salary']
classification_col = ['churn'] 

X = df[features_col]
y = df[classification_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred =logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
# print(f'Confusion Matrix:{conf_matrix}')
# print(f'Classification Report:{class_report}')

