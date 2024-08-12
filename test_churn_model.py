import unittest
import pandas as pd
from main import load_data, check_missing_values, prepare_data, train_model, evaluate_model

class TestChurnPredictionModel(unittest.TestCase):

    def setUp(self):
        # Setup the initial dataframe or load a sample dataset
        self.df_raw = pd.read_csv('split_part1.csv') # or create a sample dataframe

    def test_load_data(self):
        df = load_data('split_part1.csv')
        self.assertIsInstance(df, pd.DataFrame)
    
    def test_check_missing_values(self):
        columns_to_check = ['credit_score', 'age', 'credit_card', 'estimated_salary', 'churn']
        missing_values = check_missing_values(self.df_raw, columns_to_check)
        self.assertEqual(missing_values, {col: False for col in columns_to_check})
    
    def test_prepare_data(self):
        features_col = ['credit_score', 'estimated_salary']
        classification_col = ['churn']
        X, y = prepare_data(self.df_raw, features_col, classification_col)
        self.assertEqual(X.shape[1], len(features_col))
        self.assertEqual(y.shape[1], len(classification_col))
    
    def test_train_model(self):
        features_col = ['credit_score', 'estimated_salary']
        classification_col = ['churn']
        X, y = prepare_data(self.df_raw, features_col, classification_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model = train_model(X_train, y_train)
        self.assertIsNotNone(model)
    
    def test_evaluate_model(self):
        features_col = ['credit_score', 'estimated_salary']
        classification_col = ['churn']
        X, y = prepare_data(self.df_raw, features_col, classification_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model = train_model(X_train, y_train)
        accuracy, conf_matrix, class_report = evaluate_model(model, X_test, y_test)
        self.assertGreaterEqual(accuracy, 0)

if __name__ == '__main__':
    unittest.main()
