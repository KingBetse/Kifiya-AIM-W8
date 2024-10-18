import unittest
import pandas as pd

class TestOutlierRemoval(unittest.TestCase):
    
    def setUp(self):
        # Sample data with outliers
        self.fraud_data = pd.DataFrame({
            'purchase_value': [100, 200, 300, 400, 10000],  # 10000 is an outlier
            'transaction_count': [1, 2, 3, 4, 100]            # 100 is an outlier
        })

    def test_outlier_removal(self):
        # Calculate Q1 and Q3 for purchase_value
        Q1_purchase = self.fraud_data['purchase_value'].quantile(0.25)
        Q3_purchase = self.fraud_data['purchase_value'].quantile(0.75)
        IQR_purchase = Q3_purchase - Q1_purchase

        # Define bounds for purchase_value outliers
        lower_bound_purchase = Q1_purchase - 1.5 * IQR_purchase
        upper_bound_purchase = Q3_purchase + 1.5 * IQR_purchase

        # Calculate Q1 and Q3 for transaction_count
        Q1_transaction = self.fraud_data['transaction_count'].quantile(0.25)
        Q3_transaction = self.fraud_data['transaction_count'].quantile(0.75)
        IQR_transaction = Q3_transaction - Q1_transaction

        # Define bounds for transaction_count outliers
        lower_bound_transaction = Q1_transaction - 1.5 * IQR_transaction
        upper_bound_transaction = Q3_transaction + 1.5 * IQR_transaction

        # Debugging: Print bounds
        print(f"Bounds for purchase_value: {lower_bound_purchase}, {upper_bound_purchase}")
        print(f"Bounds for transaction_count: {lower_bound_transaction}, {upper_bound_transaction}")

        # Remove outliers
        fraud_data_cleaned = self.fraud_data[
            (self.fraud_data['purchase_value'] >= lower_bound_purchase) & 
            (self.fraud_data['purchase_value'] <= upper_bound_purchase) &
            (self.fraud_data['transaction_count'] >= lower_bound_transaction) & 
            (self.fraud_data['transaction_count'] <= upper_bound_transaction)
        ]

        # Debugging: Print the cleaned DataFrame
        print(f"Cleaned DataFrame:\n{fraud_data_cleaned}")

        # Assertions to verify that outliers have been removed
        self.assertNotIn(10000, fraud_data_cleaned['purchase_value'].values)
        self.assertNotIn(100, fraud_data_cleaned['transaction_count'].values)

if __name__ == '__main__':
    unittest.main()