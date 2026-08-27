import pandas as pd

# Load our selected dataset
file_path = "Data/synthetic_fraud_dataset1.csv"

df = pd.read_csv(file_path)

print("=" * 80)
print("FINANCIAL FRAUD DATASET")
print("=" * 80)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFraud Distribution:")
print(df["Fraud_Label"].value_counts())

print("\nFraud Percentage:")
print(df["Fraud_Label"].value_counts(normalize=True) * 100)

print("\nTransaction Type:")
print(df["Transaction_Type"].value_counts())

print("\nDevice Type:")
print(df["Device_Type"].value_counts())

print("\nMerchant Category:")
print(df["Merchant_Category"].value_counts())

print("\nLocation:")
print(df["Location"].value_counts())

print("\nCard Type:")
print(df["Card_Type"].value_counts())

print("\nPrevious Fraudulent Activity:")
print(df["Previous_Fraudulent_Activity"].value_counts())

print("\nBasic Statistics:")
print(df.describe())