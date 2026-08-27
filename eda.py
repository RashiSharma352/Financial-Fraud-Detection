import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Data/synthetic_fraud_dataset1.csv")

# 1. Fraud vs Normal
df["Fraud_Label"].value_counts().plot(kind="bar")

plt.title("Fraud vs Normal Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.xticks([0, 1], ["Normal", "Fraud"], rotation=0)

plt.tight_layout()
plt.show()


# 2. Fraud by Transaction Type
pd.crosstab(
    df["Transaction_Type"],
    df["Fraud_Label"]
).plot(kind="bar")

plt.title("Fraud by Transaction Type")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

plt.tight_layout()
plt.show()


# 3. Fraud by Device
pd.crosstab(
    df["Device_Type"],
    df["Fraud_Label"]
).plot(kind="bar")

plt.title("Fraud by Device Type")
plt.xlabel("Device Type")
plt.ylabel("Number of Transactions")

plt.tight_layout()
plt.show()


# 4. Fraud by Merchant Category
pd.crosstab(
    df["Merchant_Category"],
    df["Fraud_Label"]
).plot(kind="bar")

plt.title("Fraud by Merchant Category")
plt.xlabel("Merchant Category")
plt.ylabel("Number of Transactions")

plt.tight_layout()
plt.show()


# 5. Fraud by Location
pd.crosstab(
    df["Location"],
    df["Fraud_Label"]
).plot(kind="bar")

plt.title("Fraud by Location")
plt.xlabel("Location")
plt.ylabel("Number of Transactions")

plt.tight_layout()
plt.show()