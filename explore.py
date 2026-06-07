import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load the dataset (built-in, no download needed)
raw = fetch_california_housing(as_frame=True)
df = raw.frame

# Basic look
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nStats:")
print(df.describe())

# Plot house value distribution
plt.figure(figsize=(8,4))
sns.histplot(df['MedHouseVal'], bins=50, color='steelblue')
plt.title("Distribution of House Prices")
plt.xlabel("Median House Value ($100,000s)")
plt.savefig("price_distribution.png")
print("\nChart saved as price_distribution.png")