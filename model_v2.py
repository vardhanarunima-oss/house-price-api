import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# 1. Load data
raw = fetch_california_housing(as_frame=True)
df = raw.frame

# 2. Split features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Random Forest (more powerful than Linear Regression)
model = RandomForestRegressor(n_estimators=100, random_state=42)
# n_estimators=100 means it builds 100 decision trees and averages them
model.fit(X_train, y_train)
print("Model trained!")

# 5. Test it
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# 6. Save it (overwrites old model)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Better model saved!")

import matplotlib.pyplot as plt
import seaborn as sns

# Which features matter most to the model?
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(8,5))
sns.barplot(data=importance, x='importance', y='feature', color='steelblue')
plt.title("What affects house prices the most?")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Chart saved as feature_importance.png")