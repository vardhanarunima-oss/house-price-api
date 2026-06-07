import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# 1. Load data
raw = fetch_california_housing(as_frame=True)
df = raw.frame

# 2. Split into features (X) and target (y)
X = df.drop("MedHouseVal", axis=1)  # everything except price
y = df["MedHouseVal"]               # just the price

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 80% of data trains the model, 20% tests it

# 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained!")

# 5. Test it
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error: {mae:.2f} ($100,000s)")
print(f"R² Score: {r2:.2f}")
# R² of 1.0 = perfect, 0.6+ = decent for a first model

# 6. Save the model to a file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as model.pkl")