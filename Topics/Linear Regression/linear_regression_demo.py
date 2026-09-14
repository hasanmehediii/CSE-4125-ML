import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("student_performance.csv")
print("First 5 rows of the dataset:")
print(df.head())

X = df[["Study_Hours"]]   # input feature(s)
y = df["Marks"]            # target we want to predict
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

model = LinearRegression()
model.fit(X_train, y_train)   

b0 = model.intercept_          # this matches b0 in y_hat = b0 + b1*x
b1 = model.coef_[0]            # this matches b1 (the slope)

print("\n--- Learned Model Parameters ---")
print(f"Intercept (b0): {b0:.3f}")
print(f"Slope     (b1): {b1:.3f}")
print(f"Learned equation: Marks = {b0:.2f} + {b1:.2f} * Study_Hours")

y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    "Study_Hours": X_test["Study_Hours"].values,
    "Actual_Marks": y_test.values,
    "Predicted_Marks": y_pred.round(2)
})
print("\n--- Actual vs Predicted (Test Set) ---")
print(comparison)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"R-squared (R^2):          {r2:.3f}")

plt.figure(figsize=(8, 6))

plt.scatter(X, y, color="steelblue", label="Actual data points", alpha=0.7)

X_sorted = X.sort_values("Study_Hours")
y_line = model.predict(X_sorted)
plt.plot(X_sorted, y_line, color="red", linewidth=2, label="Best fit line")

plt.title("Linear Regression: Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("linear_regression_plot.png", dpi=150)
print("\nPlot saved as 'linear_regression_plot.png'")

new_study_hours = pd.DataFrame({"Study_Hours": [6.5]})
predicted_marks = model.predict(new_study_hours)[0]
print(f"\nIf a student studies 6.5 hours, predicted marks = {predicted_marks:.2f}")
