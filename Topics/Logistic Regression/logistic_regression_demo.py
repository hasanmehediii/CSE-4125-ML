import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# -------------------------------------------------------------
# STEP 1: Load the dataset
# -------------------------------------------------------------
df = pd.read_csv("student_performance.csv")
print("First 5 rows of the dataset:")
print(df.head())

# We use TWO features here (Multiple Logistic Regression):
#   Study_Hours and Attendance
# Target: Pass (1 = passed, 0 = failed)
X = df[["Study_Hours", "Attendance"]]
y = df["Pass"]

# -------------------------------------------------------------
# STEP 2: Split into training set and testing set
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# stratify=y keeps the Pass/Fail ratio similar in both train and test sets

print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# -------------------------------------------------------------
# STEP 3: Create and train the Logistic Regression model
# -------------------------------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------------------------------------
# STEP 4: Look at the learned parameters
# -------------------------------------------------------------
b0 = model.intercept_[0]
b1, b2 = model.coef_[0]

print("\n--- Learned Model Parameters ---")
print(f"Intercept (b0):            {b0:.4f}")
print(f"Coefficient for Study_Hours (b1): {b1:.4f}")
print(f"Coefficient for Attendance  (b2): {b2:.4f}")
print(f"Learned equation: z = {b0:.3f} + {b1:.3f}*Study_Hours + {b2:.3f}*Attendance")
print("Probability of Pass = sigmoid(z) = 1 / (1 + e^-z)")

# -------------------------------------------------------------
# STEP 5: Predict probabilities and classes on the test set
# -------------------------------------------------------------
# predict_proba gives probability for both classes: [P(Fail), P(Pass)]
# we only need the probability of class 1 (Pass), which is column index 1
y_proba = model.predict_proba(X_test)[:, 1]

# predict() applies the default 0.5 threshold automatically
y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    "Study_Hours": X_test["Study_Hours"].values,
    "Attendance": X_test["Attendance"].values,
    "Actual_Pass": y_test.values,
    "Predicted_Probability": y_proba.round(3),
    "Predicted_Class": y_pred
})
print("\n--- Actual vs Predicted (Test Set) ---")
print(comparison)

# -------------------------------------------------------------
# STEP 6: Evaluate the model
# -------------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

print("\nConfusion Matrix:")
print("                 Predicted Fail   Predicted Pass")
print(f"Actual Fail            {cm[0][0]:<10}      {cm[0][1]}")
print(f"Actual Pass            {cm[1][0]:<10}      {cm[1][1]}")

print("\nFull classification report:")
print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

# -------------------------------------------------------------
# STEP 7: Visualize the Sigmoid Curve (using Study_Hours only)
# -------------------------------------------------------------
# To draw a clean 2D sigmoid curve, we train a SECOND simple model
# using only Study_Hours, just for this visualization.
simple_model = LogisticRegression()
simple_model.fit(df[["Study_Hours"]], df["Pass"])

hours_range = pd.DataFrame({
    "Study_Hours": np.linspace(df["Study_Hours"].min(), df["Study_Hours"].max(), 300)
})
prob_range = simple_model.predict_proba(hours_range)[:, 1]

plt.figure(figsize=(8, 6))
plt.scatter(df["Study_Hours"], df["Pass"], color="steelblue", alpha=0.6, label="Actual data (0=Fail, 1=Pass)")
plt.plot(hours_range, prob_range, color="red", linewidth=2, label="Sigmoid curve (P(Pass))")
plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Threshold = 0.5")

plt.title("Logistic Regression: Study Hours vs Probability of Passing")
plt.xlabel("Study Hours")
plt.ylabel("Probability of Pass")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("logistic_regression_plot.png", dpi=150)
print("\nPlot saved as 'logistic_regression_plot.png'")

# -------------------------------------------------------------
# STEP 8: Predict Pass/Fail for a brand-new student
# -------------------------------------------------------------
new_student = pd.DataFrame({"Study_Hours": [5.0], "Attendance": [70.0]})[["Study_Hours", "Attendance"]]
new_proba = model.predict_proba(new_student)[0][1]
new_class = model.predict(new_student)[0]
result_text = "Pass" if new_class == 1 else "Fail"

print(f"\nNew student -> Study_Hours=5.0, Attendance=70.0")
print(f"Predicted probability of passing: {new_proba:.3f}")
print(f"Final prediction (threshold 0.5): {result_text}")
