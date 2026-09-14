# Linear Regression — সহজ বাংলা নোট

## 1. Linear Regression কী?

**Linear Regression** হলো একটি Supervised Learning algorithm যা সাধারণত continuous numerical value predict করতে ব্যবহার করা হয়।

Basic equation:

\[
\hat{y}=b_0+b_1x
\]

এখানে:

- \(x\) = input
- \(\hat{y}\) = predicted output
- \(b_0\) = intercept
- \(b_1\) = slope

---

## 2. বাস্তব জীবনের উদাহরণ

ধরা যাক:

| Study Hours | Marks |
|---:|---:|
| 1 | 40 |
| 2 | 50 |
| 3 | 60 |
| 4 | 70 |
| 5 | 80 |

একটি model হতে পারে:

\[
\hat{y}=30+10x
\]

যদি student 4 ঘণ্টা পড়ে:

\[
\hat{y}=30+10(4)=70
\]

Predicted marks = 70।

---

# 3. Slope

Equation:

\[
\hat{y}=b_0+b_1x
\]

এখানে \(b_1\) হলো slope।

Slope বলে input 1 unit বাড়লে output কত পরিবর্তিত হবে।

### উদাহরণ

\[
\hat{y}=30+10x
\]

এখানে slope:

\[
b_1=10
\]

অর্থাৎ study hour 1 বাড়লে predicted marks 10 বাড়ে।

---

# 4. Intercept

\(b_0\) হলো intercept।

যখন:

\[
x=0
\]

তখন predicted value:

\[
\hat{y}=b_0
\]

### উদাহরণ

\[
\hat{y}=30+10x
\]

Intercept = 30।

---

# 5. Best Fit Line

Linear Regression এমন একটি line খুঁজে যা data points-এর সাথে সবচেয়ে ভালো fit করে।

এটিকে বলা হয়:

**Best Fit Line** বা **Regression Line**

Goal:

\[
\boxed{\text{Overall Error Minimum করা}}
\]

---

# 6. Residual

Actual value এবং predicted value-এর difference হলো residual।

\[
e_i=y_i-\hat{y}_i
\]

### উদাহরণ

Actual:

\[
y=75
\]

Prediction:

\[
\hat{y}=70
\]

Residual:

\[
75-70=5
\]

---

# 7. Mean Squared Error

Common loss:

\[
MSE=
\frac{1}{n}
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

### উদাহরণ

Actual:

\[
[10,20]
\]

Predicted:

\[
[12,18]
\]

Squared errors:

\[
4,4
\]

তাই:

\[
MSE=\frac{4+4}{2}=4
\]

---

# 8. Least Squares Method

Linear Regression এমন \(b_0\) এবং \(b_1\) খোঁজে যাতে:

\[
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

minimum হয়।

এটাই Least Squares idea।

---

# 9. Simple Linear Regression

একটি feature থাকলে:

\[
\hat{y}=b_0+b_1x
\]

### উদাহরণ

\[
\text{Study Hours} \rightarrow \text{Marks}
\]

---

# 10. Multiple Linear Regression

একাধিক feature থাকলে:

\[
\hat{y}
=
b_0+b_1x_1+b_2x_2+\cdots+b_nx_n
\]

### উদাহরণ

House price:

\[
\text{Price}
=
b_0+b_1(\text{Size})+b_2(\text{Rooms})+b_3(\text{Location})
\]

---

# 11. Assumptions

Basic assumptions:

- Relationship approximately linear
- Errors independent
- Error variance approximately constant
- Features খুব বেশি redundant না হওয়া ভালো

---

# 12. কোথায় ব্যবহার হয়?

- House price prediction
- Salary prediction
- Sales forecasting
- Marks prediction
- Demand prediction
- Temperature estimation

---

# 13. Linear Regression vs Classification

Linear Regression ব্যবহার হয়:

\[
\boxed{\text{Number Predict করতে}}
\]

Classification ব্যবহার হয়:

\[
\boxed{\text{Category Predict করতে}}
\]

### উদাহরণ

House price = Linear Regression

Pass/Fail = Classification

---

# 14. Short Exam Definition

**Linear Regression is a supervised learning algorithm used to model a linear relationship between input variables and a continuous numerical output.**

---

# 15. Quick Revision

\[
\boxed{\hat{y}=b_0+b_1x}
\]

মনে রাখবে:

- \(b_0\) = intercept
- \(b_1\) = slope
- \(y-\hat{y}\) = residual
- MSE = error measurement
- Least Squares = best-fit parameter finding

### Final Example

Study hours থেকে marks predict করলে এবং output numerical হলে Linear Regression ব্যবহার করা যায়।
