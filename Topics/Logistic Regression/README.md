# Logistic Regression — সহজ বাংলা নোট

## 1. Logistic Regression কী?

**Logistic Regression** নামের মধ্যে Regression থাকলেও এটি মূলত একটি **Classification Algorithm**।

বিশেষ করে Binary Classification-এর জন্য ব্যবহৃত হয়।

### উদাহরণ

\[
\text{Email} \rightarrow \text{Spam / Not Spam}
\]

\[
\text{Student} \rightarrow \text{Pass / Fail}
\]

\[
\text{Patient} \rightarrow \text{Disease / No Disease}
\]

---

# 2. Linear Regression Classification-এর জন্য যথেষ্ট নয় কেন?

Linear Regression-এর output হতে পারে:

\[
-\infty < \hat{y} < \infty
\]

কিন্তু probability-এর range:

\[
0 \le P \le 1
\]

তাই classification-এর জন্য output-কে 0 এবং 1-এর মধ্যে আনতে Sigmoid Function ব্যবহার করা হয়।

---

# 3. Sigmoid Function

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

যেখানে:

\[
z=b_0+b_1x_1+b_2x_2+\cdots
\]

Sigmoid output:

\[
0<\sigma(z)<1
\]

তাই এটি probability হিসেবে ব্যবহার করা যায়।

---

# 4. Sigmoid Example

যদি:

\[
z=0
\]

তাহলে:

\[
\sigma(0)=\frac{1}{1+1}=0.5
\]

অর্থাৎ 50% probability।

---

# 5. Prediction

ধরা যাক:

\[
P(\text{Pass})=0.82
\]

Threshold:

\[
0.5
\]

কারণ:

\[
0.82>0.5
\]

Prediction:

\[
\boxed{\text{Pass}}
\]

---

# 6. Decision Threshold

Default threshold সাধারণত:

\[
0.5
\]

Rule:

\[
P(y=1|x)\geq0.5
\Rightarrow
\text{Class 1}
\]

\[
P(y=1|x)<0.5
\Rightarrow
\text{Class 0}
\]

Threshold সবসময় 0.5 হওয়া বাধ্যতামূলক নয়।

### বাস্তব উদাহরণ

Disease detection-এ false negative কমানোর জন্য threshold 0.3 করা যেতে পারে।

---

# 7. Probability Connection

Logistic Regression মূলত estimate করে:

\[
P(Y=1|X)
\]

অর্থাৎ features \(X\) দেওয়া থাকলে class 1 হওয়ার probability।

### উদাহরণ

\[
P(\text{Pass}|\text{Study Hours, Attendance})=0.85
\]

অর্থাৎ Pass হওয়ার probability 85%।

---

# 8. Odds

Probability \(p\) হলে:

\[
Odds=\frac{p}{1-p}
\]

### উদাহরণ

যদি:

\[
p=0.8
\]

তাহলে:

\[
Odds=\frac{0.8}{0.2}=4
\]

অর্থাৎ odds = 4:1।

---

# 9. Log-Odds বা Logit

Logistic Regression log-odds-কে linearভাবে model করে:

\[
\log
\left(
\frac{p}{1-p}
\right)
=
b_0+b_1x_1+\cdots+b_nx_n
\]

এটিই Logit।

---

# 10. Binary Target

Binary Classification-এ:

\[
Y\in\{0,1\}
\]

যেমন:

- Pass = 1
- Fail = 0

- Spam = 1
- Not Spam = 0

---

# 11. Bernoulli Connection

Binary target-কে Bernoulli Random Variable হিসেবে ধরা যায়।

যদি:

\[
P(Y=1)=p
\]

তাহলে:

\[
P(Y=0)=1-p
\]

Logistic Regression মূলত \(p\) estimate করে।

---

# 12. Log Loss

Common loss function:

\[
L=
-\left[
y\log(p)+(1-y)\log(1-p)
\right]
\]

এখানে:

- \(y\) = actual class
- \(p\) = predicted probability

Correct class-এর probability বেশি হলে loss কম হয়।

---

# 13. Maximum Likelihood Idea

Training-এর সময় আমরা এমন parameters খুঁজি যাতে observed training labels পাওয়ার probability বেশি হয়।

এটি Maximum Likelihood-এর basic idea।

---

# 14. Linear vs Logistic Regression

| Linear Regression | Logistic Regression |
|---|---|
| Regression | Classification |
| Number predict করে | Probability/Class predict করে |
| Straight line | Sigmoid curve |
| Output any real number | Output 0–1 probability |
| MSE common | Log Loss common |
| House price | Pass/Fail |

---

# 15. বাস্তব উদাহরণ

Features:

- Study Hours
- Attendance

Target:

- Pass = 1
- Fail = 0

Model predict করল:

\[
P(\text{Pass})=0.73
\]

Threshold:

\[
0.5
\]

তাহলে:

\[
0.73>0.5
\]

Prediction:

\[
\boxed{\text{Pass}}
\]

---

# 16. কোথায় ব্যবহার হয়?

- Spam detection
- Disease prediction
- Loan approval
- Fraud detection
- Customer churn
- Pass/Fail prediction

---

# 17. Short Exam Definition

**Logistic Regression is a supervised classification algorithm that estimates the probability of a class using the sigmoid function.**

---

# 18. Quick Revision

\[
\boxed{\text{Input} \rightarrow \text{Probability} \rightarrow \text{Class}}
\]

Important:

\[
\boxed{\sigma(z)=\frac{1}{1+e^{-z}}}
\]

\[
\boxed{P(Y=1|X)}
\]

\[
\boxed{\text{Odds}=\frac{p}{1-p}}
\]

### Final Example

যদি model বলে কোনো email-এর Spam probability 0.91 এবং threshold 0.5, তাহলে email-টি Spam হিসেবে classify হবে।
