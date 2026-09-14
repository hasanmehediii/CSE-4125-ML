# Logistic Regression — সহজ বাংলা নোট

## 1. Logistic Regression কী?

**Logistic Regression** নামের মধ্যে Regression থাকলেও এটি মূলত একটি **Classification Algorithm**।

বিশেষ করে Binary Classification-এর জন্য ব্যবহৃত হয়।

**উদাহরণ:**

$$
\text{Email} \rightarrow \text{Spam / Not Spam}
$$

$$
\text{Student} \rightarrow \text{Pass / Fail}
$$

$$
\text{Patient} \rightarrow \text{Disease / No Disease}
$$

---

## 2. Linear Regression Classification-এর জন্য যথেষ্ট নয় কেন?

Linear Regression-এর output হতে পারে:

$$
-\infty < \hat{y} < \infty
$$

কিন্তু probability-এর range:

$$
0 \le P \le 1
$$

তাই classification-এর জন্য output-কে 0 এবং 1-এর মধ্যে আনতে Sigmoid Function ব্যবহার করা হয়।

---

## 3. Sigmoid Function

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

যেখানে:

$$
z = b_0 + b_1 x_1 + b_2 x_2 + \cdots
$$

Sigmoid output:

$$
0 < \sigma(z) < 1
$$

তাই এটি probability হিসেবে ব্যবহার করা যায়।

---

## 4. Sigmoid Example

যদি $z = 0$, তাহলে:

$$
\sigma(0) = \frac{1}{1+1} = 0.5
$$

অর্থাৎ 50% probability।

> **লক্ষ্য করো:** $z$ যত বড় positive হয়, $\sigma(z)$ তত 1-এর কাছাকাছি যায়; $z$ যত বড় negative হয়, $\sigma(z)$ তত 0-এর কাছাকাছি যায়। এই কারণেই sigmoid curve-টা S-আকৃতির (S-shaped)।

---

## 5. Prediction

ধরা যাক:

$$
P(\text{Pass}) = 0.82
$$

Threshold: $0.5$

কারণ $0.82 > 0.5$, তাই Prediction:

$$
\boxed{\text{Pass}}
$$

---

## 6. Decision Threshold

Default threshold সাধারণত $0.5$।

Rule:

$$
P(y=1 \mid x) \geq 0.5 \Rightarrow \text{Class 1}
$$

$$
P(y=1 \mid x) < 0.5 \Rightarrow \text{Class 0}
$$

Threshold সবসময় 0.5 হওয়া বাধ্যতামূলক নয়।

**বাস্তব উদাহরণ:**

Disease detection-এ false negative কমানোর জন্য threshold 0.3 করা যেতে পারে — কারণ রোগ miss করার cost, healthy মানুষকে ভুলভাবে flag করার cost থেকে অনেক বেশি গুরুত্বপূর্ণ।

---

## 7. Probability Connection

Logistic Regression মূলত estimate করে:

$$
P(Y=1 \mid X)
$$

অর্থাৎ features $X$ দেওয়া থাকলে class 1 হওয়ার probability।

**উদাহরণ:**

$$
P(\text{Pass} \mid \text{Study Hours, Attendance}) = 0.85
$$

অর্থাৎ Pass হওয়ার probability 85%।

---

## 8. Odds

Probability $p$ হলে:

$$
\text{Odds} = \frac{p}{1-p}
$$

**উদাহরণ:**

যদি $p = 0.8$, তাহলে:

$$
\text{Odds} = \frac{0.8}{0.2} = 4
$$

অর্থাৎ odds = 4:1।

---

## 9. Log-Odds বা Logit

Logistic Regression log-odds-কে linear ভাবে model করে:

$$
\log\left(\frac{p}{1-p}\right) = b_0 + b_1 x_1 + \cdots + b_n x_n
$$

এটিই **Logit**।

> এই কারণেই নামের মধ্যে "Regression" আছে — এটি আসলে log-odds-এর উপর একটি linear regression, যার output-কে পরে sigmoid দিয়ে probability-তে রূপান্তর করা হয়।

---

## 10. Binary Target

Binary Classification-এ:

$$
Y \in \{0, 1\}
$$

যেমন:

- Pass = 1, Fail = 0
- Spam = 1, Not Spam = 0

---

## 11. Bernoulli Connection

Binary target-কে Bernoulli Random Variable হিসেবে ধরা যায়।

যদি $P(Y=1) = p$, তাহলে:

$$
P(Y=0) = 1 - p
$$

Logistic Regression মূলত $p$ estimate করে।

---

## 12. Log Loss

Common loss function:

$$
L = -\left[ y \log(p) + (1-y)\log(1-p) \right]
$$

এখানে:

- $y$ = actual class
- $p$ = predicted probability

Correct class-এর probability বেশি হলে loss কম হয়।

---

## 13. Maximum Likelihood Idea

Training-এর সময় আমরা এমন parameters খুঁজি যাতে observed training labels পাওয়ার probability বেশি হয়।

এটি **Maximum Likelihood Estimation (MLE)**-এর basic idea। Log Loss আসলে negative log-likelihood, তাই Log Loss minimize করা মানে likelihood maximize করা।

---

## 14. Linear vs Logistic Regression

| বৈশিষ্ট্য               | Linear Regression       | Logistic Regression            |
| ------------------------ | ------------------------ | -------------------------------- |
| Task type                 | Regression                | Classification                     |
| কী predict করে              | Number predict করে         | Probability/Class predict করে       |
| Curve shape                | Straight line               | Sigmoid curve                        |
| Output range                | Any real number             | 0–1 probability                       |
| Common loss                 | MSE                          | Log Loss                              |
| উদাহরণ                       | House price                  | Pass/Fail                              |

---

## 15. বাস্তব উদাহরণ

Features:

- Study Hours
- Attendance

Target: Pass = 1, Fail = 0

Model predict করল:

$$
P(\text{Pass}) = 0.73
$$

Threshold: $0.5$, এবং $0.73 > 0.5$, তাহলে Prediction:

$$
\boxed{\text{Pass}}
$$

---

## 16. Confusion Matrix ও Evaluation Metrics

শুধু Accuracy দিয়ে সবসময় classification model বোঝা যায় না, বিশেষ করে data imbalanced হলে। তাই Confusion Matrix থেকে আসা কিছু metric ব্যবহার করা হয়:

|                     | Predicted Positive  | Predicted Negative  |
| ------------------- | -------------------- | -------------------- |
| **Actual Positive**  | True Positive (TP)    | False Negative (FN)   |
| **Actual Negative**  | False Positive (FP)   | True Negative (TN)    |

গুরুত্বপূর্ণ metrics:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Intuition:** Disease detection-এর মতো ক্ষেত্রে Recall বেশি গুরুত্বপূর্ণ (রোগী miss করা যাবে না), আর Spam detection-এ Precision বেশি গুরুত্বপূর্ণ (ভালো email ভুলে spam বানানো যাবে না)।

---

## 17. কোথায় ব্যবহার হয়?

- Spam detection
- Disease prediction
- Loan approval
- Fraud detection
- Customer churn
- Pass/Fail prediction

---

## 18. Short Exam Definition

**Logistic Regression is a supervised classification algorithm that estimates the probability of a class using the sigmoid function.**

---

## 19. Quick Revision

$$
\boxed{\text{Input} \rightarrow \text{Probability} \rightarrow \text{Class}}
$$

Important:

$$
\boxed{\sigma(z) = \frac{1}{1+e^{-z}}}
$$

$$
\boxed{P(Y=1 \mid X)}
$$

$$
\boxed{\text{Odds} = \frac{p}{1-p}}
$$

**Final Example:**

যদি model বলে কোনো email-এর Spam probability 0.91 এবং threshold 0.5, তাহলে email-টি Spam হিসেবে classify হবে।