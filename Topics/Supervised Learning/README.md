# Supervised Learning — সহজ বাংলা নোট

## 1. Supervised Learning কী?

**Supervised Learning** হলো Machine Learning-এর এমন একটি পদ্ধতি যেখানে আমরা কম্পিউটারকে কিছু **উদাহরণসহ সঠিক উত্তর** দেখাই।

অর্থাৎ, Training Data-এর প্রতিটি input-এর সাথে একটি correct output বা **label** থাকে।

\[
\text{Input} \rightarrow \text{Correct Output}
\]

মডেল এই input-output সম্পর্ক শিখে পরে নতুন input-এর জন্য output অনুমান করে।

### বাস্তব জীবনের উদাহরণ

ধরো, তোমার কাছে 1000টি email আছে। প্রতিটি email-এর পাশে লেখা আছে:

- Spam
- Not Spam

এখন এই data দিয়ে model-কে train করানো হলো। পরে নতুন email এলে model বলবে সেটি Spam নাকি Not Spam।

---

## 2. Supervised Learning-এ Data কেমন হয়?

Supervised Learning-এর data সাধারণত দুইটি অংশে থাকে:

- **Feature (X):** Input information
- **Label / Target (y):** Correct answer

ধরা যাক, আমরা একটি বাসার দাম predict করতে চাই।

| Size (sq ft) | Rooms | Location Score | Price |
|---:|---:|---:|---:|
| 900 | 2 | 6 | 35 lakh |
| 1200 | 3 | 8 | 50 lakh |
| 1600 | 4 | 9 | 70 lakh |

এখানে:

- `Size`, `Rooms`, `Location Score` = Features
- `Price` = Label / Target

Model শিখবে:

\[
X \rightarrow y
\]

---

## 3. Training কী?

Training হলো model-কে data দেখিয়ে input এবং output-এর মধ্যকার relationship শেখানোর process।

Model শুরুতে random prediction করতে পারে। এরপর সে নিজের prediction এবং correct answer-এর difference দেখে নিজেকে improve করে।

### উদাহরণ

সঠিক বাসার দাম:

\[
50 \text{ lakh}
\]

Model predict করল:

\[
42 \text{ lakh}
\]

Error:

\[
50 - 42 = 8 \text{ lakh}
\]

Training-এর সময় model চেষ্টা করবে এই error কমাতে।

---

## 4. Supervised Learning-এর প্রধান দুই ধরন

Supervised Learning মূলত দুই ধরনের:

1. **Classification**
2. **Regression**

---

# 5. Classification

Classification-এ output একটি **category বা class** হয়।

### উদাহরণ

Email:

\[
\text{Email} \rightarrow \text{Spam / Not Spam}
\]

Medical test:

\[
\text{Patient Data} \rightarrow \text{Disease / No Disease}
\]

Image recognition:

\[
\text{Image} \rightarrow \text{Cat / Dog}
\]

### Binary Classification

যখন দুইটি class থাকে।

উদাহরণ:

- Yes / No
- Pass / Fail
- Fraud / Not Fraud

### Multi-class Classification

যখন দুইটির বেশি class থাকে।

উদাহরণ:

\[
\text{Image} \rightarrow \text{Cat / Dog / Bird / Horse}
\]

---

# 6. Regression

Regression-এ output সাধারণত একটি **continuous numerical value** হয়।

### উদাহরণ

বাসার দাম:

\[
\text{House Information} \rightarrow 52.5 \text{ lakh}
\]

Temperature prediction:

\[
\text{Weather Data} \rightarrow 31.7^\circ C
\]

Salary prediction:

\[
\text{Experience, Skills} \rightarrow 45000
\]

Classification category predict করে, কিন্তু Regression number predict করে।

---

# 7. Classification vs Regression

| Classification | Regression |
|---|---|
| Category predict করে | Number predict করে |
| Spam / Not Spam | House Price |
| Pass / Fail | Temperature |
| Cat / Dog | Salary |

সহজ নিয়ম:

> Output যদি category হয় → Classification  
> Output যদি numerical value হয় → Regression

---

# 8. Training Set এবং Test Set

একই data দিয়ে model train এবং evaluate করলে model-এর আসল performance বোঝা যায় না।

তাই data সাধারণত ভাগ করা হয়।

উদাহরণ:

\[
80\% \text{ Training Data}
\]

\[
20\% \text{ Test Data}
\]

### Training Set

Model এই data দেখে শেখে।

### Test Set

Model আগে দেখেনি এমন data দিয়ে পরীক্ষা করা হয়।

### বাস্তব উদাহরণ

তুমি যদি পরীক্ষার আগে শুধু একই 20টি প্রশ্ন মুখস্থ করো এবং পরীক্ষায়ও সেই প্রশ্ন আসে, তাহলে তোমার আসল understanding বোঝা যাবে না।

Test data হলো নতুন প্রশ্নের মতো।

---

# 9. Model কী?

Model হলো একটি mathematical system যা data থেকে pattern শেখে।

একটি খুব সহজ model হতে পারে:

\[
y = wx + b
\]

এখানে:

- \(x\) = input
- \(y\) = predicted output
- \(w\) = weight
- \(b\) = bias

### উদাহরণ

ধরা যাক:

\[
y = 5x + 10
\]

যদি:

\[
x = 4
\]

তাহলে:

\[
y = 5(4)+10 = 30
\]

Machine Learning-এ training-এর কাজ হলো ভালো \(w\) এবং \(b\) খুঁজে বের করা।

---

# 10. Prediction কী?

Training শেষ হওয়ার পর model নতুন input-এর জন্য output estimate করে।

এটিকে বলা হয় **Prediction** বা **Inference**।

### উদাহরণ

Training data:

\[
\text{Study Hours} \rightarrow \text{Exam Marks}
\]

Model শিখল যে বেশি সময় পড়লে সাধারণত marks বাড়ে।

নতুন student:

\[
6 \text{ hours/day}
\]

Model predict করতে পারে:

\[
82 \text{ marks}
\]

---

# 11. Error / Loss Function

Model-এর prediction কতটা ভুল হয়েছে তা measure করার জন্য **Loss Function** ব্যবহার করা হয়।

ধরা যাক:

Actual value:

\[
y = 100
\]

Prediction:

\[
\hat{y} = 90
\]

Error:

\[
100 - 90 = 10
\]

Regression-এ বহুল ব্যবহৃত একটি loss হলো Mean Squared Error (MSE):

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y_i})^2
\]

লক্ষ্য হলো loss যত সম্ভব কমানো।

---

# 12. Overfitting

যখন model training data খুব ভালোভাবে মনে রাখে কিন্তু নতুন data-তে খারাপ perform করে, তাকে **Overfitting** বলে।

### বাস্তব উদাহরণ

একজন student শুধু guidebook-এর answer মুখস্থ করল।

Guidebook-এর question এলে সে খুব ভালো করে, কিন্তু নতুন question এলে পারে না।

এটাই Overfitting-এর মতো।

---

# 13. Underfitting

যখন model training data-ই ঠিকমতো শিখতে পারে না, তাকে **Underfitting** বলে।

### উদাহরণ

একজন student exam-এর আগে chapter শুধু একবার চোখ বুলিয়েছে। সে পুরনো question এবং নতুন question—দুটিতেই খারাপ করবে।

---

# 14. Generalization

একটি ভালো model শুধু training data নয়, নতুন unseen data-তেও ভালো perform করে।

একে বলা হয় **Generalization**।

Machine Learning-এর আসল লক্ষ্য সাধারণত training data মুখস্থ করা নয়; বরং general pattern শেখা।

---

# 15. Common Supervised Learning Algorithms

## Linear Regression

Numerical value predict করার জন্য ব্যবহৃত হয়।

উদাহরণ:

\[
\text{Study Hours} \rightarrow \text{Marks}
\]

## Logistic Regression

নাম Regression হলেও এটি সাধারণত Classification-এর জন্য ব্যবহৃত হয়।

উদাহরণ:

\[
\text{Patient Data} \rightarrow \text{Disease / No Disease}
\]

## Decision Tree

একটি decision tree প্রশ্ন করতে করতে decision নেয়।

উদাহরণ:

- Income বেশি?
- Previous loan আছে?
- Payment history ভালো?

এরপর:

\[
\text{Loan Approved / Rejected}
\]

## K-Nearest Neighbors (KNN)

নতুন data point-এর কাছাকাছি থাকা training examples দেখে class নির্ধারণ করে।

উদাহরণ: একটি নতুন ফলের weight এবং color যদি বেশিরভাগ apple-এর মতো হয়, তাহলে model সেটিকে apple বলতে পারে।

## Support Vector Machine (SVM)

Different classes-এর মধ্যে সবচেয়ে ভালো separating boundary খুঁজে বের করার চেষ্টা করে।

উদাহরণ: Red এবং Blue data points আলাদা করার জন্য একটি ভালো line বা plane খুঁজে বের করা।

## Neural Network

অনেক layer এবং neuron ব্যবহার করে complex pattern শিখতে পারে।

উদাহরণ:

\[
\text{Image} \rightarrow \text{Cat / Dog}
\]

---

# 16. Supervised Learning Workflow

সাধারণ workflow:

\[
\text{Collect Data}
\rightarrow
\text{Prepare Data}
\rightarrow
\text{Split Data}
\rightarrow
\text{Train Model}
\rightarrow
\text{Evaluate}
\rightarrow
\text{Predict}
\]

### উদাহরণ

ধরা যাক তুমি student result predict করবে।

1. Student data collect করলে
2. Missing data পরিষ্কার করলে
3. Training এবং Test set বানালে
4. Model train করলে
5. Test data দিয়ে performance check করলে
6. নতুন student-এর result predict করলে

---

# 17. Evaluation Metrics

## Classification-এর জন্য

### Accuracy

\[
Accuracy =
\frac{\text{Correct Predictions}}
{\text{Total Predictions}}
\]

উদাহরণ:

100টির মধ্যে 90টি prediction সঠিক হলে:

\[
Accuracy = 90\%
\]

### Precision

Model positive বলেছে—এর মধ্যে আসলে কতগুলো positive ছিল।

### Recall

আসল positive example-এর মধ্যে model কতগুলো ধরতে পেরেছে।

### F1 Score

Precision এবং Recall-এর একটি balanced measurement।

---

## Regression-এর জন্য

### Mean Absolute Error (MAE)

\[
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y_i}|
\]

### Mean Squared Error (MSE)

\[
MSE =
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y_i})^2
\]

---

# 18. একটি Complete Real-Life Example

ধরা যাক তুমি predict করতে চাও একজন student পরীক্ষায় pass করবে কি না।

Features:

- Study hours
- Attendance
- Previous marks

Target:

- Pass
- Fail

Data:

| Study Hours | Attendance | Previous Marks | Result |
|---:|---:|---:|---|
| 2 | 60% | 40 | Fail |
| 5 | 80% | 65 | Pass |
| 7 | 90% | 75 | Pass |
| 1 | 50% | 30 | Fail |

এটি একটি **Classification Problem**, কারণ output হলো:

\[
\text{Pass / Fail}
\]

নতুন student-এর data:

- Study = 6 hours
- Attendance = 85%
- Previous Marks = 70

Model হয়তো predict করবে:

\[
\boxed{\text{Pass}}
\]

---

# 19. Supervised Learning-এর সীমাবদ্ধতা

Supervised Learning-এর সবচেয়ে বড় সমস্যা হলো **labeled data প্রয়োজন**।

অনেক ক্ষেত্রে label তৈরি করতে মানুষের প্রয়োজন হয়।

উদাহরণ:

10 lakh medical image থাকলে প্রতিটি image-এ disease আছে কি না একজন doctor দিয়ে label করানো খুব expensive হতে পারে।

---

# 20. খুব সংক্ষেপে মনে রাখবে

Supervised Learning:

\[
\boxed{\text{Input + Correct Answer} \rightarrow \text{Learning}}
\]

দুই প্রধান ধরন:

\[
\boxed{\text{Classification + Regression}}
\]

Classification:

\[
\boxed{\text{Category Prediction}}
\]

Regression:

\[
\boxed{\text{Numerical Prediction}}
\]

---

# Probability Theory — Machine Learning-এর জন্য শুরু

Supervised Learning বুঝতে গেলে Probability খুব গুরুত্বপূর্ণ, বিশেষ করে Classification, Bayesian models, uncertainty এবং prediction confidence বোঝার জন্য।

---

# 21. Probability কী?

Probability হলো কোনো ঘটনা ঘটার সম্ভাবনা।

\[
0 \leq P(A) \leq 1
\]

যেখানে:

- \(P(A)=0\) → ঘটনা ঘটবে না
- \(P(A)=1\) → ঘটনা নিশ্চিতভাবে ঘটবে
- \(P(A)=0.5\) → 50% chance

### উদাহরণ

একটি fair coin toss করলে:

\[
P(\text{Head}) = \frac{1}{2}
\]

\[
P(\text{Tail}) = \frac{1}{2}
\]

---

# 22. Sample Space

সব possible outcome-এর set-কে **Sample Space** বলে।

একটি dice roll করলে:

\[
S = \{1,2,3,4,5,6\}
\]

### উদাহরণ

Even number পাওয়ার event:

\[
A = \{2,4,6\}
\]

তাই:

\[
P(A)=\frac{3}{6}=\frac{1}{2}
\]

---

# 23. Event

Sample space-এর কোনো subset-কে Event বলে।

Dice example:

\[
S = \{1,2,3,4,5,6\}
\]

Even number:

\[
A = \{2,4,6\}
\]

এখানে \(A\) একটি event।

---

# 24. Conditional Probability

একটি ঘটনা ইতোমধ্যে ঘটেছে জানা থাকলে অন্য একটি ঘটনার probability-কে Conditional Probability বলে।

\[
P(A|B)
=
\frac{P(A\cap B)}{P(B)}
\]

### বাস্তব উদাহরণ

ধরা যাক:

- 100 জন student
- 60 জন programming জানে
- Programming জানা 60 জনের মধ্যে 45 জন AI course-এ ভালো করেছে

তাহলে:

\[
P(\text{Good in AI}|\text{Knows Programming})
=
\frac{45}{60}
=
0.75
\]

অর্থাৎ 75%।

---

# 25. Independent Events

যদি একটি event ঘটলে অন্যটির probability পরিবর্তন না হয়, তাহলে event দুইটি independent।

### উদাহরণ

দুইবার coin toss করা।

প্রথমবার Head আসা দ্বিতীয় toss-এর result-কে affect করে না।

\[
P(A\cap B)=P(A)P(B)
\]

---

# 26. Bayes' Theorem

Machine Learning-এ Probability-এর সবচেয়ে গুরুত্বপূর্ণ formula-গুলোর একটি হলো Bayes' Theorem।

\[
P(A|B)
=
\frac{P(B|A)P(A)}
{P(B)}
\]

এটি নতুন evidence পাওয়ার পর probability update করতে সাহায্য করে।

### বাস্তব উদাহরণ

ধরো কোনো disease rare, কিন্তু test positive এসেছে।

শুধু test positive হলেই যে disease নিশ্চিত—এটা বলা যায় না।

Bayes' Theorem দিয়ে আমরা হিসাব করতে পারি:

\[
P(\text{Disease}|\text{Positive Test})
\]

এই idea Machine Learning-এর Bayesian classification-এ খুব গুরুত্বপূর্ণ।

---

# 27. Random Variable

Random experiment-এর numerical outcome-কে Random Variable বলে।

### উদাহরণ

একটি dice roll করা হলো।

\[
X = \text{Dice-এর result}
\]

তাহলে:

\[
X \in \{1,2,3,4,5,6\}
\]

---

# 28. Expected Value

Expected Value হলো একটি random variable-এর long-term average value।

\[
E[X] = \sum_x xP(X=x)
\]

### Dice Example

Fair dice-এর expected value:

\[
E[X]
=
\frac{1+2+3+4+5+6}{6}
=
3.5
\]

এর মানে এই না যে dice-এ 3.5 উঠবে। অনেকবার roll করলে average প্রায় 3.5-এর দিকে যাবে।

---

# 29. Variance

Variance বলে data average-এর চারপাশে কতটা spread হয়েছে।

\[
Var(X)=E[(X-\mu)^2]
\]

যেখানে:

\[
\mu=E[X]
\]

Variance বেশি → data বেশি spread out।

Variance কম → data mean-এর কাছাকাছি।

---

# 30. ML-এ Probability কেন দরকার?

Probability ব্যবহার হয়:

- Classification confidence-এ
- Naive Bayes algorithm-এ
- Bayesian Network-এ
- Uncertainty measure করতে
- Data distribution বুঝতে
- Logistic Regression-এর output বুঝতে
- Probabilistic reasoning-এ

### উদাহরণ

একটি classifier বলতে পারে:

\[
P(\text{Cat})=0.90
\]

\[
P(\text{Dog})=0.10
\]

তাহলে model Cat predict করবে।

---

# Final Revision Map

এই sequence-এ পড়লে সহজ হবে:

\[
\boxed{
\text{Supervised Learning}
\rightarrow
\text{Classification}
\rightarrow
\text{Regression}
\rightarrow
\text{Training/Test}
\rightarrow
\text{Loss}
\rightarrow
\text{Overfitting}
\rightarrow
\text{Metrics}
\rightarrow
\text{Probability}
\rightarrow
\text{Conditional Probability}
\rightarrow
\text{Bayes' Theorem}
}
\]

পরবর্তীতে Probability থেকে এগুলো পড়া ভালো:

1. Probability Distribution
2. Bernoulli Distribution
3. Binomial Distribution
4. Normal Distribution
5. Joint Probability
6. Marginal Probability
7. Conditional Independence
8. Bayes' Rule
9. Bayesian Network
