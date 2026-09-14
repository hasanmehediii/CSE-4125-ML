# Unsupervised Learning — সহজ বাংলা নোট

## 1. Unsupervised Learning কী?

**Unsupervised Learning** হলো Machine Learning-এর এমন একটি পদ্ধতি যেখানে training data-তে কোনো **label** বা correct answer দেওয়া থাকে না।

অর্থাৎ:

\[
\text{Input Data} \rightarrow \text{Hidden Pattern খুঁজে বের করা}
\]

### বাস্তব জীবনের উদাহরণ

ধরো, একটি shopping mall-এর কাছে customer data আছে:

- Age
- Monthly Income
- Spending Score

কিন্তু কোনো customer-এর পাশে লেখা নেই সে কোন group-এর।

Model নিজে similarity দেখে group বানাতে পারে।

---

## 2. Supervised vs Unsupervised Learning

| Supervised Learning | Unsupervised Learning |
|---|---|
| Label থাকে | Label থাকে না |
| Correct answer জানা থাকে | Correct answer জানা থাকে না |
| Prediction করা হয় | Pattern খোঁজা হয় |
| Classification, Regression | Clustering, PCA |

মনে রাখবে:

\[
\boxed{\text{Unsupervised = No Label + Pattern Finding}}
\]

---

## 3. প্রধান কাজ

Unsupervised Learning-এর প্রধান কাজ:

1. Clustering
2. Dimensionality Reduction
3. Anomaly Detection
4. Association Rule Learning

---

# 4. Clustering

Similar data points-কে একই group-এ রাখাকে **Clustering** বলে।

### উদাহরণ

Student data:

- CGPA
- Attendance
- Assignment Score

Model student-দের group করতে পারে:

- High performer
- Average performer
- Weak performer

---

# 5. K-Means Clustering

K-Means একটি জনপ্রিয় clustering algorithm।

এখানে \(K\) = cluster-এর সংখ্যা।

যদি:

\[
K=3
\]

তাহলে model data-কে 3টি cluster-এ ভাগ করার চেষ্টা করবে।

### Steps

1. \(K\) টি centroid নেওয়া
2. প্রতিটি point-এর nearest centroid বের করা
3. Point-কে সেই cluster-এ assign করা
4. নতুন centroid calculate করা
5. Stable না হওয়া পর্যন্ত repeat করা

---

# 6. Centroid

Centroid হলো cluster-এর center।

যদি points হয়:

\[
(2,4), (4,6), (6,8)
\]

তাহলে:

\[
x_c=\frac{2+4+6}{3}=4
\]

\[
y_c=\frac{4+6+8}{3}=6
\]

Centroid:

\[
(4,6)
\]

---

# 7. Euclidean Distance

দুইটি point-এর distance:

\[
d=\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}
\]

### উদাহরণ

\[
A=(1,2), \quad B=(4,6)
\]

\[
d=\sqrt{(4-1)^2+(6-2)^2}
\]

\[
=\sqrt{9+16}=5
\]

---

# 8. Dimensionality Reduction

Feature সংখ্যা কমানোর process-কে Dimensionality Reduction বলে।

### উদাহরণ

একটি dataset-এ 100টি feature আছে।

PCA ব্যবহার করে হয়তো 10টি important feature রাখা যায়।

উপকারিতা:

- Training faster
- Storage কম
- Visualization সহজ
- Noise কমতে পারে

---

# 9. PCA

**Principal Component Analysis (PCA)** high-dimensional data-কে কম dimension-এ represent করে।

### উদাহরণ

100-dimensional student data-কে 2D graph-এ দেখাতে PCA ব্যবহার করা যেতে পারে।

---

# 10. Anomaly Detection

Normal data থেকে খুব unusual data detect করাকে anomaly detection বলে।

### উদাহরণ

একজন customer সাধারণত 500–1000 টাকা transaction করে।

হঠাৎ 2,00,000 টাকার transaction হলে সেটি anomaly হতে পারে।

ব্যবহার:

- Fraud detection
- Network attack detection
- Machine fault detection

---

# 11. Unsupervised Learning-এর সুবিধা

- Label দরকার হয় না
- Unknown pattern খুঁজে পাওয়া যায়
- Large unlabeled data-তে useful
- Customer segmentation-এ useful

---

# 12. সীমাবদ্ধতা

- Result evaluate করা কঠিন
- Cluster-এর meaning বুঝতে human interpretation দরকার
- Wrong \(K\) দিলে poor result হতে পারে
- Data scaling গুরুত্বপূর্ণ

---

# 13. Short Exam Definition

**Unsupervised Learning is a type of Machine Learning where a model learns hidden patterns from unlabeled data.**

---

# 14. Quick Revision

\[
\boxed{\text{No Label} \rightarrow \text{Find Hidden Structure}}
\]

Important topics:

\[
\boxed{\text{Clustering + PCA + Anomaly Detection}}
\]

### Final Example

Customer data-কে কোনো predefined label ছাড়াই VIP, regular এবং low-spending group-এ ভাগ করা হলো।

এটি **Unsupervised Learning**।
