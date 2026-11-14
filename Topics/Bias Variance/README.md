# Bias–Variance — সহজ বাংলা রিভিশন নোট

> **মূল ধারণা:** Model কি আসল সম্পর্ক ধরতে পারছে না, নাকি training data সামান্য বদলালেই prediction অনেক বদলে যাচ্ছে? প্রথমটি high bias-এর, দ্বিতীয়টি high variance-এর ধারণা।

## 1. Bias ও Variance কী?

ধরি, একই population থেকে একই size-এর অনেক training dataset নিতে পারলাম। প্রতিটিতে একই learning procedure চালিয়ে model fit করলাম। একটি **স্থির test input**-এ তাদের predictions তুলনা করব।

| ধারণা | প্রশ্ন | সহজ অর্থ |
| --- | --- | --- |
| Bias | গড় prediction আসল function থেকে কত দূরে? | Systematic prediction error |
| Variance | Training dataset বদলালে prediction কতটা বদলায়? | Data পরিবর্তনের প্রতি model-এর sensitivity |
| Noise | Features জানা থাকলেও target কতটা অনিশ্চিত? | বর্তমান তথ্যের বাইরে থাকা randomness |

এখানে **bias মানে neuron-এর bias parameter $b$ নয়**, আর variance মানে input feature-এর variance-ও নয়। এটি learning procedure-এর predictions নিয়ে আলোচনা।

## 2. একটি ছবি দিয়ে চারটি অবস্থা

ধরি, একটি নির্দিষ্ট input-এ আসল function value $f(x)=10$। নিচের প্রতি সারিতে আলাদা training dataset থেকে fit করা model-গুলোর predictions দেখানো হয়েছে।

```text
আসল মান = 10; প্রতি সারিতে তিনটি prediction

Low bias, low variance:    9.8 ── 10 ── 10.2       → গড় 10
Low bias, high variance:   6 ───── 10 ───── 14      → গড় 10
High bias, low variance:   5.8 ── 6 ── 6.2         → গড় 6
High bias, high variance:  2 ───── 6 ───── 10       → গড় 6

সংখ্যাগুলোর মাঝের রেখা ছড়িয়ে থাকার ধারণা বোঝায়।
```

গড় prediction আসল মানের কাছে হলে bias কম। Predictions পরস্পরের কাছাকাছি হলে variance কম। তাই একটি model **নিয়মিত একই ভুল** করতে পারে—সেটি low variance হলেও high bias।

## 3. Underfitting ও Overfitting

**Underfitting:** Model বা শেখার procedure data-র গুরুত্বপূর্ণ সম্পর্ক ধরতে পারছে না। যেমন সত্য সম্পর্ক বাঁকা, কিন্তু শুধু সরলরেখা দিয়ে fit করার চেষ্টা।

**Overfitting:** Training sample-এর noise বা বিশেষ বৈশিষ্ট্যও model শিখে ফেলছে; নতুন data-তে সেই সম্পর্ক কাজ করছে না।

| লক্ষণ | Underfitting-এর সঙ্গে সাধারণ সম্পর্ক | Overfitting-এর সঙ্গে সাধারণ সম্পর্ক |
| --- | --- | --- |
| Training error | তুলনামূলক বেশি | তুলনামূলক কম |
| Validation error | তুলনামূলক বেশি | Training error-এর চেয়ে অনেক বেশি |
| Bias–Variance ধারণা | প্রায়ই high bias | প্রায়ই high variance |
| Model flexibility | কাজের তুলনায় কম হতে পারে | Data-এর তুলনায় বেশি হতে পারে |

এগুলো **diagnostic clues**, নিশ্চিত প্রমাণ নয়। Training ঠিকমতো শেষ না হওয়া, data leakage বা train–validation distribution আলাদা হওয়াও error pattern বদলাতে পারে।

## 4. Bias–Variance decomposition-এর assumptions

নিচের সমীকরণটি **squared-error regression**-এর জন্য। একটি স্থির input $x_0$-এ:

$$
y_0=f(x_0)+\epsilon,\qquad f(x_0)=\mathbb E[y_0\mid x_0]
$$

ধরি, test noise-এর mean zero, variance $\sigma_\epsilon^2(x_0)$ এবং test noise training dataset $D$ থেকে independent। $D$-তে fit করা prediction হলো $\hat f_D(x_0)$।

Dataset বদলালে গড় prediction:

$$
\bar f(x_0)=\mathbb E_D[\hat f_D(x_0)]
$$

**Bias:**

$$
\boxed{\operatorname{Bias}(x_0)=\bar f(x_0)-f(x_0)}
$$

**Variance:**

$$
\boxed{\operatorname{Var}(x_0)
=\mathbb E_D\left[(\hat f_D(x_0)-\bar f(x_0))^2\right]}
$$

যদি learning algorithm নিজেও random হয়, training randomness-ও এই expectation-এর মধ্যে ধরতে হবে।

## 5. সবচেয়ে গুরুত্বপূর্ণ সমীকরণ

$$
\boxed{
\mathbb E_{D,\epsilon}\left[(y_0-\hat f_D(x_0))^2\right]
=\operatorname{Bias}(x_0)^2+\operatorname{Var}(x_0)+\sigma_\epsilon^2(x_0)
}
$$

অর্থাৎ:

$$
\boxed{\text{Expected Test MSE}=\text{Bias}^2+\text{Variance}+\text{Noise}}
$$

এখানে একটি স্থির $x_0$ নিয়ে কথা বলা হয়েছে। Test inputs-এর distribution-এর ওপরও average নিলে overall expected test MSE পাওয়া যায়। এটি একটি training run-এর observed error-এর হুবহু তিনটি জানা অংশ নয়।

> **পরীক্ষায় মনে রাখো:** সূত্রে **bias-এর square** আছে, bias নিজে নয়। এই exact additive decomposition সরাসরি classification accuracy, F1 বা log loss-এর সূত্র হিসেবে ব্যবহার করবে না।

## 6. সূত্রটি কোথা থেকে আসে?

সংক্ষেপে $f=f(x_0)$, $\hat f=\hat f_D(x_0)$ এবং $\bar f=\mathbb E_D\hat f$ লিখি।

$$
y_0-\hat f=(f-\bar f)+(\bar f-\hat f)+\epsilon
$$

Square করে expectation নিলে:

$$
\mathbb E[(y_0-\hat f)^2]
=(f-\bar f)^2+\mathbb E[(\bar f-\hat f)^2]+\mathbb E[\epsilon^2]
$$

Cross terms zero হয়, কারণ $\mathbb E_D[\bar f-\hat f]=0$, noise-এর mean zero এবং noise training থেকে independent। তিনটি বাকি অংশই যথাক্রমে **Bias², Variance, Noise**।

## 7. হাতে করা উদাহরণ: dataset বদলালে prediction

একটি স্থির input-এ সত্য function value **10** এবং test noise variance **1**। ধরি তিনটি training dataset সমান probability-তে পাওয়া যায়।

| Learning procedure | Dataset 1-এর prediction | Dataset 2 | Dataset 3 | গড় prediction |
| --- | ---: | ---: | ---: | ---: |
| A | 6 | 6 | 6 | 6 |
| B | 8 | 10 | 12 | 10 |
| C | 9 | 10 | 11 | 10 |

**Procedure A:** সবসময় 6 বলে।

$$
\text{Bias}=6-10=-4,\quad \text{Bias}^2=16,\quad \text{Variance}=0
$$

$$
\text{Expected Test MSE}=16+0+1=17
$$

**Procedure B:** গড় prediction ঠিক, কিন্তু dataset বদলালে prediction ছড়াচ্ছে।

$$
\text{Bias}^2=0,\qquad
\text{Variance}=\frac{(8-10)^2+(10-10)^2+(12-10)^2}{3}=\frac83
$$

$$
\text{Expected Test MSE}=0+\frac83+1=\frac{11}{3}\approx3.6667
$$

**Procedure C:** গড় prediction ঠিক এবং variation-ও কম।

$$
\text{Bias}^2=0,\quad \text{Variance}=\frac{1+0+1}{3}=\frac23,\quad
\text{Expected Test MSE}=\frac53\approx1.6667
$$

এখানে তিনটি dataset-কে সম্পূর্ণ, সমান-probability distribution ধরে expectation হিসাব করেছি, তাই denominator **3**। এটি finite sample থেকে unbiased variance estimate করার $n-1$ সূত্র নয়।

| Procedure | Bias² | Variance | Noise | Expected Test MSE |
| --- | ---: | ---: | ---: | ---: |
| A | 16 | 0 | 1 | 17 |
| B | 0 | 2.6667 | 1 | 3.6667 |
| C | 0 | 0.6667 | 1 | 1.6667 |

**শিক্ষা:** Zero bias হলেই error zero নয়; low variance হলেই model ভালো নয়। দুটো অংশ এবং noise একসঙ্গে দেখতে হবে।

## 8. Model complexity ও trade-off

অনেক প্রচলিত setting-এ model বেশি flexible করলে bias কমতে পারে, কিন্তু সীমিত data-তে variance বাড়তে পারে।

```text
Error
  ^
  |  Validation/Test error-এর ধারণামূলক curve
  |  \                         /
  |   \                       /
  |    \______         ______/
  |           \_______/
  |
  |  Training error: \_____________
  +------------------------------------> Model complexity
     Underfit       ভালো অঞ্চল      Overfit
```

এটি প্রচলিত একটি ধারণামূলক চিত্র, সব model ও dataset-এ হুবহু U-shaped curve হওয়ার নিয়ম নয়।

উদাহরণ: Polynomial regression-এ degree খুব কম হলে বাঁকা সম্পর্ক ধরতে না-ও পারে। খুব বেশি degree সীমিত noisy sample-এ অপ্রয়োজনীয় বাঁক তৈরি করতে পারে। Degree **validation/CV দিয়ে** বেছে নিতে হবে।

## 9. Regularization-এর প্রভাব

একটি L2-regularized objective:

$$
J(w)=\frac1m\sum_{i=1}^{m}(y_i-\hat y_i)^2+\lambda\|w\|_2^2
$$

এই convention-এ penalty gradient $2\lambda w$। অন্য বইয়ে $\lambda/2$ থাকলে gradient $\lambda w$ হবে।

- $\lambda$ বাড়ালে weights ছোট রাখার চাপ বাড়ে।
- অনেক setting-এ এতে variance কমে, কিন্তু bias বাড়তে পারে।
- অতিরিক্ত regularization underfitting ঘটাতে পারে।
- সেরা $\lambda$ training loss দিয়ে নয়, validation/CV দিয়ে বেছে নাও।

## 10. Learning curve: data বাড়ালে কী হয়?

**Learning curve**-এ x-axis = training sample সংখ্যা, y-axis = training ও validation error। এটি complexity curve থেকে আলাদা।

| Pattern | সম্ভাব্য ব্যাখ্যা | কী চেষ্টা করা যায়? |
| --- | --- | --- |
| Train ও validation error দুটোই বেশি, gap ছোট | High bias বা optimization সমস্যা | Features/model উন্নত করা, regularization কমানো, training ঠিক হচ্ছে কি না দেখা |
| Train error কম, validation error বেশি; gap বড় | High variance বা distribution সমস্যা | বেশি উপযুক্ত data, regularization, model সরল করা, split পরীক্ষা |
| Data বাড়লে gap ছোট হচ্ছে | Variance কমার সম্ভাবনা | আরও data কাজে লাগতে পারে |

একই scale ও metric-এ curves তুলনা করতে হবে। শুধু data বাড়ালে একটি ভুল model assumption-এর systematic bias সবসময় দূর হয় না।

## 11. Bias বা Variance কমাতে কী করতে পারি?

| লক্ষ্য | সাধারণ উপায় | খেয়াল রাখবে |
| --- | --- | --- |
| High bias কমানো | ভালো features, বেশি উপযুক্ত/flexible model, কম regularization | Model বড় করলেই optimization সফল হবে না |
| High variance কমানো | আরও representative data, regularization, কম flexibility | Validation score দিয়ে প্রভাব যাচাই করো |
| Variance কমাতে averaging | Bagging বা ensemble | Model-গুলোর errors খুব correlated হলে লাভ কম |
| Neural Network-এ overfitting কমানো | Early stopping, dropout, উপযুক্ত data augmentation | Validation data দিয়ে settings বাছতে হবে |
| Noise কমানো | ভালো measurement, label quality, আরও informative features | Noise-এর সংজ্ঞা বর্তমান features-এর ওপর নির্ভরশীল |

**Bagging-এর ধারণা:** একাধিক fitted model-এর predictions average করলে variation কমতে পারে। যদি $B$টি predictor-এর প্রতিটির variance $v$ এবং সব pair-এর correlation $\rho$ হয়:

$$
\operatorname{Var}\left(\frac1B\sum_{b=1}^{B}\hat f_b\right)
=v\left(\rho+\frac{1-\rho}{B}\right)
$$

এটি সমান variance ও common pairwise correlation-এর assumptions-এ। Independent predictors হলে $\rho=0$, variance $v/B$। সব predictors একই হলে $\rho=1$, averaging-এ variance কমে না।

## 12. Irreducible error মানে কী?

$f(x)=\mathbb E[y\mid x]$ জানা থাকলেও একটি individual target $y$ তার থেকে ভিন্ন হতে পারে। **বর্তমান feature set ও data-generating process ধরে** noise variance সেই অবশিষ্ট uncertainty।

একই features রেখে শুধু model জটিল করে এই অংশ পুরো দূর করা যায় না। তবে নতুন তথ্যসমৃদ্ধ feature বা ভালো measurement পেলে conditional noise বদলাতে পারে। তাই “irreducible” মানে কোনো অবস্থাতেই উন্নতি সম্ভব নয়, এমন নয়।

## 13. Cross Validation-এর সঙ্গে সম্পর্ক

CV নতুন data-তে error estimate ও model selection করতে সাহায্য করে। যেমন degree বা $\lambda$ বদলে CV score দেখলে complexity-এর উপযুক্ত মাত্রা খুঁজে পাওয়া যায়।

কিন্তু একটি CV run থেকে সরাসরি **Bias² এবং Variance আলাদা করে পাওয়া যায় না**। আসল function $f$ সাধারণত অজানা; CV fold score-এর SD-ও fixed-input prediction variance-এর সমান নয়।

## 14. সাধারণ ভুল ও এক মিনিটের রিভিশন

- Bias² + Variance + Noise হলো expected squared-error-এর decomposition; সব metric-এর জন্য নয়।
- একটি dataset-এ কম training error মানেই low bias বা ভালো generalization নিশ্চিত নয়।
- High bias ও high variance একই সঙ্গে থাকতে পারে।
- Regularization বাড়ানো বা data বাড়ানোর প্রভাব problem অনুযায়ী যাচাই করতে হবে।
- Learning curve-এর x-axis sample সংখ্যা; complexity curve-এর x-axis model flexibility।

| প্রশ্ন | উত্তর |
| --- | --- |
| Bias কী? | গড় prediction − সত্য function value |
| Variance কী? | Dataset বদলালে prediction-এর variation |
| মূল সূত্র? | Expected Test MSE = Bias² + Variance + Noise |
| Underfitting-এর সাধারণ সম্পর্ক? | High bias |
| Overfitting-এর সাধারণ সম্পর্ক? | High variance |
| Regularization-এর সাধারণ trade-off? | Variance কমতে পারে, bias বাড়তে পারে |
| শুধু CV score দিয়ে bias ও variance আলাদা পাব? | না |
| লক্ষ্য কী? | Generalization error কমানো; শুধু training error নয় |

**মনে রাখার লাইন:** **গড়ে কতটা ভুল = Bias; data বদলালে কতটা বদলায় = Variance**।

**সম্পর্কিত নোট:** [Cross Validation](../Cross%20Validation/README.md) · [Evaluation Metrics](../Evaluation%20Metrics/README.md) · [Linear Regression](../Linear%20Regression/README.md) · [Neural Network](../Neural%20Network/README.md)
