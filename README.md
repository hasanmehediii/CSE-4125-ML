# CSE 4125: Introduction to Machine Learning

This repository contains the **theory notes, lab exercises, implementations, and practice materials** for **CSE 4125: Introduction to Machine Learning**.

The repository will be updated throughout the semester as we cover the course syllabus.

---

## Course Overview

Machine Learning is the study of algorithms and statistical models that enable computers to learn patterns from data and make predictions or decisions without being explicitly programmed for every task.

This course introduces the fundamental concepts of **supervised learning, unsupervised learning, regression, classification, probabilistic models, kernel methods, clustering, mixture models, and sequential models**.

---

## Course Information

| Item | Details |
|---|---|
| **Course Code** | CSE 4125 |
| **Course Title** | Introduction to Machine Learning |
| **Course Type** | Theory + Lab |
| **Repository Purpose** | Semester-wise theory notes, lab work, implementations, and practice |
| **Primary Language for Lab** | Python |
| **Suggested Environment** | Jupyter Notebook / Google Colab |

---

# Syllabus

## 1. Fundamentals of Machine Learning

- Supervised Learning
- Unsupervised Learning
- Basic machine learning workflow

### Issues in Machine Learning

- Parametric Models
- Non-parametric Models
- Curse of Dimensionality
- Overfitting
- Model Selection

---

## 2. Linear Models for Regression

- Maximum Likelihood
- Least Squares
- Regularized Least Squares
- Bias-Variance Decomposition
- Bayesian Linear Regression

---

## 3. Linear Models for Classification

- Fisher's Linear Discriminant

### Probabilistic Generative Models

- Parametric Models
  - Maximum Likelihood Approach
  - Bayesian Approach
- Non-parametric Density Estimation

---

## 4. Probabilistic Discriminative Models

- Logistic Regression
- Log-Linear Models

---

## 5. Kernel Methods

- Kernel Methods
- Sparse Kernel Machines

---

## 6. Clustering and Mixture Models

- Clustering
- Mixture Models
- Expectation-Maximization (EM) Algorithm

---

## 7. Sequential Data and Markov Models

- Sequential Data
- Markov Models

---

# Theory Roadmap

The theory section of this repository will gradually contain notes and explanations for the following topics:

| Module | Topics |
|---|---|
| **Module 01** | Introduction to Machine Learning |
| **Module 02** | Supervised and Unsupervised Learning |
| **Module 03** | Parametric vs Non-parametric Models |
| **Module 04** | Curse of Dimensionality |
| **Module 05** | Overfitting and Model Selection |
| **Module 06** | Maximum Likelihood and Least Squares |
| **Module 07** | Regularized Least Squares |
| **Module 08** | Bias-Variance Decomposition |
| **Module 09** | Bayesian Linear Regression |
| **Module 10** | Fisher's Linear Discriminant |
| **Module 11** | Probabilistic Generative Models |
| **Module 12** | Parametric Density Estimation |
| **Module 13** | Non-parametric Density Estimation |
| **Module 14** | Logistic Regression |
| **Module 15** | Log-Linear Models |
| **Module 16** | Kernel Methods |
| **Module 17** | Sparse Kernel Machines |
| **Module 18** | Clustering |
| **Module 19** | Mixture Models |
| **Module 20** | Expectation-Maximization Algorithm |
| **Module 21** | Sequential Data |
| **Module 22** | Markov Models |

---

# Lab Roadmap

The lab section will focus on implementing and experimenting with the concepts discussed in theory.

| Lab | Topic | Possible Tasks |
|---|---|---|
| **Lab 01** | Python for Machine Learning | NumPy, Pandas, Matplotlib, basic data handling |
| **Lab 02** | Data Preprocessing | Missing values, encoding, normalization, train-test split |
| **Lab 03** | Linear Regression | Implement and evaluate simple/multiple linear regression |
| **Lab 04** | Least Squares | Solve regression problems using least squares |
| **Lab 05** | Regularized Regression | Ridge/L2 and related regularization concepts |
| **Lab 06** | Bias and Variance | Observe underfitting, overfitting, and model complexity |
| **Lab 07** | Bayesian Linear Regression | Basic Bayesian regression experiment |
| **Lab 08** | Fisher's Linear Discriminant | Implement linear discriminant-based classification |
| **Lab 09** | Density Estimation | Parametric and non-parametric density estimation |
| **Lab 10** | Logistic Regression | Binary classification using logistic regression |
| **Lab 11** | Kernel Methods | Explore kernel-based learning |
| **Lab 12** | Sparse Kernel Machines | Experiment with sparse kernel-based models |
| **Lab 13** | Clustering | K-Means and clustering evaluation |
| **Lab 14** | Mixture Models | Gaussian Mixture Models |
| **Lab 15** | EM Algorithm | Implement/visualize Expectation-Maximization |
| **Lab 16** | Markov Models | Basic sequential-data and Markov-model experiments |

> The exact lab order may be adjusted according to the semester's class schedule.

---

# Suggested Repository Structure

```text
CSE-4125-Introduction-to-Machine-Learning/
│
├── README.md
│
├── theory/
│   ├── 01-introduction/
│   ├── 02-supervised-unsupervised-learning/
│   ├── 03-parametric-non-parametric-models/
│   ├── 04-curse-of-dimensionality/
│   ├── 05-overfitting-model-selection/
│   ├── 06-linear-regression/
│   ├── 07-regularization/
│   ├── 08-bias-variance/
│   ├── 09-bayesian-linear-regression/
│   ├── 10-linear-classification/
│   ├── 11-generative-models/
│   ├── 12-density-estimation/
│   ├── 13-logistic-regression/
│   ├── 14-log-linear-models/
│   ├── 15-kernel-methods/
│   ├── 16-sparse-kernel-machines/
│   ├── 17-clustering/
│   ├── 18-mixture-models-em/
│   └── 19-sequential-data-markov-models/
│
├── lab/
│   ├── lab-01/
│   ├── lab-02/
│   ├── lab-03/
│   └── ...
│
├── datasets/
│
├── assignments/
│
├── practice/
│
└── resources/
```

---

# Tools and Technologies

The lab exercises may use:

- **Python**
- **Jupyter Notebook**
- **Google Colab**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Scikit-learn**
- **SciPy**

Depending on the topic, additional libraries may be introduced during the semester.

---

# Typical Machine Learning Workflow

```text
Problem Definition
        ↓
Data Collection
        ↓
Data Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Feature Selection / Engineering
        ↓
Model Selection
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Improvement
        ↓
Final Prediction / Analysis
```

---

# Learning Goals

By the end of the course, we aim to be able to:

- Understand the difference between supervised and unsupervised learning.
- Explain parametric and non-parametric approaches.
- Understand the curse of dimensionality.
- Identify and handle overfitting.
- Understand model selection and model complexity.
- Derive and apply least-squares and maximum-likelihood methods.
- Understand regularization and bias-variance trade-offs.
- Understand Bayesian linear regression.
- Apply linear models for classification.
- Understand probabilistic generative and discriminative models.
- Implement logistic regression.
- Understand density estimation techniques.
- Understand kernel methods and sparse kernel machines.
- Apply clustering techniques.
- Understand mixture models and the EM algorithm.
- Work with sequential data and basic Markov models.

---

# Theory and Lab Organization

Each topic may contain some or all of the following:

```text
Topic/
├── README.md          # Theory summary
├── notes.md           # Detailed notes
├── examples.ipynb     # Worked examples
├── implementation.ipynb
├── exercises.md       # Practice problems
└── assets/            # Figures/diagrams if needed
```

For lab work:

```text
lab-XX/
├── README.md
├── problem.md
├── solution.ipynb
├── dataset/
└── output/
```

---

# Progress Tracker

- [ ] Supervised and Unsupervised Learning
- [ ] Parametric and Non-parametric Models
- [ ] Curse of Dimensionality
- [ ] Overfitting
- [ ] Model Selection
- [ ] Maximum Likelihood
- [ ] Least Squares
- [ ] Regularized Least Squares
- [ ] Bias-Variance Decomposition
- [ ] Bayesian Linear Regression
- [ ] Fisher's Linear Discriminant
- [ ] Probabilistic Generative Models
- [ ] Parametric Density Estimation
- [ ] Non-parametric Density Estimation
- [ ] Logistic Regression
- [ ] Log-Linear Models
- [ ] Kernel Methods
- [ ] Sparse Kernel Machines
- [ ] Clustering
- [ ] Mixture Models
- [ ] Expectation-Maximization Algorithm
- [ ] Sequential Data
- [ ] Markov Models

---

# How to Use This Repository

Clone the repository:

```bash
git clone <your-repository-url>
cd CSE-4125-Introduction-to-Machine-Learning
```

Create a virtual environment if needed:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the common machine learning packages:

```bash
pip install numpy pandas matplotlib scipy scikit-learn jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

---

# Notes

- This repository is intended primarily for **academic and learning purposes**.
- Materials will be added and updated as the semester progresses.
- Theory explanations and implementations may be refined after lectures and lab sessions.
- Some topics may contain both **from-scratch implementations** and **library-based implementations** for better understanding.

---

## Course Syllabus Reference

> **CSE 4125: Introduction to Machine Learning**  
> Supervised and Unsupervised Learning; issues in machine learning: parametric and non-parametric models, curse of dimensionality, over-fitting, and model selection. Linear Models for Regression: Maximum Likelihood and least squares, regularized least squares, Bias variance decomposition, Bayesian linear regression. Linear Models for classification: Fisher's linear discriminant, probabilistic generative models—parametric (maximum likelihood and Bayesian) and non-parametric density estimation. Probabilistic discriminative models: logistic regression, log-linear models, Kernel methods and Sparse Kernel Machines. Clustering, mixture models and Expectation Maximization algorithm. Sequential data and Markov models.

---

## Contributions

This repository is maintained for our semester coursework. Suggestions, corrections, and improvements are welcome through issues or pull requests.

---

## License

This repository is intended for educational use. Add a license file if you plan to distribute or reuse the materials publicly.

---

**CSE 4125 — Introduction to Machine Learning**
