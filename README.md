# 🧠 Machine Learning Projects Repository

Welcome to the *Machine Learning* project repository. This collection contains four diverse machine learning projects exploring different paradigms such as supervised learning (classification and regression), neural networks, and unsupervised learning (clustering). Each project includes datasets, implementation scripts, and visualizations.

---

## 📚 Table of Contents

1. [Project 1: Decision Tree Classifier](#project-1-decision-tree-classifier)
2. [Project 2: Regression Analysis](#project-2-regression-analysis)
3. [Project 3: Neural Networks with Keras](#project-3-neural-networks-with-keras)
4. [Project 4: Clustering Human Activity Data](#project-4-clustering-human-activity-data)
5. [Getting Started](#getting-started)
6. [Installation](#installation)
7. [License](#license)

---

## 📁 Project 1: Decision Tree Classifier

A decision tree model is trained on the **Breast Cancer Wisconsin dataset** to predict tumor malignancy.

### Highlights
- Uses entropy as the splitting criterion.
- Accuracy evaluation and visualization of decision trees.
- Hyperparameter tuning using `GridSearchCV`.
- Evaluation based on train/test accuracy vs. tree depth.

### File
- `script_Parnia_Zare.py`

---

## 📁 Project 2: Regression Analysis

This project demonstrates both **linear regression** (predicting profit based on population) and **logistic regression** (predicting hiring decision based on interview scores), with a basic introduction to **multi-class classification**.

### Highlights
- Scatter plots and regression lines for linear modeling.
- Logistic regression classifier with colored prediction plots.
- Conceptual explanation of One-vs-Rest and One-vs-One strategies.

### Files
- `RegressionAssignment.py`
- `RegressionData.csv`
- `LogisticRegressionData.csv`

---

## 📁 Project 3: Neural Networks with Keras

A fully connected feedforward neural network is implemented using **Keras** and trained on the **MNIST digit dataset**.

### Highlights
- 2 hidden layers: 300 and 100 neurons with ReLU activation.
- Output layer with 10 neurons using softmax.
- Model trained for 20 epochs.
- Prediction visualization for sample test images.

### File
- `NeuralNetworksKeras.py`

---

## 📁 Project 4: Clustering Human Activity Data

Applies unsupervised learning (KMeans) to **MHEALTH sensor data** collected from wearable devices during physical activities.

### Highlights
- Preprocessing includes Z-score outlier removal and feature scaling.
- Dimensionality reduction via PCA.
- KMeans clustering with 5 clusters.
- Achieved silhouette score: **0.66**.
- Visualized clustering in 2D space.

### Files
- `script_ParniaZare.py`
- `report_ParniaZare.pdf`

---

## 🚀 Getting Started

Each project is self-contained and executable as a standalone script. For dependencies, see the installation section below.

---

## 🔧 Installation

Make sure you have Python 3.x installed. Then install required packages:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
