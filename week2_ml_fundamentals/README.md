# Week 2: Machine Learning Fundamentals

A comprehensive curriculum covering SVM, clustering, and complete ML pipelines.

## Overview

This week covers essential machine learning algorithms and practical projects:

| Day | Topic | Assignment |
|-----|-------|------------|
| 12 | Support Vector Machines | MNIST digit classification |
| 13 | K-Means & Clustering | Customer segmentation |
| 14 | Week 2 Project | Kaggle Titanic competition pipeline |

## Directory Structure

```
week2_ml_fundamentals/
├── day12_svm/
│   └── svm_digit_classification.py    # SVM, kernels, hyperparameter tuning
├── day13_kmeans_clustering/
│   └── kmeans_customer_segmentation.py # K-Means, silhouette, hierarchical
├── day14_week2_project/
│   └── titanic_ml_pipeline.py         # Complete ML pipeline
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Individual Days

```bash
# Day 12: SVM Digit Classification
python day12_svm/svm_digit_classification.py

# Day 13: K-Means Customer Segmentation
python day13_kmeans_clustering/kmeans_customer_segmentation.py

# Day 14: Titanic ML Pipeline
python day14_week2_project/titanic_ml_pipeline.py
```

## Day-by-Day Content

### Day 12: Support Vector Machines (SVM)

**Concepts:**
- Maximum margin classifier
- Support vectors and decision boundaries
- C parameter (regularization)
- Kernel trick: Linear, RBF, Polynomial
- Gamma parameter for RBF kernel
- When to use SVM vs other algorithms

**Key Code:**
```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# RBF kernel SVM with hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}

svm = SVC(kernel='rbf')
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

**Outputs:**
- svm_margins_c_parameter.png
- svm_kernels_comparison.png
- svm_gamma_effect.png
- svm_confusion_matrix.png

### Day 13: K-Means & Clustering

**Concepts:**
- Unsupervised learning fundamentals
- K-Means algorithm steps: init, assign, update
- Choosing K: Elbow method, Silhouette score
- Hierarchical clustering and dendrograms
- Cluster interpretation for business insights

**Key Code:**
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Find optimal K using silhouette
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"K={k}: Silhouette = {score:.4f}")
```

**Outputs:**
- kmeans_steps.png
- elbow_method.png
- silhouette_scores.png
- customer_segments.png
- segment_heatmap.png

### Day 14: Week 2 Project

**Complete ML Pipeline:**
1. Data Loading & Exploration
2. Feature Engineering
3. Preprocessing & Missing Values
4. Model Comparison (8 algorithms)
5. Hyperparameter Tuning (GridSearchCV)
6. Final Model Evaluation
7. Ensemble Methods
8. Kaggle Submission

**Key Code:**
```python
# Compare multiple models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(probability=True),
    # ... more models
}

# Cross-validation comparison
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {cv_scores.mean():.4f}")
```

**Outputs:**
- titanic_eda.png
- model_comparison.png
- final_model_evaluation.png
- titanic_submission.csv

## Learning Outcomes

By completing this week, you will:

1. **Understand SVM fundamentals**
   - How maximum margin classifiers work
   - The kernel trick for non-linear boundaries
   - Tuning C and gamma parameters

2. **Master clustering techniques**
   - K-Means algorithm implementation
   - Methods for choosing optimal K
   - Interpreting clusters for business insights

3. **Build complete ML pipelines**
   - End-to-end workflow from data to submission
   - Model comparison and selection
   - Hyperparameter optimization

4. **Apply best practices**
   - Feature engineering
   - Cross-validation
   - Model documentation

## Algorithm Decision Guide

### When to Use SVM
- High-dimensional data (text, images)
- Clear margin of separation
- Small to medium datasets (<10K samples)
- Binary classification

### When to Use K-Means
- Customer segmentation
- Document clustering
- Image compression
- Anomaly detection (with modifications)

### When to Use Ensemble Methods
- Kaggle competitions
- Production systems requiring robustness
- When single models plateau

## Tips for Success

1. **Run each day's code** before moving to the next
2. **Experiment** with different hyperparameters
3. **Visualize** your data and results
4. **Document** what works and why
5. **Cross-validate** to avoid overfitting

## Troubleshooting

### Memory Issues
```python
# For large datasets, use incremental learning
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='hinge')  # SVM-like
```

### Slow Training
```python
# Use fewer hyperparameter combinations
# Use RandomizedSearchCV instead of GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
```

### Poor Clustering Results
```python
# Try different scalers
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()  # Better for outliers
```

## Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [SVM Tutorial](https://scikit-learn.org/stable/modules/svm.html)
- [Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html)
- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)

## License

This educational content is provided for learning purposes.

---

Happy Learning!
