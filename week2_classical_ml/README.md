# Week 2: Classical Machine Learning

A comprehensive curriculum covering fundamental machine learning algorithms with hands-on implementations from scratch and using scikit-learn.

## Overview

This week introduces classical machine learning concepts and algorithms:

| Day | Topic | Key Concepts | Assignment |
|-----|-------|--------------|------------|
| 8 | Linear Regression | y=mx+b, Cost function, Gradient descent | House price prediction |
| 9 | Logistic Regression | Sigmoid, Binary classification, ROC/AUC | Spam email classifier (>85% acc) |
| 10 | Decision Trees | Gini impurity, Information gain, Pruning | Loan approval prediction |
| 11 | Random Forests | Bagging, Boosting, Feature importance | Improve Day 10 model |

## Directory Structure

```
week2_classical_ml/
├── day08_linear_regression/
│   └── linear_regression.py      # Linear regression from scratch + sklearn
├── day09_logistic_regression/
│   └── logistic_regression.py    # Logistic regression + spam classifier
├── day10_decision_trees/
│   └── decision_trees.py         # Decision trees + loan approval
├── day11_random_forests/
│   └── random_forests.py         # Ensemble methods + improved model
├── requirements.txt
└── README.md
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

### 2. Run Scripts

```bash
python day08_linear_regression/linear_regression.py
python day09_logistic_regression/logistic_regression.py
python day10_decision_trees/decision_trees.py
python day11_random_forests/random_forests.py
```

## Day 8: Linear Regression

### Topics Covered
- **Mathematical foundations**: y = mx + b (simple), y = Xw (matrix form)
- **Cost function**: Mean Squared Error (MSE)
- **Optimization**: Gradient descent and Normal equation
- **Evaluation metrics**: MSE, RMSE, MAE, R² score

### Key Implementations
- `LinearRegressionScratch` class with gradient descent and normal equation
- Comparison with scikit-learn's `LinearRegression`
- House price prediction using California Housing dataset

### Key Code Examples

```python
# Manual OLS Implementation
def calculate_ols_parameters(X, y):
    x_mean, y_mean = np.mean(X), np.mean(y)
    m = np.sum((X - x_mean) * (y - y_mean)) / np.sum((X - x_mean) ** 2)
    b = y_mean - m * x_mean
    return m, b

# Gradient Descent
def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m, b = np.random.randn(), np.random.randn()
    for i in range(n_iterations):
        y_pred = m * X + b
        dm = -(2/n) * np.sum(X * (y - y_pred))
        db = -(2/n) * np.sum(y - y_pred)
        m -= learning_rate * dm
        b -= learning_rate * db
    return m, b

# Scikit-learn
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Day 9: Logistic Regression

### Topics Covered
- **Sigmoid function**: σ(z) = 1 / (1 + e^(-z))
- **Binary cross-entropy loss**: Log loss for classification
- **Decision boundary**: Where P(y=1|x) = 0.5
- **Evaluation**: Confusion matrix, Precision, Recall, F1, ROC/AUC

### Key Implementations
- `LogisticRegressionScratch` class with gradient descent
- TF-IDF feature extraction for text classification
- Spam email classifier achieving >85% accuracy

## Day 10: Decision Trees

### Topics Covered
- **Splitting criteria**: Gini impurity, Information gain (Entropy)
- **Tree construction**: Recursive splitting algorithm
- **Pruning**: Pre-pruning (max_depth, min_samples) and post-pruning (ccp_alpha)
- **Overfitting**: Understanding and preventing

### Key Implementations
- `DecisionTreeScratch` class with Gini/Entropy
- Tree visualization and decision boundary plots
- Loan approval prediction system

## Day 11: Random Forests & Ensemble Methods

### Topics Covered
- **Bagging**: Bootstrap aggregating to reduce variance
- **Random Forest**: Bagging + random feature selection
- **Boosting**: AdaBoost, Gradient Boosting
- **Feature importance**: Analyzing which features matter most

### Key Implementations
- `RandomForestScratch` class
- Comparison: Decision Tree vs Random Forest vs Gradient Boosting
- Improved loan approval model with ensemble methods

## Key Takeaways

### Linear Regression (Day 8)
1. Linear regression minimizes MSE to find the best-fit line
2. Gradient descent iteratively updates weights
3. Normal equation provides closed-form solution
4. Feature scaling is crucial for gradient descent

### Logistic Regression (Day 9)
1. Sigmoid function outputs probabilities (0, 1)
2. Log loss is convex, unlike MSE for classification
3. Precision/Recall trade-off depends on application
4. ROC AUC measures discriminative ability

### Decision Trees (Day 10)
1. Trees split using Gini impurity or Information Gain
2. Deeper trees overfit - use pruning!
3. Trees create axis-parallel decision boundaries
4. Easy to interpret and visualize

### Random Forests (Day 11)
1. Bagging reduces variance through averaging
2. Random Forest decorrelates trees
3. More trees improve stability (diminishing returns)
4. Ensemble methods provide more robust predictions

## Output Visualizations

Each script generates visualization plots saved in respective directories:
- `simple_linear_regression.png`, `house_price_prediction.png`
- `sigmoid_function.png`, `decision_boundary.png`, `spam_classifier.png`
- `splitting_criteria.png`, `decision_boundaries.png`, `loan_tree_visualization.png`
- `ensemble_comparison.png`, `n_estimators_effect.png`, `loan_improvement.png`

## Troubleshooting

### Gradient Descent Not Converging

```python
# Check learning rate
# Too high: cost increases/oscillates
# Too low: very slow convergence

# Try these values
learning_rates = [0.001, 0.01, 0.1]

# Scale features first!
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Poor Model Performance

```python
# Check for:
# 1. Feature scaling issues
# 2. Non-linear relationships (try polynomial features)
# 3. Outliers affecting the fit
# 4. Multicollinearity between features

# Visualize residuals
plt.scatter(y_pred, y_test - y_pred)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residual')
```

## Resources

### Documentation
- [Scikit-learn Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)

### Books
- "Introduction to Statistical Learning" - James et al.
- "Hands-On Machine Learning" - Aurelien Geron

### Videos
- [3Blue1Brown - Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [StatQuest - Linear Regression](https://www.youtube.com/watch?v=7ArmBVF2dCs)

## Requirements

```bash
pip install -r requirements.txt
```

## Next Steps

After completing Week 2, proceed to:
- **Week 3**: Deep Learning (Neural Networks, CNNs, RNNs)
- **Week 4**: Advanced ML (NLP, Transformers, Deployment)

Or explore further:
1. Try different datasets (Boston Housing, California Housing)
2. Implement polynomial regression
3. Explore regularization (Ridge, Lasso)
