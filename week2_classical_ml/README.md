# Week 2: Classical Machine Learning

A comprehensive curriculum covering fundamental machine learning algorithms, starting with Linear Regression.

## Overview

This week introduces classical machine learning concepts and algorithms:

| Day | Topic | Assignment |
|-----|-------|------------|
| 8 | Linear Regression | House price prediction - manual vs sklearn |
| 9 | Logistic Regression | Binary classification tasks |
| 10 | Decision Trees | Tree-based classification and regression |
| 11 | Random Forests | Ensemble methods |
| 12 | Support Vector Machines | SVM classification |
| 13 | Model Evaluation | Cross-validation, metrics, tuning |
| 14 | Week 2 Project | Complete ML pipeline |

## Directory Structure

```
week2_classical_ml/
├── day08_linear_regression/
│   └── day08_linear_regression.ipynb     # Linear regression from scratch and sklearn
├── day09_logistic_regression/            # (Coming soon)
├── day10_decision_trees/                 # (Coming soon)
├── day11_random_forests/                 # (Coming soon)
├── day12_svm/                           # (Coming soon)
├── day13_model_evaluation/              # (Coming soon)
├── day14_week2_project/                 # (Coming soon)
├── requirements.txt                      # Python dependencies
└── README.md                            # This file
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

### 2. Run Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

## Day 8: Linear Regression (Available Now)

### Topics Covered

1. **Introduction to Linear Regression**
   - What is linear regression?
   - Simple vs multiple linear regression
   - Use cases and applications

2. **Mathematical Foundations**
   - The equation: y = mx + b
   - Matrix notation for multiple features
   - Ordinary Least Squares (OLS) solution
   - Normal equation

3. **Cost Function and Optimization**
   - Mean Squared Error (MSE)
   - Why MSE is used
   - Cost function surface visualization

4. **Gradient Descent**
   - Algorithm explanation
   - Update rules
   - Learning rate effects
   - Convergence visualization

5. **Implementation from Scratch**
   - Custom LinearRegression class
   - OLS solution implementation
   - Gradient descent implementation
   - Comparison of methods

6. **Scikit-learn Implementation**
   - LinearRegression class
   - fit(), predict(), score() methods
   - Comparison with manual implementation

7. **Model Evaluation Metrics**
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - R-squared (R²) score
   - Manual implementation of all metrics

8. **House Price Prediction Assignment**
   - Real-world dataset creation
   - Feature analysis and EDA
   - Model building (manual vs sklearn)
   - Cross-validation
   - Residual analysis

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

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
```

### Assignment: House Price Prediction

Build a house price prediction model using:
1. Manual implementation (gradient descent)
2. Scikit-learn implementation

Compare:
- Training and test performance
- Coefficient values
- Convergence behavior
- Residual analysis

## Learning Outcomes

By completing Day 8, you will:

1. **Understand Linear Regression Math**
   - Derive the OLS solution
   - Understand gradient descent optimization
   - Interpret model coefficients

2. **Implement Algorithms from Scratch**
   - Build a complete linear regression class
   - Implement cost functions
   - Debug gradient descent convergence

3. **Use Scikit-learn Effectively**
   - Understand the API design
   - Compare with manual implementation
   - Use cross-validation

4. **Evaluate Regression Models**
   - Calculate and interpret metrics
   - Analyze residuals
   - Identify model issues

5. **Build Practical Applications**
   - Work with real-world features
   - Handle data preprocessing
   - Create production-ready models

## Tips for Success

1. **Understand the math first** before looking at code
2. **Visualize** cost functions and gradients
3. **Experiment** with learning rates
4. **Compare** manual and sklearn implementations
5. **Analyze residuals** to check model assumptions
6. **Use cross-validation** for robust evaluation

## Prerequisites

Before starting Week 2, ensure you understand:
- Python programming fundamentals
- NumPy array operations
- Pandas DataFrame manipulation
- Basic statistics (mean, variance, correlation)
- Feature engineering concepts (Week 1)

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

### Memory Issues

```python
# For large datasets
# Use SGD instead of batch gradient descent
from sklearn.linear_model import SGDRegressor
model = SGDRegressor(max_iter=1000, tol=1e-3)
```

## Next Steps

After completing this notebook:
1. Try different datasets (Boston Housing, California Housing)
2. Implement polynomial regression
3. Explore regularization (Ridge, Lasso)
4. Move on to Day 9: Logistic Regression

---

Happy Learning!
