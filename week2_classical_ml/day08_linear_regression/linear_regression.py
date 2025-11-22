"""
DAY 8: Linear Regression
========================
Topics Covered:
- Math fundamentals: y = mx + b, cost function, gradient descent
- Implementation from scratch using NumPy
- Using Scikit-learn's LinearRegression
- Evaluation metrics: MSE, RMSE, R² score
- Assignment: House price prediction (manual vs sklearn comparison)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)


# =============================================================================
# PART 1: MATHEMATICAL FOUNDATIONS
# =============================================================================

def explain_linear_regression():
    """
    Linear Regression Theory
    ========================

    Simple Linear Regression: y = mx + b
    - y: predicted output (dependent variable)
    - x: input feature (independent variable)
    - m: slope (weight/coefficient)
    - b: y-intercept (bias)

    Multiple Linear Regression: y = w0 + w1*x1 + w2*x2 + ... + wn*xn
    - Can be written as: y = X @ W (matrix notation)

    Cost Function (Mean Squared Error):
    J(w) = (1/2m) * sum((y_pred - y_actual)^2)

    Goal: Minimize the cost function by finding optimal weights

    Gradient Descent:
    - Update rule: w = w - learning_rate * gradient
    - gradient = (1/m) * X.T @ (predictions - y)
    """
    print("=" * 60)
    print("LINEAR REGRESSION THEORY")
    print("=" * 60)
    print("""
    1. EQUATION: y = mx + b (simple) or y = Xw (matrix form)

    2. COST FUNCTION (MSE):
       J(w) = (1/2m) * Σ(y_pred - y_actual)²

    3. GRADIENT DESCENT:
       Repeat until convergence:
           w := w - α * ∂J/∂w
       where α is the learning rate

    4. NORMAL EQUATION (closed-form solution):
       w = (X^T * X)^(-1) * X^T * y
    """)


# =============================================================================
# PART 2: LINEAR REGRESSION FROM SCRATCH
# =============================================================================

class LinearRegressionScratch:
    """
    Linear Regression implemented from scratch using NumPy.
    Supports both gradient descent and normal equation methods.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, method='gradient_descent'):
        """
        Initialize the Linear Regression model.

        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent
        n_iterations : int
            Number of iterations for gradient descent
        method : str
            'gradient_descent' or 'normal_equation'
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.weights = None
        self.bias = None
        self.cost_history = []

    def _compute_cost(self, X, y, predictions):
        """
        Compute Mean Squared Error cost function.

        J(w) = (1/2m) * sum((predictions - y)^2)
        """
        m = len(y)
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost

    def _gradient_descent(self, X, y):
        """
        Perform gradient descent to find optimal weights.

        Update rules:
        w := w - α * (1/m) * X.T @ (predictions - y)
        b := b - α * (1/m) * sum(predictions - y)
        """
        m, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            # Forward pass: compute predictions
            predictions = X @ self.weights + self.bias

            # Compute cost for monitoring
            cost = self._compute_cost(X, y, predictions)
            self.cost_history.append(cost)

            # Compute gradients
            dw = (1 / m) * (X.T @ (predictions - y))
            db = (1 / m) * np.sum(predictions - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.n_iterations}, Cost: {cost:.6f}")

    def _normal_equation(self, X, y):
        """
        Compute weights using the normal equation (closed-form solution).

        w = (X^T * X)^(-1) * X^T * y

        Note: We add a column of ones to X for the bias term.
        """
        # Add bias column (column of ones)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Normal equation
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        # Extract bias and weights
        self.bias = theta[0]
        self.weights = theta[1:]

        print("Weights computed using Normal Equation (closed-form solution)")

    def fit(self, X, y):
        """
        Fit the linear regression model to training data.

        Parameters:
        -----------
        X : numpy array of shape (m_samples, n_features)
        y : numpy array of shape (m_samples,)
        """
        if self.method == 'gradient_descent':
            self._gradient_descent(X, y)
        else:
            self._normal_equation(X, y)

        return self

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        -----------
        X : numpy array of shape (m_samples, n_features)

        Returns:
        --------
        predictions : numpy array of shape (m_samples,)
        """
        return X @ self.weights + self.bias

    def get_params(self):
        """Return model parameters."""
        return {'weights': self.weights, 'bias': self.bias}


# =============================================================================
# PART 3: EVALUATION METRICS
# =============================================================================

def compute_metrics(y_true, y_pred):
    """
    Compute regression evaluation metrics.

    Parameters:
    -----------
    y_true : actual values
    y_pred : predicted values

    Returns:
    --------
    dict : Dictionary containing MSE, RMSE, MAE, R² score
    """
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))

    # R² Score (Coefficient of Determination)
    # R² = 1 - (SS_res / SS_tot)
    ss_res = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    r2 = 1 - (ss_res / ss_tot)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }


def explain_metrics():
    """Explain evaluation metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print("""
    1. MSE (Mean Squared Error):
       - Average of squared differences between predicted and actual values
       - MSE = (1/n) * Σ(y_pred - y_actual)²
       - Penalizes larger errors more heavily

    2. RMSE (Root Mean Squared Error):
       - Square root of MSE
       - RMSE = √MSE
       - Same unit as the target variable (more interpretable)

    3. MAE (Mean Absolute Error):
       - Average of absolute differences
       - MAE = (1/n) * Σ|y_pred - y_actual|
       - Less sensitive to outliers than MSE

    4. R² Score (Coefficient of Determination):
       - Proportion of variance explained by the model
       - R² = 1 - (SS_res / SS_tot)
       - Range: (-∞, 1], where 1 is perfect prediction
       - R² = 0 means model is as good as predicting mean
       - R² < 0 means model is worse than predicting mean
    """)


# =============================================================================
# PART 4: SIMPLE VISUALIZATION EXAMPLE
# =============================================================================

def demonstrate_simple_linear_regression():
    """
    Demonstrate linear regression with a simple 1D example.
    """
    print("\n" + "=" * 60)
    print("SIMPLE LINEAR REGRESSION DEMONSTRATION")
    print("=" * 60)

    # Generate synthetic data: y = 3x + 5 + noise
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 3 * X.squeeze() + 5 + np.random.randn(100) * 0.5

    print(f"\nGenerated data: y = 3x + 5 + noise")
    print(f"True slope (m): 3")
    print(f"True intercept (b): 5")

    # Train our model from scratch
    model_scratch = LinearRegressionScratch(
        learning_rate=0.1,
        n_iterations=500,
        method='gradient_descent'
    )
    model_scratch.fit(X, y)

    params = model_scratch.get_params()
    print(f"\nLearned parameters (Gradient Descent):")
    print(f"  Slope (weight): {params['weights'][0]:.4f}")
    print(f"  Intercept (bias): {params['bias']:.4f}")

    # Train using normal equation
    model_normal = LinearRegressionScratch(method='normal_equation')
    model_normal.fit(X, y)

    params_normal = model_normal.get_params()
    print(f"\nLearned parameters (Normal Equation):")
    print(f"  Slope (weight): {params_normal['weights'][0]:.4f}")
    print(f"  Intercept (bias): {params_normal['bias']:.4f}")

    # Compare with sklearn
    sklearn_model = LinearRegression()
    sklearn_model.fit(X, y)

    print(f"\nLearned parameters (Sklearn):")
    print(f"  Slope (weight): {sklearn_model.coef_[0]:.4f}")
    print(f"  Intercept (bias): {sklearn_model.intercept_:.4f}")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Data and regression line
    X_line = np.linspace(0, 2, 100).reshape(-1, 1)

    axes[0].scatter(X, y, alpha=0.6, label='Data points')
    axes[0].plot(X_line, model_scratch.predict(X_line), 'r-',
                 linewidth=2, label=f'Scratch (GD): y = {params["weights"][0]:.2f}x + {params["bias"]:.2f}')
    axes[0].plot(X_line, sklearn_model.predict(X_line), 'g--',
                 linewidth=2, label=f'Sklearn: y = {sklearn_model.coef_[0]:.2f}x + {sklearn_model.intercept_:.2f}')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('y')
    axes[0].set_title('Linear Regression: Data and Fitted Lines')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Cost history (gradient descent)
    axes[1].plot(model_scratch.cost_history)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Cost (MSE)')
    axes[1].set_title('Gradient Descent: Cost Function Convergence')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/claude_ai_course/week2_classical_ml/day08_linear_regression/simple_linear_regression.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved: simple_linear_regression.png")


# =============================================================================
# PART 5: ASSIGNMENT - HOUSE PRICE PREDICTION
# =============================================================================

def house_price_prediction():
    """
    Assignment: Predict house prices using linear regression.
    Compare manual implementation vs sklearn.
    """
    print("\n" + "=" * 60)
    print("ASSIGNMENT: HOUSE PRICE PREDICTION")
    print("=" * 60)

    # Load California Housing dataset
    print("\nLoading California Housing dataset...")
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    feature_names = housing.feature_names

    print(f"\nDataset Information:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Feature names: {feature_names}")
    print(f"  Target: Median house value (in $100,000s)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Feature scaling (important for gradient descent)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================================================
    # METHOD 1: Our implementation from scratch (Gradient Descent)
    # ==========================================================================
    print("\n" + "-" * 50)
    print("METHOD 1: From Scratch (Gradient Descent)")
    print("-" * 50)

    model_gd = LinearRegressionScratch(
        learning_rate=0.01,
        n_iterations=1000,
        method='gradient_descent'
    )
    model_gd.fit(X_train_scaled, y_train)

    y_pred_gd = model_gd.predict(X_test_scaled)
    metrics_gd = compute_metrics(y_test, y_pred_gd)

    print("\nGradient Descent Results:")
    for metric, value in metrics_gd.items():
        print(f"  {metric}: {value:.6f}")

    # ==========================================================================
    # METHOD 2: Our implementation from scratch (Normal Equation)
    # ==========================================================================
    print("\n" + "-" * 50)
    print("METHOD 2: From Scratch (Normal Equation)")
    print("-" * 50)

    # Normal equation doesn't require scaling, but we use scaled for fair comparison
    model_ne = LinearRegressionScratch(method='normal_equation')
    model_ne.fit(X_train_scaled, y_train)

    y_pred_ne = model_ne.predict(X_test_scaled)
    metrics_ne = compute_metrics(y_test, y_pred_ne)

    print("\nNormal Equation Results:")
    for metric, value in metrics_ne.items():
        print(f"  {metric}: {value:.6f}")

    # ==========================================================================
    # METHOD 3: Scikit-learn LinearRegression
    # ==========================================================================
    print("\n" + "-" * 50)
    print("METHOD 3: Scikit-learn LinearRegression")
    print("-" * 50)

    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train_scaled, y_train)

    y_pred_sklearn = sklearn_model.predict(X_test_scaled)
    metrics_sklearn = compute_metrics(y_test, y_pred_sklearn)

    print("\nScikit-learn Results:")
    for metric, value in metrics_sklearn.items():
        print(f"  {metric}: {value:.6f}")

    # Also use sklearn's built-in metrics for verification
    print("\nScikit-learn's built-in metrics (verification):")
    print(f"  MSE: {mean_squared_error(y_test, y_pred_sklearn):.6f}")
    print(f"  R²: {r2_score(y_test, y_pred_sklearn):.6f}")

    # ==========================================================================
    # COMPARISON AND VISUALIZATION
    # ==========================================================================
    print("\n" + "-" * 50)
    print("COMPARISON SUMMARY")
    print("-" * 50)

    comparison_data = {
        'Method': ['Gradient Descent', 'Normal Equation', 'Sklearn'],
        'MSE': [metrics_gd['MSE'], metrics_ne['MSE'], metrics_sklearn['MSE']],
        'RMSE': [metrics_gd['RMSE'], metrics_ne['RMSE'], metrics_sklearn['RMSE']],
        'R²': [metrics_gd['R²'], metrics_ne['R²'], metrics_sklearn['R²']]
    }

    print(f"\n{'Method':<20} {'MSE':>12} {'RMSE':>12} {'R²':>12}")
    print("-" * 58)
    for i in range(3):
        print(f"{comparison_data['Method'][i]:<20} "
              f"{comparison_data['MSE'][i]:>12.6f} "
              f"{comparison_data['RMSE'][i]:>12.6f} "
              f"{comparison_data['R²'][i]:>12.6f}")

    # Feature importance (coefficients)
    print("\n" + "-" * 50)
    print("FEATURE IMPORTANCE (Sklearn Coefficients)")
    print("-" * 50)

    feature_importance = list(zip(feature_names, sklearn_model.coef_))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'Feature':<15} {'Coefficient':>12}")
    print("-" * 30)
    for feat, coef in feature_importance:
        print(f"{feat:<15} {coef:>12.4f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred_sklearn, alpha=0.3)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                     'r--', linewidth=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('Actual House Value ($100,000s)')
    axes[0, 0].set_ylabel('Predicted House Value ($100,000s)')
    axes[0, 0].set_title('Sklearn: Actual vs Predicted House Values')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Residual Distribution
    residuals = y_test - y_pred_sklearn
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Residual (Actual - Predicted)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Residual Distribution (Mean: {np.mean(residuals):.4f})')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Cost History (Gradient Descent)
    axes[1, 0].plot(model_gd.cost_history)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Cost (MSE)')
    axes[1, 0].set_title('Gradient Descent Convergence')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Feature Importance
    features = [f[0] for f in feature_importance]
    coefficients = [f[1] for f in feature_importance]
    colors = ['green' if c > 0 else 'red' for c in coefficients]

    axes[1, 1].barh(features, coefficients, color=colors, alpha=0.7)
    axes[1, 1].set_xlabel('Coefficient Value')
    axes[1, 1].set_title('Feature Importance (Coefficient Magnitude)')
    axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/claude_ai_course/week2_classical_ml/day08_linear_regression/house_price_prediction.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved: house_price_prediction.png")

    return {
        'gradient_descent': metrics_gd,
        'normal_equation': metrics_ne,
        'sklearn': metrics_sklearn
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DAY 8: LINEAR REGRESSION")
    print("=" * 60)

    # Part 1: Theory explanation
    explain_linear_regression()

    # Part 2: Explain metrics
    explain_metrics()

    # Part 3: Simple demonstration
    demonstrate_simple_linear_regression()

    # Part 4: House price prediction assignment
    results = house_price_prediction()

    print("\n" + "=" * 60)
    print("DAY 8 COMPLETE!")
    print("=" * 60)
    print("""
    Key Takeaways:
    1. Linear regression finds the best-fit line by minimizing MSE
    2. Gradient descent iteratively updates weights to minimize cost
    3. Normal equation provides closed-form solution (faster for small datasets)
    4. Feature scaling is crucial for gradient descent convergence
    5. R² score indicates how well the model explains variance in data
    6. All three methods (GD, Normal Eq, Sklearn) produce similar results
    """)
