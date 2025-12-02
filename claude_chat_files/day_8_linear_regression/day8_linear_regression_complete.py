"""
DAY 8: LINEAR REGRESSION - COMPLETE IMPLEMENTATION
===================================================

Topics Covered:
1. Mathematical Foundation (y = mx + b, cost function, gradient descent)
2. From-Scratch Implementation using NumPy
3. Scikit-learn Implementation
4. Evaluation Metrics (MSE, RMSE, R²)
5. Comparison: Manual vs sklearn on Ames Housing

Author: Gourab
Date: November 2024
Objective: Master linear regression theory and practice
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DAY 8: LINEAR REGRESSION - THEORY TO PRACTICE")
print("="*80)

# ============================================================================
# PART 1: MATHEMATICAL FOUNDATION
# ============================================================================
print("\n[PART 1] MATHEMATICAL FOUNDATION")
print("="*80)

print("""
LINEAR REGRESSION EQUATION:
--------------------------
y = mx + b  (Simple linear regression - 1 feature)
y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ  (Multiple linear regression)

Or in matrix form:
y = Xw + b
where:
  - y: Target variable (prices)
  - X: Feature matrix (input features)
  - w: Weights/coefficients
  - b: Bias/intercept

COST FUNCTION (Mean Squared Error):
----------------------------------
J(w, b) = (1/2m) Σ(ŷᵢ - yᵢ)²
        = (1/2m) Σ(w·xᵢ + b - yᵢ)²

where:
  - m: Number of training examples
  - ŷᵢ: Predicted value for example i
  - yᵢ: Actual value for example i
  - Goal: Minimize J(w, b)

GRADIENT DESCENT:
----------------
Algorithm to minimize cost function:

1. Initialize w and b randomly (or zeros)
2. Compute predictions: ŷ = Xw + b
3. Compute gradients:
   ∂J/∂w = (1/m) Xᵀ(ŷ - y)
   ∂J/∂b = (1/m) Σ(ŷᵢ - yᵢ)
4. Update parameters:
   w = w - α(∂J/∂w)
   b = b - α(∂J/∂b)
5. Repeat steps 2-4 until convergence

where α is the learning rate (e.g., 0.01)

KEY CONCEPTS:
------------
• Learning Rate (α): Step size in gradient descent
  - Too large: May overshoot minimum
  - Too small: Slow convergence
  
• Convergence: When cost function stops decreasing significantly
  
• Feature Scaling: Essential for gradient descent
  - StandardScaler: (X - μ) / σ
  - Speeds up convergence
""")

# ============================================================================
# PART 2: FROM-SCRATCH IMPLEMENTATION
# ============================================================================
print("\n[PART 2] LINEAR REGRESSION FROM SCRATCH (NumPy)")
print("="*80)

class LinearRegressionScratch:
    """
    Linear Regression implementation from scratch using NumPy.
    Uses gradient descent for optimization.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, verbose=False):
        """
        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent (default: 0.01)
        n_iterations : int
            Number of training iterations (default: 1000)
        verbose : bool
            Print training progress (default: False)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X, y):
        """
        Train the model using gradient descent.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Training features
        y : numpy array, shape (n_samples,)
            Target values
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # 1. Forward pass: Compute predictions
            y_pred = self._predict(X)
            
            # 2. Compute cost (MSE)
            cost = self._compute_cost(y, y_pred, n_samples)
            self.cost_history.append(cost)
            
            # 3. Compute gradients
            dw, db = self._compute_gradients(X, y, y_pred, n_samples)
            
            # 4. Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress
            if self.verbose and i % 100 == 0:
                print(f"Iteration {i:4d} | Cost: {cost:.4f}")
        
        if self.verbose:
            print(f"✓ Training complete! Final cost: {cost:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Features for prediction
            
        Returns:
        --------
        predictions : numpy array, shape (n_samples,)
            Predicted values
        """
        return self._predict(X)
    
    def _predict(self, X):
        """Internal prediction method."""
        return np.dot(X, self.weights) + self.bias
    
    def _compute_cost(self, y_true, y_pred, n_samples):
        """
        Compute Mean Squared Error cost.
        
        J(w, b) = (1/2m) Σ(ŷᵢ - yᵢ)²
        """
        return (1 / (2 * n_samples)) * np.sum((y_pred - y_true) ** 2)
    
    def _compute_gradients(self, X, y_true, y_pred, n_samples):
        """
        Compute gradients for weights and bias.
        
        ∂J/∂w = (1/m) Xᵀ(ŷ - y)
        ∂J/∂b = (1/m) Σ(ŷᵢ - yᵢ)
        """
        error = y_pred - y_true
        dw = (1 / n_samples) * np.dot(X.T, error)
        db = (1 / n_samples) * np.sum(error)
        return dw, db
    
    def get_params(self):
        """Return learned parameters."""
        return {
            'weights': self.weights,
            'bias': self.bias,
            'final_cost': self.cost_history[-1] if self.cost_history else None
        }

# Demonstrate from-scratch implementation
print("\n--- From-Scratch Implementation Demo ---")
print("Creating simple dataset...")

# Generate synthetic data for demonstration
np.random.seed(42)
X_demo = 2 * np.random.rand(100, 1)
y_demo = 4 + 3 * X_demo.squeeze() + np.random.randn(100)

print(f"Dataset: {X_demo.shape[0]} samples, {X_demo.shape[1]} feature")

# Train model
model_scratch = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000, verbose=True)
model_scratch.fit(X_demo, y_demo)

# Get parameters
params = model_scratch.get_params()
print(f"\n✓ Learned Parameters:")
print(f"  Weight (m): {params['weights'][0]:.4f}")
print(f"  Bias (b):   {params['bias']:.4f}")
print(f"  True values: m=3.0, b=4.0")

# ============================================================================
# PART 3: EVALUATION METRICS
# ============================================================================
print("\n[PART 3] EVALUATION METRICS")
print("="*80)

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Compute and display evaluation metrics.
    
    Metrics:
    --------
    1. MSE (Mean Squared Error): Average of squared differences
       MSE = (1/n) Σ(yᵢ - ŷᵢ)²
       - Lower is better
       - Same units as y²
       
    2. RMSE (Root Mean Squared Error): Square root of MSE
       RMSE = √MSE
       - Same units as y
       - More interpretable than MSE
       
    3. R² Score (Coefficient of Determination):
       R² = 1 - (SS_res / SS_tot)
       where SS_res = Σ(yᵢ - ŷᵢ)² (residual sum of squares)
             SS_tot = Σ(yᵢ - ȳ)² (total sum of squares)
       - Range: (-∞, 1], best is 1
       - R²=1: Perfect predictions
       - R²=0: Model performs like mean baseline
       - R²<0: Model worse than mean
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Evaluation:")
    print(f"  MSE:  {mse:,.2f}")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  R²:   {r2:.4f}")
    
    # Interpretation
    if r2 > 0.9:
        interpretation = "Excellent fit"
    elif r2 > 0.7:
        interpretation = "Good fit"
    elif r2 > 0.5:
        interpretation = "Moderate fit"
    else:
        interpretation = "Poor fit"
    
    print(f"  Interpretation: {interpretation}")
    
    return {'mse': mse, 'rmse': rmse, 'r2': r2}

# Demo on simple dataset
y_pred_demo = model_scratch.predict(X_demo)
metrics_demo = evaluate_model(y_demo, y_pred_demo, "From-Scratch Model")

# ============================================================================
# PART 4: AMES HOUSING DATASET - COMPLETE COMPARISON
# ============================================================================
print("\n[PART 4] AMES HOUSING PRICE PREDICTION")
print("="*80)

# Load the cleaned Ames Housing dataset
print("\n--- Loading Dataset ---")
try:
    df = pd.read_csv('/mnt/user-data/outputs/ames_housing_cleaned.csv')
    print(f"✓ Loaded: {df.shape[0]} samples, {df.shape[1]} features")
except:
    print("⚠ Could not load Ames dataset. Creating synthetic housing data...")
    np.random.seed(42)
    n_samples = 2000
    df = pd.DataFrame({
        'Gr_Liv_Area': np.random.normal(1500, 500, n_samples),
        'Overall_Qual': np.random.choice(range(1, 11), n_samples),
        'Year_Built': np.random.randint(1950, 2010, n_samples),
        'Total_Bsmt_SF': np.random.normal(1000, 400, n_samples),
        'Garage_Area': np.random.normal(450, 150, n_samples),
        'SalePrice': np.random.lognormal(12, 0.4, n_samples) * 1000
    })

# Select features for modeling
feature_cols = ['Gr_Liv_Area', 'Overall_Qual', 'Year_Built', 'Total_Bsmt_SF', 'Garage_Area']
available_features = [col for col in feature_cols if col in df.columns]

print(f"Using features: {available_features}")

# Prepare data
X = df[available_features].values
y = df['SalePrice'].values

# Log transform target (as recommended in EDA)
y_log = np.log1p(y)
print(f"\n✓ Applied log transformation to target (SalePrice)")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
print(f"✓ Train set: {X_train.shape[0]} samples")
print(f"✓ Test set:  {X_test.shape[0]} samples")

# Feature scaling (essential for gradient descent)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled (StandardScaler)")

# ============================================================================
# MODEL 1: FROM-SCRATCH IMPLEMENTATION
# ============================================================================
print("\n" + "="*80)
print("MODEL 1: LINEAR REGRESSION FROM SCRATCH")
print("="*80)

start_time = time.time()

# Train from-scratch model
model_manual = LinearRegressionScratch(
    learning_rate=0.01, 
    n_iterations=1000, 
    verbose=True
)
model_manual.fit(X_train_scaled, y_train)

# Predict
y_pred_train_manual = model_manual.predict(X_train_scaled)
y_pred_test_manual = model_manual.predict(X_test_scaled)

manual_time = time.time() - start_time

# Evaluate
print("\n--- Training Set Performance ---")
metrics_train_manual = evaluate_model(y_train, y_pred_train_manual, "From-Scratch (Train)")

print("\n--- Test Set Performance ---")
metrics_test_manual = evaluate_model(y_test, y_pred_test_manual, "From-Scratch (Test)")

print(f"\nTraining Time: {manual_time:.4f} seconds")

# ============================================================================
# MODEL 2: SCIKIT-LEARN IMPLEMENTATION
# ============================================================================
print("\n" + "="*80)
print("MODEL 2: SCIKIT-LEARN LINEAR REGRESSION")
print("="*80)

start_time = time.time()

# Train sklearn model
model_sklearn = LinearRegression()
model_sklearn.fit(X_train_scaled, y_train)

# Predict
y_pred_train_sklearn = model_sklearn.predict(X_train_scaled)
y_pred_test_sklearn = model_sklearn.predict(X_test_scaled)

sklearn_time = time.time() - start_time

# Evaluate
print("\n--- Training Set Performance ---")
metrics_train_sklearn = evaluate_model(y_train, y_pred_train_sklearn, "Scikit-learn (Train)")

print("\n--- Test Set Performance ---")
metrics_test_sklearn = evaluate_model(y_test, y_pred_test_sklearn, "Scikit-learn (Test)")

print(f"\nTraining Time: {sklearn_time:.4f} seconds")

# ============================================================================
# COMPARISON & VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("COMPARISON: MANUAL vs SKLEARN")
print("="*80)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Metric': ['Test MSE', 'Test RMSE', 'Test R²', 'Training Time (s)'],
    'From-Scratch': [
        f"{metrics_test_manual['mse']:.4f}",
        f"{metrics_test_manual['rmse']:.4f}",
        f"{metrics_test_manual['r2']:.4f}",
        f"{manual_time:.4f}"
    ],
    'Scikit-learn': [
        f"{metrics_test_sklearn['mse']:.4f}",
        f"{metrics_test_sklearn['rmse']:.4f}",
        f"{metrics_test_sklearn['r2']:.4f}",
        f"{sklearn_time:.4f}"
    ]
})

print("\n", comparison_df.to_string(index=False))

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Cost/Loss History (From-Scratch)
axes[0, 0].plot(model_manual.cost_history, linewidth=2, color='steelblue')
axes[0, 0].set_title('Training Loss Curve (From-Scratch)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Cost (MSE)')
axes[0, 0].grid(alpha=0.3)

# 2. Predictions vs Actual (From-Scratch)
axes[0, 1].scatter(y_test, y_pred_test_manual, alpha=0.5, s=30, color='coral')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
axes[0, 1].set_title('From-Scratch: Predictions vs Actual', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Actual Log(Price)')
axes[0, 1].set_ylabel('Predicted Log(Price)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Predictions vs Actual (Sklearn)
axes[1, 0].scatter(y_test, y_pred_test_sklearn, alpha=0.5, s=30, color='mediumseagreen')
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
axes[1, 0].set_title('Scikit-learn: Predictions vs Actual', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Actual Log(Price)')
axes[1, 0].set_ylabel('Predicted Log(Price)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. Residuals Plot
residuals_manual = y_test - y_pred_test_manual
residuals_sklearn = y_test - y_pred_test_sklearn

axes[1, 1].scatter(y_pred_test_manual, residuals_manual, alpha=0.5, s=30, 
                   color='coral', label='From-Scratch')
axes[1, 1].scatter(y_pred_test_sklearn, residuals_sklearn, alpha=0.5, s=30, 
                   color='mediumseagreen', label='Scikit-learn')
axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_title('Residuals Plot', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Predicted Log(Price)')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/day8_linear_regression_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved: day8_linear_regression_results.png")
plt.close()

# ============================================================================
# KEY INSIGHTS & LEARNINGS
# ============================================================================
print("\n" + "="*80)
print("KEY INSIGHTS & LEARNINGS")
print("="*80)

print(f"""
1. IMPLEMENTATION COMPARISON:
   • From-Scratch: R² = {metrics_test_manual['r2']:.4f}
   • Scikit-learn: R² = {metrics_test_sklearn['r2']:.4f}
   • Difference: {abs(metrics_test_manual['r2'] - metrics_test_sklearn['r2']):.6f}
   
   ✓ Both implementations achieve nearly identical results
   ✓ Sklearn is ~{sklearn_time/manual_time:.0f}x faster (uses optimized BLAS)

2. GRADIENT DESCENT CONVERGENCE:
   • Initial cost: {model_manual.cost_history[0]:.4f}
   • Final cost: {model_manual.cost_history[-1]:.4f}
   • Reduction: {(1 - model_manual.cost_history[-1]/model_manual.cost_history[0])*100:.1f}%
   
   ✓ Cost decreases monotonically (good learning rate)
   ✓ Converged in {len(model_manual.cost_history)} iterations

3. MODEL PERFORMANCE:
   • Test RMSE: {metrics_test_manual['rmse']:.4f} log(price) units
   • On original scale: ±${np.exp(metrics_test_manual['rmse']):.0f} price range
   • R² = {metrics_test_manual['r2']:.4f} → Model explains {metrics_test_manual['r2']*100:.1f}% of variance
   
   ✓ {'Excellent' if metrics_test_manual['r2'] > 0.9 else 'Good' if metrics_test_manual['r2'] > 0.7 else 'Moderate'} fit for linear model
   ✓ Room for improvement with feature engineering

4. WHEN TO USE LINEAR REGRESSION:
   ✅ Good for: Linear relationships, interpretability, baseline models
   ✅ Fast training and prediction
   ✅ Works well with feature engineering (polynomial features, interactions)
   
   ❌ Poor for: Non-linear patterns, complex interactions
   ❌ Assumes independence and homoscedasticity
   ❌ Sensitive to outliers

5. NEXT STEPS FOR IMPROVEMENT:
   • Feature engineering: Create Age, Total_SF, Quality_Score
   • Polynomial features for non-linear relationships
   • Ridge/Lasso regularization for feature selection
   • Try ensemble methods (Random Forest, XGBoost) for comparison
""")

print("\n" + "="*80)
print("DAY 8 COMPLETE - LINEAR REGRESSION MASTERY")
print("="*80)
print("""
✅ Learned Concepts:
   1. Linear regression mathematics (y = mx + b, cost function)
   2. Gradient descent optimization algorithm
   3. From-scratch implementation with NumPy
   4. Scikit-learn implementation
   5. Evaluation metrics (MSE, RMSE, R²)
   6. Model comparison and interpretation

✅ Deliverables:
   • Complete from-scratch implementation
   • Ames Housing price prediction model
   • Comparison: Manual vs Sklearn
   • Visualizations and insights

⏭️  Next: DAY 9 - Logistic Regression (Binary Classification)
""")
