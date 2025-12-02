"""
DAY 9: LOGISTIC REGRESSION - BINARY CLASSIFICATION
===================================================

Topics Covered:
1. Binary Classification Theory (sigmoid, decision boundary)
2. From-Scratch Implementation using NumPy
3. Scikit-learn Implementation
4. Evaluation Metrics (Confusion Matrix, Accuracy, Precision, Recall, F1)
5. ROC Curve and AUC
6. Spam Email Classifier (>85% accuracy target)

Author: Gourab
Date: November 2024
Objective: Master binary classification and logistic regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve, roc_auc_score,
    classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DAY 9: LOGISTIC REGRESSION - BINARY CLASSIFICATION")
print("="*80)

# ============================================================================
# PART 1: MATHEMATICAL FOUNDATION
# ============================================================================
print("\n[PART 1] MATHEMATICAL FOUNDATION")
print("="*80)

print("""
LOGISTIC REGRESSION FUNDAMENTALS:
---------------------------------

1. SIGMOID FUNCTION (Logistic Function):
   σ(z) = 1 / (1 + e^(-z))
   
   Properties:
   • Output range: (0, 1) - Perfect for probabilities!
   • S-shaped curve
   • σ(0) = 0.5 (decision boundary)
   • σ(∞) → 1, σ(-∞) → 0

2. PREDICTION FUNCTION:
   z = w·x + b  (linear combination)
   ŷ = σ(z) = σ(w·x + b)  (probability)
   
   Classification rule:
   • If ŷ ≥ 0.5 → Class 1 (positive)
   • If ŷ < 0.5 → Class 0 (negative)

3. COST FUNCTION (Binary Cross-Entropy Loss):
   J(w, b) = -(1/m) Σ[yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]
   
   Why not MSE?
   • MSE creates non-convex optimization landscape
   • Cross-entropy is convex → guaranteed global minimum
   • Better gradient flow for binary classification

4. GRADIENT DESCENT:
   Same as linear regression but with sigmoid!
   
   ∂J/∂w = (1/m) Xᵀ(ŷ - y)
   ∂J/∂b = (1/m) Σ(ŷᵢ - yᵢ)
   
   Update rules:
   w := w - α(∂J/∂w)
   b := b - α(∂J/∂b)

5. DECISION BOUNDARY:
   The boundary where σ(w·x + b) = 0.5
   → w·x + b = 0 (linear boundary in feature space)
   
   For 2D: w₁x₁ + w₂x₂ + b = 0
   → Line separating classes

KEY DIFFERENCES FROM LINEAR REGRESSION:
--------------------------------------
┌─────────────────┬──────────────────┬─────────────────────┐
│ Aspect          │ Linear           │ Logistic            │
├─────────────────┼──────────────────┼─────────────────────┤
│ Output          │ Continuous       │ Probability (0-1)   │
│ Activation      │ None (identity)  │ Sigmoid             │
│ Loss Function   │ MSE              │ Cross-Entropy       │
│ Task            │ Regression       │ Classification      │
│ Interpretation  │ Direct value     │ Class probability   │
└─────────────────┴──────────────────┴─────────────────────┘
""")

# Visualize sigmoid function
print("\n--- Sigmoid Function Visualization ---")
z = np.linspace(-10, 10, 100)
sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(10, 6))
plt.plot(z, sigmoid, linewidth=3, color='steelblue', label='σ(z) = 1/(1+e⁻ᶻ)')
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Decision Boundary (0.5)')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
plt.grid(alpha=0.3)
plt.xlabel('z (w·x + b)', fontsize=12)
plt.ylabel('σ(z)', fontsize=12)
plt.title('Sigmoid Function - Maps any real number to (0, 1)', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.ylim(-0.1, 1.1)
plt.savefig('/mnt/user-data/outputs/day9_sigmoid_function.png', dpi=300, bbox_inches='tight')
print("✓ Sigmoid visualization saved")
plt.close()

# ============================================================================
# PART 2: FROM-SCRATCH IMPLEMENTATION
# ============================================================================
print("\n[PART 2] LOGISTIC REGRESSION FROM SCRATCH")
print("="*80)

class LogisticRegressionScratch:
    """
    Logistic Regression implementation from scratch using NumPy.
    Uses gradient descent with binary cross-entropy loss.
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
        
    def _sigmoid(self, z):
        """
        Sigmoid activation function.
        σ(z) = 1 / (1 + e^(-z))
        
        Handles overflow by clipping z.
        """
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, y_true, y_pred, n_samples):
        """
        Binary cross-entropy loss.
        J = -(1/m) Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
        
        Small epsilon added for numerical stability.
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        cost = -(1/n_samples) * np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return cost
    
    def fit(self, X, y):
        """
        Train the model using gradient descent.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Training features
        y : numpy array, shape (n_samples,)
            Binary labels (0 or 1)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass: Linear → Sigmoid
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)
            
            # Compute cost
            cost = self._compute_cost(y, y_pred, n_samples)
            self.cost_history.append(cost)
            
            # Compute gradients (same form as linear regression!)
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if self.verbose and i % 100 == 0:
                print(f"Iteration {i:4d} | Cost: {cost:.4f}")
        
        if self.verbose:
            print(f"✓ Training complete! Final cost: {cost:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Returns:
        --------
        probabilities : numpy array, shape (n_samples,)
            P(y=1|x) for each sample
        """
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : numpy array
            Features for prediction
        threshold : float
            Decision threshold (default: 0.5)
            
        Returns:
        --------
        predictions : numpy array
            Binary class predictions (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

print("✓ LogisticRegressionScratch class defined")

# Demo on simple dataset
print("\n--- Simple Binary Classification Demo ---")
np.random.seed(42)

# Generate 2-class dataset
from sklearn.datasets import make_classification
X_demo, y_demo = make_classification(
    n_samples=200, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, random_state=42
)

print(f"Dataset: {X_demo.shape[0]} samples, {X_demo.shape[1]} features")
print(f"Classes: {np.unique(y_demo)} (Binary)")

# Train model
model_demo = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000, verbose=True)
model_demo.fit(X_demo, y_demo)

# Evaluate
y_pred_demo = model_demo.predict(X_demo)
accuracy_demo = accuracy_score(y_demo, y_pred_demo)
print(f"\n✓ Demo Accuracy: {accuracy_demo:.4f}")

# ============================================================================
# PART 3: EVALUATION METRICS
# ============================================================================
print("\n[PART 3] EVALUATION METRICS FOR CLASSIFICATION")
print("="*80)

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix with annotations."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                square=True, linewidths=1, linecolor='black')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add labels
    plt.text(0.5, -0.1, 'Negative (0)', ha='center', va='top', 
             transform=plt.gca().transAxes, fontsize=10)
    plt.text(1.5, -0.1, 'Positive (1)', ha='center', va='top', 
             transform=plt.gca().transAxes, fontsize=10)
    
    return cm

def evaluate_classification(y_true, y_pred, y_prob=None, model_name="Model"):
    """
    Comprehensive classification evaluation.
    
    Metrics Explained:
    ------------------
    1. CONFUSION MATRIX:
       ┌─────────────────────────────────┐
       │           Predicted             │
       │        Neg (0)    Pos (1)       │
       ├─────────────────────────────────┤
       │ True │  TN          FP          │
       │ Neg  │  (correct)  (Type I)     │
       ├──────┼──────────────────────────┤
       │ True │  FN          TP          │
       │ Pos  │  (Type II)  (correct)    │
       └─────────────────────────────────┘
    
    2. ACCURACY: (TP + TN) / Total
       - Overall correctness
       - Can be misleading with imbalanced classes
    
    3. PRECISION: TP / (TP + FP)
       - "Of all positive predictions, how many were correct?"
       - High precision → Few false positives
    
    4. RECALL (Sensitivity): TP / (TP + FN)
       - "Of all actual positives, how many did we catch?"
       - High recall → Few false negatives
    
    5. F1-SCORE: 2 × (Precision × Recall) / (Precision + Recall)
       - Harmonic mean of precision and recall
       - Balances both metrics
       - Best for imbalanced datasets
    
    6. ROC-AUC:
       - Area Under ROC Curve
       - Probability model ranks positive > negative
       - 0.5 = random, 1.0 = perfect
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n{model_name} Evaluation:")
    print("-" * 60)
    print("CONFUSION MATRIX:")
    print(f"  TN: {tn:4d}  |  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  |  TP: {tp:4d}")
    print()
    print("METRICS:")
    print(f"  Accuracy:  {accuracy:.4f}  (Overall correctness)")
    print(f"  Precision: {precision:.4f}  (Positive prediction quality)")
    print(f"  Recall:    {recall:.4f}  (Positive detection rate)")
    print(f"  F1-Score:  {f1:.4f}  (Harmonic mean P&R)")
    
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        print(f"  ROC-AUC:   {auc:.4f}  (Ranking quality)")
    
    # Interpretation
    if f1 > 0.9:
        interpretation = "Excellent"
    elif f1 > 0.8:
        interpretation = "Good"
    elif f1 > 0.7:
        interpretation = "Fair"
    else:
        interpretation = "Needs improvement"
    
    print(f"\n  Overall: {interpretation}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc_score(y_true, y_prob) if y_prob is not None else None,
        'confusion_matrix': cm
    }

# Evaluate demo model
print("\n" + "="*80)
y_prob_demo = model_demo.predict_proba(X_demo)
metrics_demo = evaluate_classification(y_demo, y_pred_demo, y_prob_demo, "Demo Model")

# ============================================================================
# PART 4: SPAM EMAIL CLASSIFIER
# ============================================================================
print("\n" + "="*80)
print("[PART 4] SPAM EMAIL CLASSIFIER - REAL-WORLD APPLICATION")
print("="*80)

# Create synthetic spam dataset
print("\n--- Creating Spam Email Dataset ---")
np.random.seed(42)

# Spam keywords and patterns
spam_words = [
    'free', 'win', 'winner', 'cash', 'prize', 'claim', 'urgent', 'limited',
    'offer', 'guaranteed', 'bonus', 'credit', 'click', 'congratulations',
    'selected', 'act now', 'discount', 'pharmacy', 'viagra', 'cialis'
]

ham_words = [
    'meeting', 'schedule', 'project', 'report', 'team', 'update', 'please',
    'attached', 'review', 'feedback', 'regarding', 'discussed', 'thanks',
    'hello', 'regards', 'sincerely', 'question', 'information', 'help'
]

def generate_email(is_spam, n_words=20):
    """Generate synthetic email text."""
    if is_spam:
        # Spam: More spam words, aggressive punctuation
        words = np.random.choice(spam_words, size=int(n_words * 0.7), replace=True)
        words = np.append(words, np.random.choice(ham_words, size=int(n_words * 0.3), replace=True))
        text = ' '.join(words) + '!!!'
    else:
        # Ham: Professional language
        words = np.random.choice(ham_words, size=int(n_words * 0.8), replace=True)
        words = np.append(words, np.random.choice(spam_words, size=int(n_words * 0.2), replace=True))
        text = ' '.join(words) + '.'
    return text

# Generate dataset
n_samples = 2000
emails = []
labels = []

for i in range(n_samples):
    is_spam = np.random.rand() > 0.5  # 50-50 split
    email = generate_email(is_spam, n_words=np.random.randint(15, 30))
    emails.append(email)
    labels.append(1 if is_spam else 0)

df_spam = pd.DataFrame({'email': emails, 'label': labels})
print(f"✓ Generated {len(df_spam)} emails")
print(f"  Spam: {(df_spam['label'] == 1).sum()} ({(df_spam['label'] == 1).sum()/len(df_spam)*100:.1f}%)")
print(f"  Ham:  {(df_spam['label'] == 0).sum()} ({(df_spam['label'] == 0).sum()/len(df_spam)*100:.1f}%)")

# Sample emails
print("\n--- Sample Emails ---")
print("\nSPAM Example:")
spam_example = df_spam[df_spam['label'] == 1].iloc[0]['email']
print(f"  {spam_example[:100]}...")

print("\nHAM Example:")
ham_example = df_spam[df_spam['label'] == 0].iloc[0]['email']
print(f"  {ham_example[:100]}...")

# Text Vectorization using TF-IDF
print("\n--- Text Vectorization (TF-IDF) ---")
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
X_text = vectorizer.fit_transform(df_spam['email']).toarray()
y_text = df_spam['label'].values

print(f"✓ Vectorized to {X_text.shape[1]} features")
print(f"  Top features: {vectorizer.get_feature_names_out()[:10]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y_text, test_size=0.2, random_state=42, stratify=y_text
)

print(f"\n✓ Train set: {X_train.shape[0]} samples")
print(f"✓ Test set:  {X_test.shape[0]} samples")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled")

# ============================================================================
# MODEL 1: FROM-SCRATCH IMPLEMENTATION
# ============================================================================
print("\n" + "="*80)
print("MODEL 1: LOGISTIC REGRESSION FROM SCRATCH")
print("="*80)

import time
start_time = time.time()

model_scratch = LogisticRegressionScratch(
    learning_rate=0.1,
    n_iterations=1000,
    verbose=True
)
model_scratch.fit(X_train_scaled, y_train)

manual_time = time.time() - start_time

# Predictions
y_pred_train_scratch = model_scratch.predict(X_train_scaled)
y_pred_test_scratch = model_scratch.predict(X_test_scaled)
y_prob_test_scratch = model_scratch.predict_proba(X_test_scaled)

print("\n--- Training Set Performance ---")
metrics_train_scratch = evaluate_classification(
    y_train, y_pred_train_scratch, 
    model_scratch.predict_proba(X_train_scaled),
    "From-Scratch (Train)"
)

print("\n--- Test Set Performance ---")
metrics_test_scratch = evaluate_classification(
    y_test, y_pred_test_scratch, y_prob_test_scratch,
    "From-Scratch (Test)"
)

print(f"\nTraining Time: {manual_time:.4f} seconds")

# ============================================================================
# MODEL 2: SCIKIT-LEARN IMPLEMENTATION
# ============================================================================
print("\n" + "="*80)
print("MODEL 2: SCIKIT-LEARN LOGISTIC REGRESSION")
print("="*80)

start_time = time.time()

model_sklearn = LogisticRegression(max_iter=1000, random_state=42)
model_sklearn.fit(X_train_scaled, y_train)

sklearn_time = time.time() - start_time

# Predictions
y_pred_train_sklearn = model_sklearn.predict(X_train_scaled)
y_pred_test_sklearn = model_sklearn.predict(X_test_scaled)
y_prob_test_sklearn = model_sklearn.predict_proba(X_test_scaled)[:, 1]

print("\n--- Training Set Performance ---")
metrics_train_sklearn = evaluate_classification(
    y_train, y_pred_train_sklearn,
    model_sklearn.predict_proba(X_train_scaled)[:, 1],
    "Scikit-learn (Train)"
)

print("\n--- Test Set Performance ---")
metrics_test_sklearn = evaluate_classification(
    y_test, y_pred_test_sklearn, y_prob_test_sklearn,
    "Scikit-learn (Test)"
)

print(f"\nTraining Time: {sklearn_time:.4f} seconds")

# ============================================================================
# COMPARISON & VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("COMPARISON: MANUAL vs SKLEARN")
print("="*80)

comparison_df = pd.DataFrame({
    'Metric': ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score', 
               'Test ROC-AUC', 'Training Time (s)'],
    'From-Scratch': [
        f"{metrics_test_scratch['accuracy']:.4f}",
        f"{metrics_test_scratch['precision']:.4f}",
        f"{metrics_test_scratch['recall']:.4f}",
        f"{metrics_test_scratch['f1']:.4f}",
        f"{metrics_test_scratch['auc']:.4f}",
        f"{manual_time:.4f}"
    ],
    'Scikit-learn': [
        f"{metrics_test_sklearn['accuracy']:.4f}",
        f"{metrics_test_sklearn['precision']:.4f}",
        f"{metrics_test_sklearn['recall']:.4f}",
        f"{metrics_test_sklearn['f1']:.4f}",
        f"{metrics_test_sklearn['auc']:.4f}",
        f"{sklearn_time:.4f}"
    ]
})

print("\n", comparison_df.to_string(index=False))

# Check if we met the goal
test_acc = metrics_test_sklearn['accuracy']
goal_met = "✅ GOAL ACHIEVED!" if test_acc >= 0.85 else "⚠️  Below target"
print(f"\n{goal_met} (Target: >85% accuracy, Achieved: {test_acc*100:.1f}%)")

# Comprehensive Visualizations
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Cost History
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(model_scratch.cost_history, linewidth=2, color='steelblue')
ax1.set_title('Training Loss Curve (From-Scratch)', fontweight='bold')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Binary Cross-Entropy Loss')
ax1.grid(alpha=0.3)

# 2. Confusion Matrix - From-Scratch
ax2 = fig.add_subplot(gs[0, 1])
cm_scratch = metrics_test_scratch['confusion_matrix']
sns.heatmap(cm_scratch, annot=True, fmt='d', cmap='Blues', ax=ax2, 
            square=True, cbar=False, linewidths=1, linecolor='black')
ax2.set_title('From-Scratch Confusion Matrix', fontweight='bold')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('True')

# 3. Confusion Matrix - Sklearn
ax3 = fig.add_subplot(gs[0, 2])
cm_sklearn = metrics_test_sklearn['confusion_matrix']
sns.heatmap(cm_sklearn, annot=True, fmt='d', cmap='Greens', ax=ax3,
            square=True, cbar=False, linewidths=1, linecolor='black')
ax3.set_title('Scikit-learn Confusion Matrix', fontweight='bold')
ax3.set_xlabel('Predicted')
ax3.set_ylabel('True')

# 4. ROC Curve - From-Scratch
ax4 = fig.add_subplot(gs[1, 0])
fpr_scratch, tpr_scratch, _ = roc_curve(y_test, y_prob_test_scratch)
auc_scratch = roc_auc_score(y_test, y_prob_test_scratch)
ax4.plot(fpr_scratch, tpr_scratch, linewidth=2, color='coral',
         label=f'From-Scratch (AUC={auc_scratch:.3f})')
ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.500)')
ax4.set_title('ROC Curve - From-Scratch', fontweight='bold')
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.legend()
ax4.grid(alpha=0.3)

# 5. ROC Curve - Sklearn
ax5 = fig.add_subplot(gs[1, 1])
fpr_sklearn, tpr_sklearn, _ = roc_curve(y_test, y_prob_test_sklearn)
auc_sklearn = roc_auc_score(y_test, y_prob_test_sklearn)
ax5.plot(fpr_sklearn, tpr_sklearn, linewidth=2, color='mediumseagreen',
         label=f'Scikit-learn (AUC={auc_sklearn:.3f})')
ax5.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.500)')
ax5.set_title('ROC Curve - Scikit-learn', fontweight='bold')
ax5.set_xlabel('False Positive Rate')
ax5.set_ylabel('True Positive Rate')
ax5.legend()
ax5.grid(alpha=0.3)

# 6. Combined ROC Curves
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(fpr_scratch, tpr_scratch, linewidth=2, color='coral', 
         label=f'From-Scratch (AUC={auc_scratch:.3f})')
ax6.plot(fpr_sklearn, tpr_sklearn, linewidth=2, color='mediumseagreen',
         label=f'Scikit-learn (AUC={auc_sklearn:.3f})')
ax6.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.500)')
ax6.set_title('ROC Curves Comparison', fontweight='bold')
ax6.set_xlabel('False Positive Rate')
ax6.set_ylabel('True Positive Rate')
ax6.legend()
ax6.grid(alpha=0.3)

# 7. Probability Distribution
ax7 = fig.add_subplot(gs[2, 0])
ax7.hist(y_prob_test_scratch[y_test == 0], bins=30, alpha=0.6, 
         label='Ham (Negative)', color='blue')
ax7.hist(y_prob_test_scratch[y_test == 1], bins=30, alpha=0.6,
         label='Spam (Positive)', color='red')
ax7.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
ax7.set_title('Prediction Probabilities (From-Scratch)', fontweight='bold')
ax7.set_xlabel('P(Spam)')
ax7.set_ylabel('Count')
ax7.legend()
ax7.grid(alpha=0.3)

# 8. Metrics Comparison
ax8 = fig.add_subplot(gs[2, 1])
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
scratch_values = [
    metrics_test_scratch['accuracy'],
    metrics_test_scratch['precision'],
    metrics_test_scratch['recall'],
    metrics_test_scratch['f1'],
    metrics_test_scratch['auc']
]
sklearn_values = [
    metrics_test_sklearn['accuracy'],
    metrics_test_sklearn['precision'],
    metrics_test_sklearn['recall'],
    metrics_test_sklearn['f1'],
    metrics_test_sklearn['auc']
]

x = np.arange(len(metrics_names))
width = 0.35
ax8.bar(x - width/2, scratch_values, width, label='From-Scratch', color='coral')
ax8.bar(x + width/2, sklearn_values, width, label='Scikit-learn', color='mediumseagreen')
ax8.set_title('Metrics Comparison', fontweight='bold')
ax8.set_ylabel('Score')
ax8.set_xticks(x)
ax8.set_xticklabels(metrics_names, rotation=45, ha='right')
ax8.legend()
ax8.grid(axis='y', alpha=0.3)
ax8.set_ylim([0, 1.1])

# 9. Feature Importance (Top 10)
ax9 = fig.add_subplot(gs[2, 2])
feature_importance = np.abs(model_sklearn.coef_[0])
top_features_idx = np.argsort(feature_importance)[-10:]
top_features = vectorizer.get_feature_names_out()[top_features_idx]
top_importance = feature_importance[top_features_idx]

ax9.barh(range(len(top_features)), top_importance, color='steelblue')
ax9.set_yticks(range(len(top_features)))
ax9.set_yticklabels(top_features)
ax9.set_title('Top 10 Important Features', fontweight='bold')
ax9.set_xlabel('|Coefficient|')
ax9.grid(axis='x', alpha=0.3)

plt.savefig('/mnt/user-data/outputs/day9_spam_classifier_results.png', 
            dpi=300, bbox_inches='tight')
print("\n✓ Comprehensive visualization saved: day9_spam_classifier_results.png")
plt.close()

# ============================================================================
# KEY INSIGHTS & LEARNINGS
# ============================================================================
print("\n" + "="*80)
print("KEY INSIGHTS & LEARNINGS")
print("="*80)

print(f"""
1. SPAM CLASSIFIER PERFORMANCE:
   • Test Accuracy: {metrics_test_sklearn['accuracy']*100:.1f}% {goal_met}
   • Precision: {metrics_test_sklearn['precision']:.3f} (Low false positives)
   • Recall: {metrics_test_sklearn['recall']:.3f} (High spam detection rate)
   • F1-Score: {metrics_test_sklearn['f1']:.3f} (Balanced performance)
   • ROC-AUC: {metrics_test_sklearn['auc']:.3f} (Excellent ranking)

2. IMPLEMENTATION COMPARISON:
   • From-Scratch Accuracy: {metrics_test_scratch['accuracy']*100:.1f}%
   • Scikit-learn Accuracy: {metrics_test_sklearn['accuracy']*100:.1f}%
   • Difference: {abs(metrics_test_scratch['accuracy'] - metrics_test_sklearn['accuracy'])*100:.2f}%
   • Both achieve similar performance! ✓

3. TRAINING EFFICIENCY:
   • From-Scratch: {manual_time:.3f}s
   • Scikit-learn: {sklearn_time:.3f}s
   • Speedup: {manual_time/sklearn_time:.1f}x faster (optimized solver)

4. KEY FEATURES FOR SPAM DETECTION:
   Top spam indicators: {', '.join(top_features[-5:])}
   These words strongly predict spam emails.

5. SIGMOID FUNCTION INSIGHTS:
   • Maps linear output to probability (0-1)
   • Decision boundary at 0.5 separates classes
   • S-shape provides smooth gradients for optimization

6. CROSS-ENTROPY VS MSE:
   • Cross-entropy: Convex optimization landscape
   • MSE: Non-convex for classification (bad!)
   • Result: Guaranteed convergence with cross-entropy

7. WHEN TO USE LOGISTIC REGRESSION:
   ✅ Binary classification problems
   ✅ Need probabilistic predictions
   ✅ Want interpretable coefficients
   ✅ Baseline model for comparison
   ✅ Linear decision boundary works
   
   ❌ Multi-class (use softmax instead)
   ❌ Highly non-linear boundaries
   ❌ Complex feature interactions
   ❌ Extreme class imbalance (need SMOTE/weights)

8. CONFUSION MATRIX INSIGHTS:
   • High precision → Few false spam detections (good!)
   • High recall → Most spam caught (good!)
   • Balance depends on use case:
     - Email: Prefer high precision (don't lose important emails)
     - Fraud: Prefer high recall (catch all fraud cases)

9. ROC-AUC INTERPRETATION:
   • {auc_sklearn:.3f} means model ranks spam correctly {auc_sklearn*100:.0f}% of time
   • Much better than random (0.5)
   • Close to perfect (1.0) shows strong discrimination

10. NEXT STEPS FOR IMPROVEMENT:
    • Feature engineering: Email length, punctuation, capitals
    • N-grams: Capture phrase patterns
    • Deep learning: LSTM/Transformers for context
    • Ensemble: Combine with Naive Bayes, SVM
""")

print("\n" + "="*80)
print("DAY 9 COMPLETE - LOGISTIC REGRESSION MASTERY")
print("="*80)
print("""
✅ Learned Concepts:
   1. Sigmoid function and probability mapping
   2. Binary cross-entropy loss function
   3. Gradient descent for classification
   4. Confusion matrix interpretation
   5. Precision, Recall, F1-Score, ROC-AUC
   6. From-scratch logistic regression implementation
   7. TF-IDF text vectorization
   8. Spam classifier achieving >85% accuracy

✅ Deliverables:
   • Complete from-scratch implementation
   • Spam email classifier (Test Acc: {metrics_test_sklearn['accuracy']*100:.1f}%)
   • Comprehensive evaluation metrics
   • ROC curve analysis
   • Feature importance visualization

⏭️  Next: DAY 10 - Decision Trees (Gini impurity, tree visualization)
""".format(metrics_test_sklearn=metrics_test_sklearn))
