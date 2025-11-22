"""
DAY 10: Decision Trees
======================
Topics Covered:
- Gini impurity and information gain (entropy)
- Tree construction and pruning
- Visualize decision boundaries
- Handle overfitting
- Assignment: Loan approval prediction with tree visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


# =============================================================================
# PART 1: THEORETICAL FOUNDATIONS
# =============================================================================

def explain_decision_trees():
    """
    Decision Tree Theory
    ====================
    """
    print("=" * 60)
    print("DECISION TREE THEORY")
    print("=" * 60)
    print("""
    DECISION TREES
    ==============
    A tree-like model that makes decisions based on feature values.
    Each internal node represents a "test" on a feature.
    Each branch represents the outcome of the test.
    Each leaf node represents a class label (classification) or value (regression).

    SPLITTING CRITERIA
    ==================

    1. GINI IMPURITY:
       Gini(t) = 1 - Σ p(i|t)²

       - Measures the probability of incorrect classification
       - Range: [0, 0.5] for binary classification
       - 0 = pure node (all same class)
       - Higher = more impure

       Example (binary):
       - Pure node [10, 0]: Gini = 1 - (1² + 0²) = 0
       - Mixed [5, 5]: Gini = 1 - (0.5² + 0.5²) = 0.5

    2. ENTROPY (Information Gain):
       Entropy(t) = -Σ p(i|t) * log₂(p(i|t))

       - Measures the amount of information/uncertainty
       - Range: [0, 1] for binary classification
       - 0 = pure node
       - Higher = more uncertain

       Information Gain = Entropy(parent) - Σ(weighted Entropy(children))

    TREE CONSTRUCTION (ID3/C4.5/CART Algorithm)
    ===========================================
    1. Start with all samples at root
    2. For each feature:
       - Calculate impurity reduction for each possible split
    3. Select feature with maximum reduction
    4. Create child nodes based on split
    5. Recursively repeat for each child
    6. Stop when:
       - Node is pure (all same class)
       - Max depth reached
       - Min samples threshold
       - No further gain

    PRUNING (Preventing Overfitting)
    ================================
    1. Pre-pruning (Early Stopping):
       - max_depth: Maximum tree depth
       - min_samples_split: Minimum samples to split
       - min_samples_leaf: Minimum samples in leaf
       - max_features: Features to consider for split

    2. Post-pruning (Cost-Complexity):
       - Grow full tree, then remove branches
       - Use ccp_alpha (cost complexity parameter)
       - Higher alpha = more pruning
    """)


# =============================================================================
# PART 2: DECISION TREE FROM SCRATCH (Simplified)
# =============================================================================

class DecisionTreeScratch:
    """
    Simplified Decision Tree implementation from scratch.
    Supports binary and multiclass classification.
    """

    def __init__(self, max_depth=10, min_samples_split=2, criterion='gini'):
        """
        Initialize Decision Tree.

        Parameters:
        -----------
        max_depth : int
            Maximum depth of the tree
        min_samples_split : int
            Minimum samples required to split a node
        criterion : str
            'gini' or 'entropy'
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None

    def _gini(self, y):
        """
        Calculate Gini impurity.
        Gini = 1 - Σ p(i)²
        """
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)

    def _entropy(self, y):
        """
        Calculate entropy.
        Entropy = -Σ p(i) * log₂(p(i))
        """
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        # Avoid log(0)
        proportions = proportions[proportions > 0]
        return -np.sum(proportions * np.log2(proportions))

    def _impurity(self, y):
        """Calculate impurity based on chosen criterion."""
        if self.criterion == 'gini':
            return self._gini(y)
        return self._entropy(y)

    def _information_gain(self, y, y_left, y_right):
        """
        Calculate information gain from a split.
        IG = Impurity(parent) - weighted_avg(Impurity(children))
        """
        parent_impurity = self._impurity(y)

        n = len(y)
        n_left, n_right = len(y_left), len(y_right)

        if n_left == 0 or n_right == 0:
            return 0

        # Weighted average of children impurities
        child_impurity = (n_left / n) * self._impurity(y_left) + \
                         (n_right / n) * self._impurity(y_right)

        return parent_impurity - child_impurity

    def _best_split(self, X, y):
        """
        Find the best split for a node.
        Returns: (best_feature, best_threshold, best_gain)
        """
        best_gain = -1
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                # Split data
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                y_left = y[left_mask]
                y_right = y[right_mask]

                # Calculate information gain
                gain = self._information_gain(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or
            n_classes == 1 or
            n_samples < self.min_samples_split):
            # Return leaf node with most common class
            leaf_value = np.bincount(y).argmax()
            return {'leaf': True, 'value': leaf_value, 'samples': n_samples}

        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)

        if best_gain == 0:
            # No improvement, return leaf
            leaf_value = np.bincount(y).argmax()
            return {'leaf': True, 'value': leaf_value, 'samples': n_samples}

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Build child trees
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree,
            'samples': n_samples,
            'impurity': self._impurity(y)
        }

    def fit(self, X, y):
        """
        Fit the decision tree to training data.
        """
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, y)
        return self

    def _predict_single(self, x, node):
        """
        Predict class for a single sample.
        """
        if node['leaf']:
            return node['value']

        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        return self._predict_single(x, node['right'])

    def predict(self, X):
        """
        Predict class labels for samples.
        """
        return np.array([self._predict_single(x, self.tree) for x in X])

    def print_tree(self, node=None, depth=0, feature_names=None):
        """
        Print tree structure.
        """
        if node is None:
            node = self.tree

        indent = "  " * depth

        if node['leaf']:
            print(f"{indent}Leaf: Class {node['value']} (samples: {node['samples']})")
        else:
            feat_name = feature_names[node['feature']] if feature_names else f"Feature {node['feature']}"
            print(f"{indent}{feat_name} <= {node['threshold']:.2f} (samples: {node['samples']}, "
                  f"impurity: {node['impurity']:.3f})")
            print(f"{indent}Left:")
            self.print_tree(node['left'], depth + 1, feature_names)
            print(f"{indent}Right:")
            self.print_tree(node['right'], depth + 1, feature_names)


# =============================================================================
# PART 3: GINI VS ENTROPY DEMONSTRATION
# =============================================================================

def demonstrate_splitting_criteria():
    """
    Demonstrate Gini impurity and Entropy calculations.
    """
    print("\n" + "=" * 60)
    print("SPLITTING CRITERIA DEMONSTRATION")
    print("=" * 60)

    # Different class distributions
    distributions = [
        [50, 50],   # Equal split
        [80, 20],   # Unequal
        [100, 0],   # Pure
        [60, 40],   # Slightly unequal
        [33, 33, 34],  # Three classes
    ]

    print("\n{:<25} {:>12} {:>12}".format("Distribution", "Gini", "Entropy"))
    print("-" * 50)

    for dist in distributions:
        total = sum(dist)
        props = [d / total for d in dist]

        # Gini
        gini = 1 - sum(p**2 for p in props)

        # Entropy
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in props)

        print(f"{str(dist):<25} {gini:>12.4f} {entropy:>12.4f}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # For binary classification
    p = np.linspace(0.01, 0.99, 100)

    # Gini
    gini = 1 - p**2 - (1-p)**2
    axes[0].plot(p, gini, 'b-', linewidth=2)
    axes[0].set_xlabel('Probability of Class 1')
    axes[0].set_ylabel('Gini Impurity')
    axes[0].set_title('Gini Impurity vs Class Distribution')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.5, color='r', linestyle='--', label='Maximum (0.5)')
    axes[0].legend()

    # Entropy
    entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
    axes[1].plot(p, entropy, 'g-', linewidth=2)
    axes[1].set_xlabel('Probability of Class 1')
    axes[1].set_ylabel('Entropy')
    axes[1].set_title('Entropy vs Class Distribution')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=1.0, color='r', linestyle='--', label='Maximum (1.0)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('/home/user/claude_ai_course/week2_classical_ml/day10_decision_trees/splitting_criteria.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved: splitting_criteria.png")


# =============================================================================
# PART 4: DECISION BOUNDARY VISUALIZATION
# =============================================================================

def visualize_decision_boundaries():
    """
    Visualize how decision trees create decision boundaries.
    """
    print("\n" + "-" * 50)
    print("Visualizing Decision Boundaries")
    print("-" * 50)

    # Generate synthetic 2D data
    np.random.seed(42)
    n_samples = 300

    # Create three clusters
    X1 = np.random.randn(n_samples // 3, 2) + np.array([0, 0])
    X2 = np.random.randn(n_samples // 3, 2) + np.array([3, 3])
    X3 = np.random.randn(n_samples // 3, 2) + np.array([3, 0])

    X = np.vstack([X1, X2, X3])
    y = np.array([0] * (n_samples // 3) + [1] * (n_samples // 3) + [2] * (n_samples // 3))

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Compare different max_depths
    depths = [1, 3, 5, None]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for idx, max_depth in enumerate(depths):
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        clf.fit(X, y)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
        axes[idx].scatter(X[y == 0, 0], X[y == 0, 1], c='blue', marker='o',
                          edgecolors='black', label='Class 0')
        axes[idx].scatter(X[y == 1, 0], X[y == 1, 1], c='green', marker='s',
                          edgecolors='black', label='Class 1')
        axes[idx].scatter(X[y == 2, 0], X[y == 2, 1], c='red', marker='^',
                          edgecolors='black', label='Class 2')

        train_acc = clf.score(X, y)
        depth_str = str(max_depth) if max_depth else 'None (unlimited)'
        axes[idx].set_title(f'max_depth={depth_str}\nTrain Accuracy: {train_acc:.3f}')
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')
        if idx == 0:
            axes[idx].legend()

    plt.suptitle('Decision Tree: Effect of max_depth on Decision Boundary', fontsize=14)
    plt.tight_layout()
    plt.savefig('/home/user/claude_ai_course/week2_classical_ml/day10_decision_trees/decision_boundaries.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved: decision_boundaries.png")


# =============================================================================
# PART 5: OVERFITTING DEMONSTRATION
# =============================================================================

def demonstrate_overfitting():
    """
    Demonstrate overfitting and how pruning helps.
    """
    print("\n" + "-" * 50)
    print("Demonstrating Overfitting and Pruning")
    print("-" * 50)

    # Generate noisy data
    np.random.seed(42)
    n_samples = 200

    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    # Add noise
    noise_idx = np.random.choice(n_samples, size=20, replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Compare different complexity levels
    depths = range(1, 16)
    train_scores = []
    test_scores = []
    cv_scores = []

    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)

        train_scores.append(clf.score(X_train, y_train))
        test_scores.append(clf.score(X_test, y_test))
        cv_scores.append(cross_val_score(clf, X_train, y_train, cv=5).mean())

    # Find optimal depth
    best_depth = depths[np.argmax(cv_scores)]
    print(f"\nOptimal max_depth (by CV): {best_depth}")
    print(f"Best CV score: {max(cv_scores):.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Learning curves
    axes[0].plot(depths, train_scores, 'b-o', label='Training Score', linewidth=2)
    axes[0].plot(depths, test_scores, 'r-s', label='Test Score', linewidth=2)
    axes[0].plot(depths, cv_scores, 'g-^', label='CV Score (5-fold)', linewidth=2)
    axes[0].axvline(x=best_depth, color='purple', linestyle='--',
                    label=f'Optimal depth = {best_depth}')
    axes[0].fill_between(depths, train_scores, test_scores, alpha=0.2,
                          where=np.array(train_scores) > np.array(test_scores),
                          color='red', label='Overfitting region')
    axes[0].set_xlabel('Max Depth')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Overfitting: Train vs Test Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.5, 1.05)

    # Plot 2: Cost complexity pruning path
    clf = DecisionTreeClassifier(random_state=42)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    impurities = path.impurities

    axes[1].plot(ccp_alphas, impurities, 'b-o', linewidth=2)
    axes[1].set_xlabel('Cost Complexity Parameter (alpha)')
    axes[1].set_ylabel('Total Impurity of Leaves')
    axes[1].set_title('Cost Complexity Pruning Path')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/claude_ai_course/week2_classical_ml/day10_decision_trees/overfitting.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved: overfitting.png")

    return best_depth


# =============================================================================
# PART 6: ASSIGNMENT - LOAN APPROVAL PREDICTION
# =============================================================================

def create_loan_dataset():
    """
    Create a synthetic loan approval dataset.
    """
    np.random.seed(42)
    n_samples = 1000

    # Features
    income = np.random.normal(50000, 20000, n_samples).clip(15000, 150000)
    credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
    loan_amount = np.random.normal(200000, 100000, n_samples).clip(50000, 500000)
    employment_years = np.random.exponential(5, n_samples).clip(0, 30)
    debt_ratio = np.random.beta(2, 5, n_samples)  # 0-1
    age = np.random.normal(40, 12, n_samples).clip(18, 75)
    num_credit_cards = np.random.poisson(3, n_samples).clip(0, 10)
    previous_defaults = np.random.binomial(1, 0.15, n_samples)  # 15% have defaulted

    # Create approval decision based on rules (with some noise)
    score = (
        0.3 * (income / 150000) +
        0.25 * (credit_score / 850) +
        0.15 * (1 - loan_amount / 500000) +
        0.1 * (employment_years / 30) +
        0.1 * (1 - debt_ratio) +
        0.05 * (1 - previous_defaults) +
        0.05 * (1 - num_credit_cards / 10)
    )

    # Add noise
    score += np.random.normal(0, 0.1, n_samples)

    # Approval threshold
    approved = (score > 0.45).astype(int)

    # Create feature matrix
    X = np.column_stack([
        income,
        credit_score,
        loan_amount,
        employment_years,
        debt_ratio,
        age,
        num_credit_cards,
        previous_defaults
    ])

    feature_names = [
        'Income', 'Credit_Score', 'Loan_Amount', 'Employment_Years',
        'Debt_Ratio', 'Age', 'Num_Credit_Cards', 'Previous_Defaults'
    ]

    return X, approved, feature_names


def loan_approval_prediction():
    """
    Assignment: Create decision tree for loan approval prediction.
    """
    print("\n" + "=" * 60)
    print("ASSIGNMENT: LOAN APPROVAL PREDICTION")
    print("=" * 60)

    # Create dataset
    print("\nCreating loan approval dataset...")
    X, y, feature_names = create_loan_dataset()

    print(f"\nDataset Information:")
    print(f"  Total samples: {len(y)}")
    print(f"  Approved: {sum(y)} ({100*sum(y)/len(y):.1f}%)")
    print(f"  Denied: {len(y)-sum(y)} ({100*(len(y)-sum(y))/len(y):.1f}%)")
    print(f"  Features: {feature_names}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================================================
    # METHOD 1: Our implementation from scratch
    # ==========================================================================
    print("\n" + "-" * 50)
    print("METHOD 1: Decision Tree from Scratch")
    print("-" * 50)

    model_scratch = DecisionTreeScratch(max_depth=5, criterion='gini')
    model_scratch.fit(X_train, y_train)

    y_pred_scratch = model_scratch.predict(X_test)
    acc_scratch = accuracy_score(y_test, y_pred_scratch)

    print(f"\nAccuracy (From Scratch): {acc_scratch:.4f}")

    print("\nTree Structure (From Scratch):")
    model_scratch.print_tree(feature_names=feature_names)

    # ==========================================================================
    # METHOD 2: Scikit-learn Decision Tree (with pruning)
    # ==========================================================================
    print("\n" + "-" * 50)
    print("METHOD 2: Scikit-learn Decision Tree")
    print("-" * 50)

    # Find optimal hyperparameters using cross-validation
    best_params = {'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5}

    # Test different depths
    depths = range(1, 15)
    cv_scores = []
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        cv_scores.append(scores.mean())

    best_depth = depths[np.argmax(cv_scores)]
    print(f"\nOptimal max_depth (CV): {best_depth}")

    # Train with optimal parameters
    sklearn_model = DecisionTreeClassifier(
        max_depth=best_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    sklearn_model.fit(X_train, y_train)

    y_pred_sklearn = sklearn_model.predict(X_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

    print(f"Test Accuracy (Sklearn): {acc_sklearn:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_sklearn, target_names=['Denied', 'Approved']))

    # Feature importance
    print("\nFeature Importance:")
    importance = sklearn_model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    for idx in sorted_idx:
        print(f"  {feature_names[idx]:<20}: {importance[idx]:.4f}")

    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    fig = plt.figure(figsize=(20, 16))

    # Plot 1: Decision Tree Visualization
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.axis('off')
    ax1.set_title('Decision Tree Structure (max_depth=3 for clarity)', fontsize=12)

    # Create a simpler tree for visualization
    simple_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    simple_tree.fit(X_train, y_train)

    # Plot 2: Full tree visualization (separate figure)
    fig_tree, ax_tree = plt.subplots(figsize=(25, 15))
    plot_tree(simple_tree, feature_names=feature_names, class_names=['Denied', 'Approved'],
              filled=True, rounded=True, fontsize=10, ax=ax_tree)
    ax_tree.set_title('Decision Tree for Loan Approval', fontsize=14)
    plt.tight_layout()
    fig_tree.savefig('/home/user/claude_ai_course/week2_classical_ml/day10_decision_trees/loan_tree_visualization.png',
                     dpi=150, bbox_inches='tight')
    plt.close(fig_tree)
    print("\nTree visualization saved: loan_tree_visualization.png")

    # Continue with main figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_sklearn)
    im = axes[0, 0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[0, 0].set_title('Confusion Matrix')
    plt.colorbar(im, ax=axes[0, 0])

    classes = ['Denied', 'Approved']
    tick_marks = np.arange(len(classes))
    axes[0, 0].set_xticks(tick_marks)
    axes[0, 0].set_yticks(tick_marks)
    axes[0, 0].set_xticklabels(classes)
    axes[0, 0].set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0, 0].text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black",
                           fontsize=14)
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_xlabel('Predicted')

    # Plot 2: Feature Importance
    sorted_idx = np.argsort(importance)
    axes[0, 1].barh([feature_names[i] for i in sorted_idx], importance[sorted_idx])
    axes[0, 1].set_xlabel('Feature Importance')
    axes[0, 1].set_title('Feature Importance')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Cross-validation scores vs depth
    axes[1, 0].plot(depths, cv_scores, 'b-o', linewidth=2)
    axes[1, 0].axvline(x=best_depth, color='r', linestyle='--', label=f'Optimal depth = {best_depth}')
    axes[1, 0].set_xlabel('Max Depth')
    axes[1, 0].set_ylabel('CV Score (5-fold)')
    axes[1, 0].set_title('Cross-Validation Score vs Max Depth')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Top 2 features decision boundary
    top_features = sorted_idx[-2:]  # Top 2 features
    X_2d = X_test[:, top_features]

    x_min, x_max = X_2d[:, 0].min() - 0.1 * (X_2d[:, 0].max() - X_2d[:, 0].min()), \
                   X_2d[:, 0].max() + 0.1 * (X_2d[:, 0].max() - X_2d[:, 0].min())
    y_min, y_max = X_2d[:, 1].min() - 0.1 * (X_2d[:, 1].max() - X_2d[:, 1].min()), \
                   X_2d[:, 1].max() + 0.1 * (X_2d[:, 1].max() - X_2d[:, 1].min())

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Train a simple tree on just these 2 features
    clf_2d = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf_2d.fit(X_train[:, top_features], y_train)
    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axes[1, 1].contourf(xx, yy, Z, alpha=0.4, cmap='RdYlGn')
    axes[1, 1].scatter(X_2d[y_test == 0, 0], X_2d[y_test == 0, 1],
                       c='red', marker='o', edgecolors='black', label='Denied')
    axes[1, 1].scatter(X_2d[y_test == 1, 0], X_2d[y_test == 1, 1],
                       c='green', marker='s', edgecolors='black', label='Approved')
    axes[1, 1].set_xlabel(feature_names[top_features[0]])
    axes[1, 1].set_ylabel(feature_names[top_features[1]])
    axes[1, 1].set_title('Decision Boundary (Top 2 Features)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('/home/user/claude_ai_course/week2_classical_ml/day10_decision_trees/loan_approval_analysis.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Analysis plot saved: loan_approval_analysis.png")

    # Print tree rules as text
    print("\n" + "-" * 50)
    print("DECISION TREE RULES (Text Format)")
    print("-" * 50)
    tree_rules = export_text(sklearn_model, feature_names=feature_names)
    print(tree_rules)

    return {
        'scratch_accuracy': acc_scratch,
        'sklearn_accuracy': acc_sklearn,
        'best_depth': best_depth,
        'feature_importance': dict(zip(feature_names, importance))
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DAY 10: DECISION TREES")
    print("=" * 60)

    # Part 1: Theory explanation
    explain_decision_trees()

    # Part 2: Splitting criteria demonstration
    demonstrate_splitting_criteria()

    # Part 3: Decision boundary visualization
    visualize_decision_boundaries()

    # Part 4: Overfitting demonstration
    best_depth = demonstrate_overfitting()

    # Part 5: Loan approval assignment
    results = loan_approval_prediction()

    print("\n" + "=" * 60)
    print("DAY 10 COMPLETE!")
    print("=" * 60)
    print(f"""
    Key Takeaways:
    1. Decision trees split data using Gini impurity or Information Gain
    2. Gini and Entropy both measure node impurity (uncertainty)
    3. Trees create axis-parallel decision boundaries
    4. Deeper trees can overfit - use pruning!
    5. Pre-pruning: max_depth, min_samples_split, min_samples_leaf
    6. Post-pruning: cost complexity pruning (ccp_alpha)
    7. Feature importance shows which features drive decisions

    Results Summary:
    - From Scratch Accuracy: {results['scratch_accuracy']:.4f}
    - Sklearn Accuracy: {results['sklearn_accuracy']:.4f}
    - Optimal max_depth: {results['best_depth']}
    """)
