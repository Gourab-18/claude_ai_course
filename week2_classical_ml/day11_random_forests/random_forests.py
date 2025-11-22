"""
DAY 11: Random Forests & Ensemble Methods
==========================================
Topics Covered:
- Bagging and boosting concepts
- Random Forest implementation
- Feature importance analysis
- Compare with single decision tree
- Assignment: Improve Day 10 loan approval model using Random Forest
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, BaggingClassifier
)
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


# =============================================================================
# PART 1: THEORETICAL FOUNDATIONS
# =============================================================================

def explain_ensemble_methods():
    """
    Ensemble Methods Theory
    =======================
    """
    print("=" * 60)
    print("ENSEMBLE METHODS THEORY")
    print("=" * 60)
    print("""
    ENSEMBLE LEARNING
    =================
    Combining multiple models to produce better predictive performance
    than any single model alone.

    "Wisdom of the crowd" - many weak learners can form a strong learner

    1. BAGGING (Bootstrap Aggregating)
    ==================================
    - Train multiple models on different random samples (with replacement)
    - Aggregate predictions: voting (classification) or averaging (regression)
    - Reduces variance (overfitting)
    - Models trained in parallel (independent)

    Process:
    1. Create N bootstrap samples from training data
    2. Train a model on each bootstrap sample
    3. Aggregate predictions (majority vote or average)

    Key insight: Each model sees ~63.2% of data (1 - 1/e)

    2. RANDOM FOREST
    ================
    - Bagging + Random feature selection
    - Each tree trained on:
      * Bootstrap sample of data
      * Random subset of features at each split
    - Decorrelates trees, further reducing variance

    Hyperparameters:
    - n_estimators: Number of trees (more = better, but diminishing returns)
    - max_features: Features considered at each split
      * sqrt(n_features) for classification
      * n_features/3 for regression
    - max_depth, min_samples_split, etc.

    3. BOOSTING
    ===========
    - Train models sequentially, each correcting previous errors
    - Reduces bias (underfitting)
    - Models trained sequentially (dependent)

    Types:
    a) AdaBoost (Adaptive Boosting):
       - Increase weights of misclassified samples
       - Each model focuses on hard examples

    b) Gradient Boosting:
       - Fit new model to residuals (errors) of previous model
       - Uses gradient descent to minimize loss

    c) XGBoost, LightGBM, CatBoost:
       - Optimized implementations of gradient boosting
       - Regularization to prevent overfitting

    COMPARISON
    ==========
    | Aspect      | Bagging        | Boosting        |
    |-------------|----------------|-----------------|
    | Variance    | Reduces        | May reduce      |
    | Bias        | Same as base   | Reduces         |
    | Training    | Parallel       | Sequential      |
    | Overfitting | Less prone     | Can overfit     |
    | Speed       | Faster         | Slower          |
    """)


# =============================================================================
# PART 2: RANDOM FOREST FROM SCRATCH (Simplified)
# =============================================================================

class RandomForestScratch:
    """
    Simplified Random Forest implementation from scratch.
    """

    def __init__(self, n_estimators=10, max_depth=10, max_features='sqrt',
                 min_samples_split=2, bootstrap=True):
        """
        Initialize Random Forest.

        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int
            Maximum depth of each tree
        max_features : str or int
            Number of features to consider: 'sqrt', 'log2', or int
        min_samples_split : int
            Minimum samples to split a node
        bootstrap : bool
            Whether to use bootstrap samples
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.trees = []
        self.feature_indices = []

    def _get_max_features(self, n_features):
        """Determine number of features to consider at each split."""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        return n_features

    def _bootstrap_sample(self, X, y):
        """Create a bootstrap sample."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def _gini(self, y):
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)

    def _best_split(self, X, y, feature_indices):
        """Find best split considering only selected features."""
        best_gain = -1
        best_feature = None
        best_threshold = None

        parent_gini = self._gini(y)

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]

                # Information gain
                n = len(y)
                child_gini = (len(y_left)/n * self._gini(y_left) +
                              len(y_right)/n * self._gini(y_right))
                gain = parent_gini - child_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0, max_features_per_split=None):
        """Recursively build a decision tree with random feature selection."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or
            n_classes == 1 or
            n_samples < self.min_samples_split):
            return {'leaf': True, 'value': np.bincount(y).argmax()}

        # Random feature selection for this split
        feature_indices = np.random.choice(
            n_features, size=max_features_per_split, replace=False
        )

        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(X, y, feature_indices)

        if best_gain <= 0:
            return {'leaf': True, 'value': np.bincount(y).argmax()}

        # Split
        left_mask = X[:, best_feature] <= best_threshold

        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1, max_features_per_split),
            'right': self._build_tree(X[~left_mask], y[~left_mask], depth + 1, max_features_per_split)
        }

    def fit(self, X, y):
        """Fit the random forest."""
        self.n_classes = len(np.unique(y))
        n_features = X.shape[1]
        max_features_per_split = self._get_max_features(n_features)

        print(f"Training {self.n_estimators} trees...")

        for i in range(self.n_estimators):
            # Bootstrap sample
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y

            # Build tree
            tree = self._build_tree(X_sample, y_sample,
                                    max_features_per_split=max_features_per_split)
            self.trees.append(tree)

            if (i + 1) % 10 == 0:
                print(f"  Trained {i + 1}/{self.n_estimators} trees")

        return self

    def _predict_tree(self, x, node):
        """Predict using a single tree."""
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_tree(x, node['left'])
        return self._predict_tree(x, node['right'])

    def predict(self, X):
        """Predict using majority voting."""
        predictions = np.zeros((X.shape[0], self.n_estimators), dtype=int)

        for i, tree in enumerate(self.trees):
            predictions[:, i] = [self._predict_tree(x, tree) for x in X]

        # Majority voting
        return np.array([np.bincount(pred).argmax() for pred in predictions])

    def predict_proba(self, X):
        """Predict class probabilities."""
        predictions = np.zeros((X.shape[0], self.n_estimators), dtype=int)

        for i, tree in enumerate(self.trees):
            predictions[:, i] = [self._predict_tree(x, tree) for x in X]

        # Average probabilities
        proba = np.zeros((X.shape[0], self.n_classes))
        for i, pred in enumerate(predictions):
            counts = np.bincount(pred, minlength=self.n_classes)
            proba[i] = counts / self.n_estimators

        return proba


# =============================================================================
# PART 3: COMPARISON DEMONSTRATION
# =============================================================================

def demonstrate_ensemble_comparison():
    """
    Compare single tree vs ensemble methods.
    """
    print("\n" + "=" * 60)
    print("ENSEMBLE METHODS COMPARISON")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500

    # Create a complex decision boundary
    X = np.random.randn(n_samples, 2) * 2
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2 > 2) |
         ((X[:, 0] > 0) & (X[:, 1] > 0))).astype(int)
    # Add noise
    noise_idx = np.random.choice(n_samples, size=50, replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Models to compare
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Bagging': BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=10),
            n_estimators=50, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42
        ),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=50, max_depth=5, random_state=42
        )
    }

    results = {}
    cv_results = {}

    print("\nTraining and evaluating models...\n")
    print(f"{'Model':<20} {'Train Acc':>12} {'Test Acc':>12} {'CV Score':>12}")
    print("-" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()

        results[name] = {'train': train_acc, 'test': test_acc, 'cv': cv_score}
        cv_results[name] = cv_score

        print(f"{name:<20} {train_acc:>12.4f} {test_acc:>12.4f} {cv_score:>12.4f}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    # Create mesh grid for decision boundaries
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    for idx, (name, model) in enumerate(models.items()):
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap='RdYlGn')
        axes[idx].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
                          c='red', marker='o', edgecolors='black', label='Class 0')
        axes[idx].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
                          c='green', marker='s', edgecolors='black', label='Class 1')

        test_acc = results[name]['test']
        axes[idx].set_title(f'{name}\nTest Accuracy: {test_acc:.3f}')
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')

    # Model comparison bar chart
    names = list(results.keys())
    train_accs = [results[n]['train'] for n in names]
    test_accs = [results[n]['test'] for n in names]
    cv_accs = [results[n]['cv'] for n in names]

    x = np.arange(len(names))
    width = 0.25

    axes[5].bar(x - width, train_accs, width, label='Train', alpha=0.8)
    axes[5].bar(x, test_accs, width, label='Test', alpha=0.8)
    axes[5].bar(x + width, cv_accs, width, label='CV', alpha=0.8)
    axes[5].set_ylabel('Accuracy')
    axes[5].set_title('Model Comparison')
    axes[5].set_xticks(x)
    axes[5].set_xticklabels(names, rotation=45, ha='right')
    axes[5].legend()
    axes[5].grid(True, alpha=0.3, axis='y')
    axes[5].set_ylim(0.5, 1.05)

    plt.tight_layout()
    plt.savefig('/home/user/claude_ai_course/week2_classical_ml/day11_random_forests/ensemble_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved: ensemble_comparison.png")

    return results


# =============================================================================
# PART 4: EFFECT OF N_ESTIMATORS
# =============================================================================

def demonstrate_n_estimators_effect():
    """
    Show how number of trees affects performance.
    """
    print("\n" + "-" * 50)
    print("Effect of n_estimators")
    print("-" * 50)

    # Generate data
    np.random.seed(42)
    n_samples = 500
    X = np.random.randn(n_samples, 10)
    y = (X[:, 0] + X[:, 1] * 2 - X[:, 2] > 0).astype(int)
    # Add noise
    noise_idx = np.random.choice(n_samples, size=50, replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    n_estimators_range = [1, 5, 10, 25, 50, 100, 200, 300]
    train_scores = []
    test_scores = []
    oob_scores = []

    print("\n{:<15} {:>12} {:>12} {:>12}".format(
        "n_estimators", "Train", "Test", "OOB"))
    print("-" * 55)

    for n_est in n_estimators_range:
        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=10,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        train_acc = rf.score(X_train, y_train)
        test_acc = rf.score(X_test, y_test)
        oob_acc = rf.oob_score_

        train_scores.append(train_acc)
        test_scores.append(test_acc)
        oob_scores.append(oob_acc)

        print(f"{n_est:<15} {train_acc:>12.4f} {test_acc:>12.4f} {oob_acc:>12.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(n_estimators_range, train_scores, 'b-o', label='Train')
    axes[0].plot(n_estimators_range, test_scores, 'r-s', label='Test')
    axes[0].plot(n_estimators_range, oob_scores, 'g-^', label='OOB')
    axes[0].set_xlabel('Number of Trees')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Random Forest: Effect of n_estimators')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')

    # Compare feature importance stability
    n_runs = 5
    importance_std = []

    for n_est in n_estimators_range:
        importances = []
        for _ in range(n_runs):
            rf = RandomForestClassifier(n_estimators=n_est, max_depth=10)
            rf.fit(X_train, y_train)
            importances.append(rf.feature_importances_)
        importance_std.append(np.mean(np.std(importances, axis=0)))

    axes[1].plot(n_estimators_range, importance_std, 'purple', marker='o', linewidth=2)
    axes[1].set_xlabel('Number of Trees')
    axes[1].set_ylabel('Mean Std of Feature Importance')
    axes[1].set_title('Feature Importance Stability')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')

    plt.tight_layout()
    plt.savefig('/home/user/claude_ai_course/week2_classical_ml/day11_random_forests/n_estimators_effect.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved: n_estimators_effect.png")


# =============================================================================
# PART 5: CREATE LOAN DATASET (Same as Day 10)
# =============================================================================

def create_loan_dataset():
    """
    Create loan approval dataset (same as Day 10 for comparison).
    """
    np.random.seed(42)
    n_samples = 1000

    income = np.random.normal(50000, 20000, n_samples).clip(15000, 150000)
    credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
    loan_amount = np.random.normal(200000, 100000, n_samples).clip(50000, 500000)
    employment_years = np.random.exponential(5, n_samples).clip(0, 30)
    debt_ratio = np.random.beta(2, 5, n_samples)
    age = np.random.normal(40, 12, n_samples).clip(18, 75)
    num_credit_cards = np.random.poisson(3, n_samples).clip(0, 10)
    previous_defaults = np.random.binomial(1, 0.15, n_samples)

    score = (
        0.3 * (income / 150000) +
        0.25 * (credit_score / 850) +
        0.15 * (1 - loan_amount / 500000) +
        0.1 * (employment_years / 30) +
        0.1 * (1 - debt_ratio) +
        0.05 * (1 - previous_defaults) +
        0.05 * (1 - num_credit_cards / 10)
    )
    score += np.random.normal(0, 0.1, n_samples)
    approved = (score > 0.45).astype(int)

    X = np.column_stack([
        income, credit_score, loan_amount, employment_years,
        debt_ratio, age, num_credit_cards, previous_defaults
    ])

    feature_names = [
        'Income', 'Credit_Score', 'Loan_Amount', 'Employment_Years',
        'Debt_Ratio', 'Age', 'Num_Credit_Cards', 'Previous_Defaults'
    ]

    return X, approved, feature_names


# =============================================================================
# PART 6: ASSIGNMENT - IMPROVE DAY 10 MODEL WITH RANDOM FOREST
# =============================================================================

def improved_loan_prediction():
    """
    Assignment: Improve Day 10 loan approval model using Random Forest.
    Compare with single Decision Tree.
    """
    print("\n" + "=" * 60)
    print("ASSIGNMENT: IMPROVED LOAN APPROVAL WITH RANDOM FOREST")
    print("=" * 60)

    # Load data
    print("\nLoading loan approval dataset...")
    X, y, feature_names = create_loan_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nDataset: {len(y)} samples")
    print(f"Training: {len(y_train)}, Testing: {len(y_test)}")

    # ==========================================================================
    # MODEL 1: Single Decision Tree (Day 10 model)
    # ==========================================================================
    print("\n" + "-" * 50)
    print("MODEL 1: Single Decision Tree (from Day 10)")
    print("-" * 50)

    dt = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
    dt.fit(X_train, y_train)

    y_pred_dt = dt.predict(X_test)
    y_proba_dt = dt.predict_proba(X_test)[:, 1]
    acc_dt = accuracy_score(y_test, y_pred_dt)

    cv_dt = cross_val_score(dt, X_train, y_train, cv=5).mean()

    print(f"Test Accuracy: {acc_dt:.4f}")
    print(f"CV Score (5-fold): {cv_dt:.4f}")

    # ==========================================================================
    # MODEL 2: Random Forest from Scratch
    # ==========================================================================
    print("\n" + "-" * 50)
    print("MODEL 2: Random Forest from Scratch")
    print("-" * 50)

    rf_scratch = RandomForestScratch(
        n_estimators=50,
        max_depth=10,
        max_features='sqrt'
    )
    rf_scratch.fit(X_train, y_train)

    y_pred_scratch = rf_scratch.predict(X_test)
    y_proba_scratch = rf_scratch.predict_proba(X_test)[:, 1]
    acc_scratch = accuracy_score(y_test, y_pred_scratch)

    print(f"Test Accuracy: {acc_scratch:.4f}")

    # ==========================================================================
    # MODEL 3: Scikit-learn Random Forest
    # ==========================================================================
    print("\n" + "-" * 50)
    print("MODEL 3: Scikit-learn Random Forest")
    print("-" * 50)

    rf_sklearn = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    rf_sklearn.fit(X_train, y_train)

    y_pred_rf = rf_sklearn.predict(X_test)
    y_proba_rf = rf_sklearn.predict_proba(X_test)[:, 1]
    acc_rf = accuracy_score(y_test, y_pred_rf)

    cv_rf = cross_val_score(rf_sklearn, X_train, y_train, cv=5).mean()

    print(f"Test Accuracy: {acc_rf:.4f}")
    print(f"CV Score (5-fold): {cv_rf:.4f}")
    print(f"OOB Score: {rf_sklearn.oob_score_:.4f}")

    # ==========================================================================
    # MODEL 4: Gradient Boosting
    # ==========================================================================
    print("\n" + "-" * 50)
    print("MODEL 4: Gradient Boosting")
    print("-" * 50)

    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train, y_train)

    y_pred_gb = gb.predict(X_test)
    y_proba_gb = gb.predict_proba(X_test)[:, 1]
    acc_gb = accuracy_score(y_test, y_pred_gb)

    cv_gb = cross_val_score(gb, X_train, y_train, cv=5).mean()

    print(f"Test Accuracy: {acc_gb:.4f}")
    print(f"CV Score (5-fold): {cv_gb:.4f}")

    # ==========================================================================
    # COMPARISON
    # ==========================================================================
    print("\n" + "-" * 50)
    print("COMPARISON SUMMARY")
    print("-" * 50)

    print(f"\n{'Model':<25} {'Test Acc':>12} {'CV Score':>12} {'Improvement':>12}")
    print("-" * 65)
    print(f"{'Decision Tree':<25} {acc_dt:>12.4f} {cv_dt:>12.4f} {'(baseline)':>12}")
    print(f"{'RF from Scratch':<25} {acc_scratch:>12.4f} {'N/A':>12} {(acc_scratch-acc_dt)*100:>+11.2f}%")
    print(f"{'Random Forest (sklearn)':<25} {acc_rf:>12.4f} {cv_rf:>12.4f} {(acc_rf-acc_dt)*100:>+11.2f}%")
    print(f"{'Gradient Boosting':<25} {acc_gb:>12.4f} {cv_gb:>12.4f} {(acc_gb-acc_dt)*100:>+11.2f}%")

    # ==========================================================================
    # FEATURE IMPORTANCE ANALYSIS
    # ==========================================================================
    print("\n" + "-" * 50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("-" * 50)

    # Compare feature importance between models
    importance_dt = dt.feature_importances_
    importance_rf = rf_sklearn.feature_importances_
    importance_gb = gb.feature_importances_

    print(f"\n{'Feature':<20} {'DT':>10} {'RF':>10} {'GB':>10}")
    print("-" * 55)
    for i, feat in enumerate(feature_names):
        print(f"{feat:<20} {importance_dt[i]:>10.4f} {importance_rf[i]:>10.4f} {importance_gb[i]:>10.4f}")

    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Model Accuracy Comparison
    models = ['Decision\nTree', 'RF\nScratch', 'Random\nForest', 'Gradient\nBoosting']
    accuracies = [acc_dt, acc_scratch, acc_rf, acc_gb]
    colors = ['skyblue', 'lightgreen', 'green', 'orange']

    axes[0, 0].bar(models, accuracies, color=colors, edgecolor='black')
    axes[0, 0].set_ylabel('Test Accuracy')
    axes[0, 0].set_title('Model Comparison')
    axes[0, 0].set_ylim(min(accuracies) - 0.05, max(accuracies) + 0.02)
    for i, (m, acc) in enumerate(zip(models, accuracies)):
        axes[0, 0].text(i, acc + 0.005, f'{acc:.4f}', ha='center', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Plot 2: Feature Importance Comparison
    x = np.arange(len(feature_names))
    width = 0.25

    axes[0, 1].barh(x - width, importance_dt, width, label='Decision Tree', alpha=0.8)
    axes[0, 1].barh(x, importance_rf, width, label='Random Forest', alpha=0.8)
    axes[0, 1].barh(x + width, importance_gb, width, label='Gradient Boosting', alpha=0.8)
    axes[0, 1].set_yticks(x)
    axes[0, 1].set_yticklabels(feature_names)
    axes[0, 1].set_xlabel('Importance')
    axes[0, 1].set_title('Feature Importance Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    # Plot 3: ROC Curves
    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    fpr_gb, tpr_gb, _ = roc_curve(y_test, y_proba_gb)

    axes[0, 2].plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC={auc(fpr_dt, tpr_dt):.3f})')
    axes[0, 2].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc(fpr_rf, tpr_rf):.3f})')
    axes[0, 2].plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC={auc(fpr_gb, tpr_gb):.3f})')
    axes[0, 2].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 2].set_xlabel('False Positive Rate')
    axes[0, 2].set_ylabel('True Positive Rate')
    axes[0, 2].set_title('ROC Curves')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Confusion Matrix (Random Forest)
    cm = confusion_matrix(y_test, y_pred_rf)
    im = axes[1, 0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[1, 0].set_title('Confusion Matrix (Random Forest)')
    plt.colorbar(im, ax=axes[1, 0])

    classes = ['Denied', 'Approved']
    tick_marks = np.arange(len(classes))
    axes[1, 0].set_xticks(tick_marks)
    axes[1, 0].set_yticks(tick_marks)
    axes[1, 0].set_xticklabels(classes)
    axes[1, 0].set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 0].text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black",
                           fontsize=14)
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_xlabel('Predicted')

    # Plot 5: Learning Curve (Random Forest)
    train_sizes, train_scores_lc, test_scores_lc = learning_curve(
        rf_sklearn, X_train, y_train, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )

    train_mean = train_scores_lc.mean(axis=1)
    train_std = train_scores_lc.std(axis=1)
    test_mean = test_scores_lc.mean(axis=1)
    test_std = test_scores_lc.std(axis=1)

    axes[1, 1].fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                            alpha=0.1, color='blue')
    axes[1, 1].fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                            alpha=0.1, color='red')
    axes[1, 1].plot(train_sizes, train_mean, 'b-o', label='Training score')
    axes[1, 1].plot(train_sizes, test_mean, 'r-s', label='Cross-validation score')
    axes[1, 1].set_xlabel('Training Set Size')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Learning Curve (Random Forest)')
    axes[1, 1].legend(loc='lower right')
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Precision-Recall Curve
    precision_dt, recall_dt, _ = precision_recall_curve(y_test, y_proba_dt)
    precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_proba_rf)
    precision_gb, recall_gb, _ = precision_recall_curve(y_test, y_proba_gb)

    axes[1, 2].plot(recall_dt, precision_dt, label='Decision Tree')
    axes[1, 2].plot(recall_rf, precision_rf, label='Random Forest')
    axes[1, 2].plot(recall_gb, precision_gb, label='Gradient Boosting')
    axes[1, 2].set_xlabel('Recall')
    axes[1, 2].set_ylabel('Precision')
    axes[1, 2].set_title('Precision-Recall Curves')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/claude_ai_course/week2_classical_ml/day11_random_forests/loan_improvement.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved: loan_improvement.png")

    # Classification Report
    print("\n" + "-" * 50)
    print("CLASSIFICATION REPORT (Random Forest)")
    print("-" * 50)
    print(classification_report(y_test, y_pred_rf, target_names=['Denied', 'Approved']))

    return {
        'decision_tree': {'accuracy': acc_dt, 'cv': cv_dt},
        'rf_scratch': {'accuracy': acc_scratch},
        'random_forest': {'accuracy': acc_rf, 'cv': cv_rf, 'oob': rf_sklearn.oob_score_},
        'gradient_boosting': {'accuracy': acc_gb, 'cv': cv_gb},
        'improvement': (acc_rf - acc_dt) * 100
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DAY 11: RANDOM FORESTS & ENSEMBLE METHODS")
    print("=" * 60)

    # Part 1: Theory explanation
    explain_ensemble_methods()

    # Part 2: Ensemble comparison demonstration
    comparison_results = demonstrate_ensemble_comparison()

    # Part 3: Effect of n_estimators
    demonstrate_n_estimators_effect()

    # Part 4: Assignment - Improved loan prediction
    results = improved_loan_prediction()

    print("\n" + "=" * 60)
    print("DAY 11 COMPLETE!")
    print("=" * 60)
    print(f"""
    Key Takeaways:
    1. Bagging reduces variance by averaging multiple models
    2. Random Forest = Bagging + Random feature selection
    3. Boosting reduces bias by sequential error correction
    4. More trees generally improve performance (diminishing returns)
    5. Random Forest is robust and requires less tuning
    6. Feature importance is more stable with ensemble methods

    Results Summary:
    - Single Decision Tree accuracy: {results['decision_tree']['accuracy']:.4f}
    - Random Forest accuracy: {results['random_forest']['accuracy']:.4f}
    - Improvement: {results['improvement']:.2f}%
    - OOB Score: {results['random_forest']['oob']:.4f}

    Random Forest improved the loan approval model by reducing overfitting
    and providing more robust predictions!
    """)
