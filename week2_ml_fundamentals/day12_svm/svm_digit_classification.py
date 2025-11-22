"""
Day 12: Support Vector Machines (SVM)
Assignment: Classify handwritten digits (MNIST subset) using SVM

This module covers:
- Linear SVM: maximum margin classifier
- Kernel trick: RBF, polynomial kernels
- Hyperparameter tuning: C, gamma
- When to use SVM vs other algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.decomposition import PCA
import time
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PART 1: UNDERSTANDING SVM THEORY
# =============================================================================

def explain_svm_concepts():
    """Print explanation of key SVM concepts."""
    print("=" * 70)
    print("SUPPORT VECTOR MACHINES (SVM) - KEY CONCEPTS")
    print("=" * 70)

    print("""
    1. MAXIMUM MARGIN CLASSIFIER
    ----------------------------
    - SVM finds the hyperplane that maximizes the margin between classes
    - Margin = distance from hyperplane to nearest data points
    - Support vectors = data points closest to the decision boundary
    - Only support vectors matter for the decision boundary

    Mathematical formulation:
    - Hyperplane: w^T * x + b = 0
    - Maximize: 2 / ||w||  (margin width)
    - Subject to: y_i * (w^T * x_i + b) >= 1 for all i

    2. SOFT MARGIN (C PARAMETER)
    ----------------------------
    - Real data often not perfectly separable
    - C = regularization parameter (cost of misclassification)
    - Large C: smaller margin, fewer misclassifications (may overfit)
    - Small C: larger margin, more misclassifications (may underfit)

    3. KERNEL TRICK
    ---------------
    - Projects data to higher dimensions without explicit computation
    - Enables SVM to learn non-linear decision boundaries
    - K(x_i, x_j) = phi(x_i)^T * phi(x_j)

    Common Kernels:
    a) Linear: K(x, y) = x^T * y
       - Use when: many features, linearly separable

    b) RBF (Radial Basis Function): K(x, y) = exp(-gamma * ||x-y||^2)
       - Use when: non-linear boundaries needed
       - gamma: controls influence radius of each training point
       - Large gamma = narrow influence = complex boundary
       - Small gamma = wide influence = smoother boundary

    c) Polynomial: K(x, y) = (gamma * x^T * y + r)^d
       - degree d controls polynomial complexity

    4. WHEN TO USE SVM
    ------------------
    Good for:
    - High-dimensional data (text classification)
    - Clear margin of separation exists
    - Number of features > number of samples
    - Binary classification problems

    Not ideal for:
    - Very large datasets (slow training O(n^2) to O(n^3))
    - Noisy data with many overlapping classes
    - When probability estimates are crucial
    """)


# =============================================================================
# PART 2: VISUALIZING SVM CONCEPTS
# =============================================================================

def visualize_svm_margins():
    """Visualize how SVM finds the maximum margin and the effect of C."""
    print("\n" + "=" * 70)
    print("VISUALIZING SVM MARGINS")
    print("=" * 70)

    # Create linearly separable data
    np.random.seed(42)
    X_class1 = np.random.randn(20, 2) + np.array([2, 2])
    X_class2 = np.random.randn(20, 2) + np.array([-2, -2])
    X = np.vstack([X_class1, X_class2])
    y = np.array([1] * 20 + [-1] * 20)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    C_values = [0.01, 1, 100]

    for ax, C in zip(axes, C_values):
        # Train SVM
        svm = SVC(kernel='linear', C=C)
        svm.fit(X, y)

        # Create mesh for decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))

        # Get decision function values
        Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary and margins
        ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 20),
                    cmap='RdYlBu', alpha=0.5)
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'],
                   linestyles=['--', '-', '--'], linewidths=2)

        # Plot data points
        ax.scatter(X_class1[:, 0], X_class1[:, 1], c='red', edgecolors='black',
                   s=50, label='Class 1')
        ax.scatter(X_class2[:, 0], X_class2[:, 1], c='blue', edgecolors='black',
                   s=50, label='Class -1')

        # Highlight support vectors
        ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                   s=200, facecolors='none', edgecolors='green', linewidths=2,
                   label=f'Support Vectors ({len(svm.support_vectors_)})')

        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f'C = {C}\nMargin Width: {2/np.linalg.norm(svm.coef_):.3f}')
        ax.legend(loc='upper left', fontsize=8)

    plt.suptitle('Effect of C Parameter on SVM Margin', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('svm_margins_c_parameter.png', dpi=150, bbox_inches='tight')
    print("   Saved: svm_margins_c_parameter.png")

    return fig


def visualize_kernels():
    """Visualize different SVM kernels on non-linear data."""
    print("\n" + "=" * 70)
    print("VISUALIZING KERNEL EFFECTS")
    print("=" * 70)

    # Create non-linearly separable data (circles)
    np.random.seed(42)
    n_samples = 200

    # Inner circle
    r1 = np.random.uniform(0, 1, n_samples // 2)
    theta1 = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X_inner = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])

    # Outer ring
    r2 = np.random.uniform(2, 3, n_samples // 2)
    theta2 = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X_outer = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])

    X = np.vstack([X_inner, X_outer])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Add some noise
    X += np.random.randn(n_samples, 2) * 0.1

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    kernels = [
        ('linear', {'kernel': 'linear'}),
        ('rbf (gamma=0.5)', {'kernel': 'rbf', 'gamma': 0.5}),
        ('rbf (gamma=5)', {'kernel': 'rbf', 'gamma': 5}),
        ('polynomial (degree=3)', {'kernel': 'poly', 'degree': 3})
    ]

    for ax, (name, params) in zip(axes.flatten(), kernels):
        # Train SVM
        svm = SVC(C=1.0, **params)
        svm.fit(X, y)

        # Create mesh
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))

        # Get predictions
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        ax.contour(xx, yy, Z, colors='black', linewidths=1)

        # Plot data points
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', edgecolors='black',
                   s=30, label='Class 0')
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', edgecolors='black',
                   s=30, label='Class 1')

        # Highlight support vectors
        ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                   s=100, facecolors='none', edgecolors='green', linewidths=2)

        accuracy = accuracy_score(y, svm.predict(X))
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f'Kernel: {name}\nAccuracy: {accuracy:.2%}, SVs: {len(svm.support_vectors_)}')
        ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('SVM Kernel Comparison on Non-Linear Data', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('svm_kernels_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved: svm_kernels_comparison.png")

    return fig


def visualize_gamma_effect():
    """Visualize the effect of gamma parameter on RBF kernel."""
    print("\n" + "=" * 70)
    print("VISUALIZING GAMMA PARAMETER EFFECT")
    print("=" * 70)

    # Create XOR-like data
    np.random.seed(42)
    X1 = np.random.randn(50, 2) + np.array([1, 1])
    X2 = np.random.randn(50, 2) + np.array([-1, -1])
    X3 = np.random.randn(50, 2) + np.array([1, -1])
    X4 = np.random.randn(50, 2) + np.array([-1, 1])

    X = np.vstack([X1, X2, X3, X4])
    y = np.array([0] * 100 + [1] * 100)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    gamma_values = [0.01, 0.1, 1, 10]

    for ax, gamma in zip(axes, gamma_values):
        svm = SVC(kernel='rbf', gamma=gamma, C=1.0)
        svm.fit(X, y)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))

        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        ax.contour(xx, yy, Z, colors='black', linewidths=1)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', edgecolors='black', s=20)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', edgecolors='black', s=20)

        accuracy = accuracy_score(y, svm.predict(X))
        ax.set_title(f'gamma = {gamma}\nAccuracy: {accuracy:.2%}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

    plt.suptitle('Effect of Gamma on RBF Kernel (Low to High)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('svm_gamma_effect.png', dpi=150, bbox_inches='tight')
    print("   Saved: svm_gamma_effect.png")

    return fig


# =============================================================================
# PART 3: MNIST DIGIT CLASSIFICATION
# =============================================================================

def load_mnist_subset():
    """Load MNIST digits dataset (8x8 subset from sklearn)."""
    print("\n" + "=" * 70)
    print("LOADING MNIST DIGITS DATASET")
    print("=" * 70)

    # Load the digits dataset
    digits = datasets.load_digits()
    X, y = digits.data, digits.target

    print(f"   Dataset shape: {X.shape}")
    print(f"   Number of samples: {len(X)}")
    print(f"   Number of features: {X.shape[1]} (8x8 pixel images)")
    print(f"   Number of classes: {len(np.unique(y))} (digits 0-9)")
    print(f"   Class distribution: {np.bincount(y)}")

    return X, y, digits


def visualize_digits(digits_data):
    """Visualize sample digits from the dataset."""
    print("\n   Visualizing sample digits...")

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    for i, ax in enumerate(axes.flatten()):
        # Find first occurrence of each digit
        idx = np.where(digits_data.target == i)[0][0]
        ax.imshow(digits_data.images[idx], cmap='gray_r')
        ax.set_title(f'Digit: {i}')
        ax.axis('off')

    plt.suptitle('Sample Digits from Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mnist_sample_digits.png', dpi=150, bbox_inches='tight')
    print("   Saved: mnist_sample_digits.png")

    return fig


def train_linear_svm(X_train, X_test, y_train, y_test):
    """Train and evaluate Linear SVM classifier."""
    print("\n" + "=" * 70)
    print("LINEAR SVM CLASSIFIER")
    print("=" * 70)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Linear SVM
    print("\n   Training Linear SVM...")
    start_time = time.time()

    linear_svm = LinearSVC(C=1.0, max_iter=5000, random_state=42)
    linear_svm.fit(X_train_scaled, y_train)

    train_time = time.time() - start_time
    print(f"   Training time: {train_time:.2f} seconds")

    # Predictions
    y_pred = linear_svm.predict(X_test_scaled)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred))

    return linear_svm, scaler, accuracy


def train_rbf_svm(X_train, X_test, y_train, y_test):
    """Train and evaluate RBF SVM classifier."""
    print("\n" + "=" * 70)
    print("RBF KERNEL SVM CLASSIFIER")
    print("=" * 70)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train RBF SVM
    print("\n   Training RBF SVM (default parameters)...")
    start_time = time.time()

    rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    rbf_svm.fit(X_train_scaled, y_train)

    train_time = time.time() - start_time
    print(f"   Training time: {train_time:.2f} seconds")

    # Predictions
    y_pred = rbf_svm.predict(X_test_scaled)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Number of Support Vectors: {len(rbf_svm.support_vectors_)}")

    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred))

    return rbf_svm, scaler, accuracy


def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV."""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING WITH GridSearchCV")
    print("=" * 70)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly']
    }

    print("\n   Parameter Grid:")
    for key, values in param_grid.items():
        print(f"   - {key}: {values}")

    print(f"\n   Total combinations: {len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])}")
    print("   Running GridSearchCV with 5-fold cross-validation...")

    start_time = time.time()

    # GridSearchCV
    grid_search = GridSearchCV(
        SVC(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)

    search_time = time.time() - start_time
    print(f"\n   Search completed in {search_time:.2f} seconds")

    print("\n   Best Parameters:")
    for key, value in grid_search.best_params_.items():
        print(f"   - {key}: {value}")

    print(f"\n   Best Cross-Validation Score: {grid_search.best_score_:.4f}")

    # Show top 5 parameter combinations
    results = grid_search.cv_results_
    sorted_indices = np.argsort(results['mean_test_score'])[::-1][:5]

    print("\n   Top 5 Parameter Combinations:")
    print("   " + "-" * 60)
    for i, idx in enumerate(sorted_indices, 1):
        print(f"   {i}. Score: {results['mean_test_score'][idx]:.4f} (+/- {results['std_test_score'][idx]:.4f})")
        print(f"      Params: {results['params'][idx]}")

    return grid_search, scaler


def evaluate_best_model(grid_search, scaler, X_test, y_test):
    """Evaluate the best model from grid search."""
    print("\n" + "=" * 70)
    print("EVALUATING BEST MODEL")
    print("=" * 70)

    best_model = grid_search.best_estimator_
    X_test_scaled = scaler.transform(X_test)

    # Predictions
    y_pred = best_model.predict(X_test_scaled)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Number of Support Vectors: {len(best_model.support_vectors_)}")

    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix - Best SVM Model')
    plt.tight_layout()
    plt.savefig('svm_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("\n   Saved: svm_confusion_matrix.png")

    return accuracy


def visualize_misclassifications(model, scaler, X_test, y_test, images):
    """Visualize misclassified digits."""
    print("\n   Visualizing misclassified digits...")

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    # Find misclassified samples
    misclassified_idx = np.where(y_pred != y_test)[0]

    if len(misclassified_idx) == 0:
        print("   No misclassifications!")
        return None

    # Show up to 10 misclassified samples
    n_show = min(10, len(misclassified_idx))

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    for i, ax in enumerate(axes.flatten()):
        if i < n_show:
            idx = misclassified_idx[i]
            ax.imshow(X_test[idx].reshape(8, 8), cmap='gray_r')
            ax.set_title(f'True: {y_test[idx]}\nPred: {y_pred[idx]}', fontsize=10)
        ax.axis('off')

    plt.suptitle(f'Misclassified Digits ({len(misclassified_idx)} total)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('svm_misclassified_digits.png', dpi=150, bbox_inches='tight')
    print("   Saved: svm_misclassified_digits.png")

    return fig


def visualize_support_vectors_pca(model, scaler, X_train, y_train):
    """Visualize support vectors using PCA."""
    print("\n   Visualizing support vectors in 2D (PCA)...")

    X_train_scaled = scaler.transform(X_train)

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)
    sv_pca = pca.transform(model.support_vectors_)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot all points
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='tab10',
                         alpha=0.5, s=30, edgecolors='none')

    # Highlight support vectors
    ax.scatter(sv_pca[:, 0], sv_pca[:, 1], c='none', s=100,
               edgecolors='black', linewidths=1, label='Support Vectors')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax.set_title(f'Support Vectors Visualization (PCA)\n{len(model.support_vectors_)} SVs out of {len(X_train)} samples')

    plt.colorbar(scatter, label='Digit Class')
    ax.legend()

    plt.tight_layout()
    plt.savefig('svm_support_vectors_pca.png', dpi=150, bbox_inches='tight')
    print("   Saved: svm_support_vectors_pca.png")

    return fig


def compare_svm_algorithms():
    """Compare SVM with other classification algorithms."""
    print("\n" + "=" * 70)
    print("WHEN TO USE SVM VS OTHER ALGORITHMS")
    print("=" * 70)

    print("""
    COMPARISON TABLE:

    | Algorithm          | Best For                    | Limitations                |
    |--------------------|-----------------------------|-----------------------------|
    | SVM (Linear)       | High-dim, text data         | Large datasets slow         |
    | SVM (RBF)          | Non-linear, medium data     | Need to tune gamma          |
    | Logistic Regression| Probability estimates       | Linear boundaries only      |
    | Random Forest      | Complex non-linear, fast    | May overfit, less precise   |
    | KNN                | Simple, no training         | Slow prediction, memory     |
    | Neural Networks    | Very large data, images     | Need lots of data, GPU      |

    DECISION GUIDE:

    1. Start with Linear SVM if:
       - You have many features (> 1000)
       - Data is high-dimensional (text, genetic)
       - Dataset is small to medium (<10K samples)

    2. Use RBF SVM if:
       - Linear doesn't work well
       - Non-linear relationships exist
       - Dataset is medium-sized

    3. Consider alternatives if:
       - Dataset is very large (>100K): Use SGDClassifier or Neural Networks
       - Need probability outputs: Use Logistic Regression or calibrated SVM
       - Need feature importance: Use Random Forest or Gradient Boosting
       - Data is image-based: Consider CNNs
    """)


# =============================================================================
# PART 4: MAIN EXECUTION
# =============================================================================

def main():
    """Main function demonstrating SVM for digit classification."""
    print("=" * 70)
    print("DAY 12: SUPPORT VECTOR MACHINES")
    print("Handwritten Digit Classification with SVM")
    print("=" * 70)

    # 1. Explain SVM concepts
    explain_svm_concepts()

    # 2. Visualize SVM concepts
    visualize_svm_margins()
    visualize_kernels()
    visualize_gamma_effect()

    # 3. Load MNIST data
    X, y, digits = load_mnist_subset()
    visualize_digits(digits)

    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n   Training set size: {len(X_train)}")
    print(f"   Test set size: {len(X_test)}")

    # 5. Train Linear SVM
    linear_model, linear_scaler, linear_acc = train_linear_svm(
        X_train, X_test, y_train, y_test
    )

    # 6. Train RBF SVM
    rbf_model, rbf_scaler, rbf_acc = train_rbf_svm(
        X_train, X_test, y_train, y_test
    )

    # 7. Hyperparameter tuning
    grid_search, grid_scaler = hyperparameter_tuning(X_train, y_train)

    # 8. Evaluate best model
    best_acc = evaluate_best_model(grid_search, grid_scaler, X_test, y_test)

    # 9. Visualize misclassifications
    visualize_misclassifications(
        grid_search.best_estimator_, grid_scaler, X_test, y_test, digits.images
    )

    # 10. Visualize support vectors
    visualize_support_vectors_pca(
        grid_search.best_estimator_, grid_scaler, X_train, y_train
    )

    # 11. Compare with other algorithms
    compare_svm_algorithms()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Results Comparison:
    -------------------
    Linear SVM Accuracy:     {linear_acc:.4f} ({linear_acc*100:.2f}%)
    RBF SVM Accuracy:        {rbf_acc:.4f} ({rbf_acc*100:.2f}%)
    Tuned SVM Accuracy:      {best_acc:.4f} ({best_acc*100:.2f}%)

    Best Parameters: {grid_search.best_params_}

    Key Takeaways:
    --------------
    1. SVM finds the maximum margin hyperplane
    2. The C parameter controls the trade-off between margin size and errors
    3. RBF kernel enables non-linear decision boundaries
    4. Gamma controls the influence of each training point
    5. Hyperparameter tuning significantly improves performance
    6. Feature scaling is critical for SVM performance

    Visualizations Saved:
    ---------------------
    - svm_margins_c_parameter.png
    - svm_kernels_comparison.png
    - svm_gamma_effect.png
    - mnist_sample_digits.png
    - svm_confusion_matrix.png
    - svm_misclassified_digits.png
    - svm_support_vectors_pca.png
    """)

    plt.close('all')
    print("\nAll visualizations saved successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
