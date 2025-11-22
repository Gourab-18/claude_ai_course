"""
Day 13: K-Means & Clustering
Assignment: Segment customers into groups, interpret each cluster

This module covers:
- Unsupervised learning introduction
- K-means algorithm: initialization, assignment, update
- Choosing K: elbow method, silhouette score
- Hierarchical clustering basics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PART 1: UNSUPERVISED LEARNING CONCEPTS
# =============================================================================

def explain_unsupervised_learning():
    """Explain unsupervised learning concepts."""
    print("=" * 70)
    print("UNSUPERVISED LEARNING - KEY CONCEPTS")
    print("=" * 70)

    print("""
    SUPERVISED vs UNSUPERVISED LEARNING
    ------------------------------------
    Supervised Learning:
    - Has labeled data (X, y pairs)
    - Goal: Learn mapping X -> y
    - Examples: Classification, Regression

    Unsupervised Learning:
    - Only has input data X (no labels)
    - Goal: Find hidden patterns in data
    - Examples: Clustering, Dimensionality Reduction

    CLUSTERING
    ----------
    Goal: Group similar data points together

    Types:
    1. Partitioning: K-Means, K-Medoids
       - Divide data into K non-overlapping groups
       - Each point belongs to exactly one cluster

    2. Hierarchical: Agglomerative, Divisive
       - Build tree of clusters (dendrogram)
       - Can visualize at different levels

    3. Density-based: DBSCAN, HDBSCAN
       - Find dense regions separated by sparse areas
       - Can find arbitrary shaped clusters

    APPLICATIONS
    ------------
    - Customer Segmentation: Group customers by behavior
    - Document Clustering: Organize text documents
    - Image Segmentation: Partition images
    - Anomaly Detection: Find outliers
    - Gene Expression Analysis: Group similar genes
    """)


# =============================================================================
# PART 2: K-MEANS ALGORITHM
# =============================================================================

def explain_kmeans():
    """Explain K-Means algorithm in detail."""
    print("\n" + "=" * 70)
    print("K-MEANS ALGORITHM")
    print("=" * 70)

    print("""
    K-MEANS ALGORITHM STEPS
    -----------------------

    1. INITIALIZATION
       Choose K initial centroids (cluster centers)
       Methods:
       - Random: Pick K random points
       - K-Means++: Smart initialization (sklearn default)
         * First centroid: random point
         * Next centroids: probability proportional to distance^2

    2. ASSIGNMENT
       Assign each point to nearest centroid
       distance(x, centroid) = ||x - centroid||^2

    3. UPDATE
       Recalculate centroids as mean of assigned points
       centroid_k = mean(points in cluster k)

    4. REPEAT
       Steps 2-3 until convergence:
       - Centroids don't change (or change < threshold)
       - Maximum iterations reached

    OBJECTIVE FUNCTION (Inertia)
    ----------------------------
    Minimize: Sum of squared distances from points to their centroids
    J = Σ Σ ||x_i - μ_k||^2
        k  i∈k

    Where:
    - K = number of clusters
    - μ_k = centroid of cluster k
    - x_i = data points in cluster k

    PROPERTIES
    ----------
    - Always converges (to local minimum)
    - Result depends on initialization (run multiple times)
    - Assumes spherical clusters of similar size
    - Sensitive to outliers
    """)


def visualize_kmeans_steps():
    """Visualize K-Means algorithm step by step."""
    print("\n   Visualizing K-Means algorithm steps...")

    # Create sample data
    np.random.seed(42)
    n_samples = 150

    # Three clusters
    X1 = np.random.randn(n_samples // 3, 2) * 0.5 + np.array([2, 2])
    X2 = np.random.randn(n_samples // 3, 2) * 0.5 + np.array([-2, 2])
    X3 = np.random.randn(n_samples // 3, 2) * 0.5 + np.array([0, -2])
    X = np.vstack([X1, X2, X3])

    # Initialize centroids randomly
    np.random.seed(0)
    k = 3
    initial_centroids = X[np.random.choice(len(X), k, replace=False)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    centroids = initial_centroids.copy()
    colors = ['red', 'blue', 'green']

    for step, ax in enumerate(axes.flatten()):
        if step == 0:
            # Initial state
            ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5, s=50)
            ax.scatter(centroids[:, 0], centroids[:, 1], c=colors, marker='X',
                       s=200, edgecolors='black', linewidths=2)
            ax.set_title('Step 0: Initial Centroids')
        else:
            # Assignment step
            distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)

            # Plot assignments
            for i in range(k):
                mask = labels == i
                ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.5, s=50)

            ax.scatter(centroids[:, 0], centroids[:, 1], c=colors, marker='X',
                       s=200, edgecolors='black', linewidths=2)

            # Update step (for next iteration)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

            # Draw arrows showing centroid movement
            for i in range(k):
                if not np.allclose(centroids[i], new_centroids[i]):
                    ax.annotate('', xy=new_centroids[i], xytext=centroids[i],
                                arrowprops=dict(arrowstyle='->', color='black', lw=2))

            centroids = new_centroids
            ax.set_title(f'Step {step}: Assign + Update')

        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)

    plt.suptitle('K-Means Algorithm Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('kmeans_steps.png', dpi=150, bbox_inches='tight')
    print("   Saved: kmeans_steps.png")

    return fig


# =============================================================================
# PART 3: CHOOSING K
# =============================================================================

def elbow_method(X, max_k=10):
    """Apply elbow method to find optimal K."""
    print("\n" + "=" * 70)
    print("ELBOW METHOD")
    print("=" * 70)

    print("""
    The elbow method looks at the inertia (within-cluster sum of squares)
    as K increases. The "elbow" point is where adding more clusters
    doesn't significantly reduce inertia.
    """)

    inertias = []
    K_range = range(1, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        print(f"   K={k}: Inertia = {kmeans.inertia_:.2f}")

    # Plot elbow curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Inertia (Within-cluster Sum of Squares)')
    ax.set_title('Elbow Method for Optimal K')
    ax.grid(True, alpha=0.3)

    # Mark the elbow point (heuristic: maximum curvature)
    # Calculate rate of change
    diffs = np.diff(inertias)
    diff2 = np.diff(diffs)
    elbow_k = np.argmax(np.abs(diff2)) + 2  # +2 because of double diff

    ax.axvline(x=elbow_k, color='red', linestyle='--', linewidth=2,
               label=f'Suggested K = {elbow_k}')
    ax.legend()

    plt.tight_layout()
    plt.savefig('elbow_method.png', dpi=150, bbox_inches='tight')
    print(f"\n   Saved: elbow_method.png")
    print(f"   Suggested K (elbow point): {elbow_k}")

    return inertias, elbow_k


def silhouette_analysis(X, max_k=10):
    """Perform silhouette analysis to find optimal K."""
    print("\n" + "=" * 70)
    print("SILHOUETTE ANALYSIS")
    print("=" * 70)

    print("""
    Silhouette Score measures how similar a point is to its own cluster
    compared to other clusters.

    For each point i:
    - a(i) = mean distance to other points in same cluster
    - b(i) = mean distance to points in nearest other cluster
    - s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Interpretation:
    - s = 1: Perfect clustering
    - s = 0: On cluster boundary
    - s = -1: Wrong cluster

    Overall score = mean of all s(i)
    """)

    silhouette_scores = []
    K_range = range(2, max_k + 1)  # Start from 2 (silhouette needs at least 2 clusters)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        print(f"   K={k}: Silhouette Score = {score:.4f}")

    # Plot silhouette scores
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(K_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Analysis for Optimal K')
    ax.grid(True, alpha=0.3)

    # Mark best K
    best_k = list(K_range)[np.argmax(silhouette_scores)]
    ax.axvline(x=best_k, color='red', linestyle='--', linewidth=2,
               label=f'Best K = {best_k} (score = {max(silhouette_scores):.4f})')
    ax.legend()

    plt.tight_layout()
    plt.savefig('silhouette_scores.png', dpi=150, bbox_inches='tight')
    print(f"\n   Saved: silhouette_scores.png")
    print(f"   Best K (highest silhouette): {best_k}")

    return silhouette_scores, best_k


def visualize_silhouette_plot(X, k):
    """Create detailed silhouette plot for a specific K."""
    print(f"\n   Creating silhouette plot for K={k}...")

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Silhouette plot
    y_lower = 10
    colors = plt.cm.tab10(np.linspace(0, 1, k))

    for i in range(k):
        cluster_silhouette_values = sample_silhouette_values[labels == i]
        cluster_silhouette_values.sort()

        cluster_size = len(cluster_silhouette_values)
        y_upper = y_lower + cluster_size

        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_silhouette_values,
                          facecolor=colors[i], edgecolor=colors[i], alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * cluster_size, str(i))
        y_lower = y_upper + 10

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--",
                label=f'Average: {silhouette_avg:.3f}')
    ax1.set_xlabel('Silhouette Coefficient')
    ax1.set_ylabel('Cluster')
    ax1.set_title(f'Silhouette Plot (K={k})')
    ax1.legend()

    # Cluster visualization
    scatter = ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=50, alpha=0.7)
    ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='red', marker='X', s=200, edgecolors='black', linewidths=2,
                label='Centroids')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title(f'Cluster Assignments (K={k})')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'silhouette_plot_k{k}.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: silhouette_plot_k{k}.png")

    return fig


# =============================================================================
# PART 4: HIERARCHICAL CLUSTERING
# =============================================================================

def explain_hierarchical_clustering():
    """Explain hierarchical clustering concepts."""
    print("\n" + "=" * 70)
    print("HIERARCHICAL CLUSTERING")
    print("=" * 70)

    print("""
    HIERARCHICAL CLUSTERING
    -----------------------
    Builds a tree (dendrogram) of clusters.

    AGGLOMERATIVE (Bottom-up):
    1. Start: Each point is its own cluster
    2. Merge: Combine two closest clusters
    3. Repeat: Until all points in one cluster

    DIVISIVE (Top-down):
    1. Start: All points in one cluster
    2. Split: Divide cluster into two
    3. Repeat: Until each point is its own cluster

    LINKAGE METHODS (How to measure cluster distance):
    -------------------------------------------------
    1. Single: min distance between points
       - Tends to create long chains
       - Good for irregular shapes

    2. Complete: max distance between points
       - Creates compact clusters
       - Sensitive to outliers

    3. Average: mean distance between points
       - Balance between single and complete

    4. Ward: minimize variance increase
       - Creates spherical clusters
       - Often best for most cases

    DENDROGRAM
    ----------
    - Visual representation of clustering hierarchy
    - Height represents distance at which clusters merge
    - Cut at different heights for different K
    """)


def hierarchical_clustering_demo(X):
    """Demonstrate hierarchical clustering with dendrogram."""
    print("\n   Performing hierarchical clustering...")

    # Use a subset for cleaner dendrogram
    np.random.seed(42)
    subset_idx = np.random.choice(len(X), min(50, len(X)), replace=False)
    X_subset = X[subset_idx]

    # Compute linkage matrix
    linkage_matrix = linkage(X_subset, method='ward')

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Dendrogram
    ax1 = axes[0, 0]
    dendrogram(linkage_matrix, ax=ax1, truncate_mode='level', p=5)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Distance')
    ax1.set_title('Dendrogram (Ward Linkage)')

    # Compare linkage methods on full data
    linkage_methods = ['ward', 'complete', 'average', 'single']
    colors = ['red', 'blue', 'green', 'purple']

    for ax, method in zip([axes[0, 1], axes[1, 0], axes[1, 1]], linkage_methods[1:]):
        hierarchical = AgglomerativeClustering(n_clusters=3, linkage=method)
        labels = hierarchical.fit_predict(X)

        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=50, alpha=0.7)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f'Hierarchical Clustering: {method.capitalize()} Linkage')

    # Ward linkage (best)
    hierarchical_ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels_ward = hierarchical_ward.fit_predict(X)
    axes[0, 1].clear()
    axes[0, 1].scatter(X[:, 0], X[:, 1], c=labels_ward, cmap='tab10', s=50, alpha=0.7)
    axes[0, 1].set_xlabel('Feature 1')
    axes[0, 1].set_ylabel('Feature 2')
    axes[0, 1].set_title('Hierarchical Clustering: Ward Linkage')

    plt.tight_layout()
    plt.savefig('hierarchical_clustering.png', dpi=150, bbox_inches='tight')
    print("   Saved: hierarchical_clustering.png")

    return fig


# =============================================================================
# PART 5: CUSTOMER SEGMENTATION CASE STUDY
# =============================================================================

def create_customer_data():
    """Create synthetic customer data for segmentation."""
    print("\n" + "=" * 70)
    print("CUSTOMER SEGMENTATION CASE STUDY")
    print("=" * 70)

    np.random.seed(42)
    n_customers = 500

    # Create different customer segments
    # Segment 1: High Income, High Spending (Premium Customers)
    income_1 = np.random.normal(90000, 15000, n_customers // 4)
    spending_1 = np.random.normal(80, 10, n_customers // 4)
    frequency_1 = np.random.normal(25, 5, n_customers // 4)
    recency_1 = np.random.normal(10, 5, n_customers // 4)

    # Segment 2: High Income, Low Spending (Potential Targets)
    income_2 = np.random.normal(85000, 12000, n_customers // 4)
    spending_2 = np.random.normal(30, 8, n_customers // 4)
    frequency_2 = np.random.normal(8, 3, n_customers // 4)
    recency_2 = np.random.normal(45, 15, n_customers // 4)

    # Segment 3: Low Income, Careful Spenders
    income_3 = np.random.normal(35000, 8000, n_customers // 4)
    spending_3 = np.random.normal(25, 7, n_customers // 4)
    frequency_3 = np.random.normal(12, 4, n_customers // 4)
    recency_3 = np.random.normal(30, 10, n_customers // 4)

    # Segment 4: Average Everything (Regular Customers)
    income_4 = np.random.normal(55000, 10000, n_customers // 4)
    spending_4 = np.random.normal(50, 12, n_customers // 4)
    frequency_4 = np.random.normal(15, 5, n_customers // 4)
    recency_4 = np.random.normal(20, 8, n_customers // 4)

    # Combine all segments
    data = {
        'Annual_Income': np.concatenate([income_1, income_2, income_3, income_4]),
        'Spending_Score': np.concatenate([spending_1, spending_2, spending_3, spending_4]),
        'Purchase_Frequency': np.concatenate([frequency_1, frequency_2, frequency_3, frequency_4]),
        'Days_Since_Last_Purchase': np.concatenate([recency_1, recency_2, recency_3, recency_4])
    }

    # Add some additional features
    df = pd.DataFrame(data)
    df['Customer_ID'] = range(1, n_customers + 1)

    # Ensure non-negative values
    df['Annual_Income'] = np.clip(df['Annual_Income'], 15000, None)
    df['Spending_Score'] = np.clip(df['Spending_Score'], 1, 100)
    df['Purchase_Frequency'] = np.clip(df['Purchase_Frequency'], 1, None)
    df['Days_Since_Last_Purchase'] = np.clip(df['Days_Since_Last_Purchase'], 1, None)

    print(f"\n   Created customer dataset with {len(df)} customers")
    print(f"\n   Features:")
    print(f"   - Annual_Income: Customer's annual income ($)")
    print(f"   - Spending_Score: Score based on purchase behavior (1-100)")
    print(f"   - Purchase_Frequency: Number of purchases per year")
    print(f"   - Days_Since_Last_Purchase: Recency metric")

    print(f"\n   Dataset Statistics:")
    print(df.describe().round(2))

    return df


def perform_customer_segmentation(df):
    """Perform customer segmentation using K-Means."""
    print("\n" + "=" * 70)
    print("PERFORMING CUSTOMER SEGMENTATION")
    print("=" * 70)

    # Select features for clustering
    features = ['Annual_Income', 'Spending_Score', 'Purchase_Frequency', 'Days_Since_Last_Purchase']
    X = df[features].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal K using silhouette
    print("\n   Finding optimal number of clusters...")
    silhouette_scores = []
    K_range = range(2, 10)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        print(f"   K={k}: Silhouette = {score:.4f}")

    optimal_k = list(K_range)[np.argmax(silhouette_scores)]
    print(f"\n   Optimal K: {optimal_k}")

    # Fit final model
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    return df, kmeans, scaler, X_scaled, optimal_k


def interpret_clusters(df):
    """Interpret and describe each customer cluster."""
    print("\n" + "=" * 70)
    print("CLUSTER INTERPRETATION")
    print("=" * 70)

    features = ['Annual_Income', 'Spending_Score', 'Purchase_Frequency', 'Days_Since_Last_Purchase']

    # Calculate cluster statistics
    cluster_stats = df.groupby('Cluster')[features].agg(['mean', 'std', 'count'])
    print("\n   Cluster Statistics:")
    print(cluster_stats.round(2))

    # Detailed interpretation
    cluster_profiles = []

    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster][features]
        profile = {
            'Cluster': cluster,
            'Size': len(cluster_data),
            'Size_%': len(cluster_data) / len(df) * 100,
            'Avg_Income': cluster_data['Annual_Income'].mean(),
            'Avg_Spending': cluster_data['Spending_Score'].mean(),
            'Avg_Frequency': cluster_data['Purchase_Frequency'].mean(),
            'Avg_Recency': cluster_data['Days_Since_Last_Purchase'].mean()
        }
        cluster_profiles.append(profile)

    profiles_df = pd.DataFrame(cluster_profiles)
    print("\n   Cluster Profiles Summary:")
    print(profiles_df.round(2))

    # Naming clusters based on characteristics
    print("\n   Cluster Interpretations:")
    print("   " + "-" * 60)

    for _, row in profiles_df.iterrows():
        cluster = int(row['Cluster'])
        income_level = 'High' if row['Avg_Income'] > 65000 else ('Medium' if row['Avg_Income'] > 45000 else 'Low')
        spending_level = 'High' if row['Avg_Spending'] > 60 else ('Medium' if row['Avg_Spending'] > 40 else 'Low')
        frequency_level = 'Frequent' if row['Avg_Frequency'] > 18 else ('Moderate' if row['Avg_Frequency'] > 10 else 'Infrequent')
        recency_level = 'Recent' if row['Avg_Recency'] < 20 else ('Moderate' if row['Avg_Recency'] < 35 else 'Lapsed')

        print(f"\n   CLUSTER {cluster}:")
        print(f"   - Size: {int(row['Size'])} customers ({row['Size_%']:.1f}%)")
        print(f"   - Income: {income_level} (${row['Avg_Income']:,.0f})")
        print(f"   - Spending: {spending_level} (Score: {row['Avg_Spending']:.1f})")
        print(f"   - Frequency: {frequency_level} ({row['Avg_Frequency']:.1f} purchases/year)")
        print(f"   - Recency: {recency_level} ({row['Avg_Recency']:.0f} days)")

        # Marketing recommendations
        if spending_level == 'High' and income_level == 'High':
            recommendation = "VIP Program: Exclusive offers, early access, loyalty rewards"
        elif income_level == 'High' and spending_level != 'High':
            recommendation = "Engagement Campaign: Personalized recommendations, upselling"
        elif recency_level == 'Lapsed':
            recommendation = "Re-activation: Win-back campaigns, special discounts"
        elif frequency_level == 'Frequent':
            recommendation = "Retention: Loyalty program, referral incentives"
        else:
            recommendation = "Awareness: Cross-sell, product education, entry-level offers"

        print(f"   - Recommendation: {recommendation}")

    return profiles_df


def visualize_customer_segments(df, X_scaled, kmeans, optimal_k):
    """Create visualizations for customer segments."""
    print("\n   Creating segment visualizations...")

    fig = plt.figure(figsize=(16, 12))

    # 1. PCA visualization
    ax1 = fig.add_subplot(2, 2, 1)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'],
                          cmap='tab10', alpha=0.6, s=50)

    # Plot centroids
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax1.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                c=range(optimal_k), cmap='tab10', marker='X', s=200,
                edgecolors='black', linewidths=2)

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title('Customer Segments (PCA)')
    plt.colorbar(scatter, ax=ax1, label='Cluster')

    # 2. Income vs Spending Score
    ax2 = fig.add_subplot(2, 2, 2)
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        ax2.scatter(cluster_data['Annual_Income'] / 1000,
                    cluster_data['Spending_Score'],
                    label=f'Cluster {cluster}', alpha=0.6, s=50)
    ax2.set_xlabel('Annual Income ($K)')
    ax2.set_ylabel('Spending Score')
    ax2.set_title('Income vs Spending Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Cluster sizes
    ax3 = fig.add_subplot(2, 2, 3)
    cluster_sizes = df['Cluster'].value_counts().sort_index()
    bars = ax3.bar(cluster_sizes.index, cluster_sizes.values, color=plt.cm.tab10(np.linspace(0, 1, optimal_k)))
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Number of Customers')
    ax3.set_title('Cluster Size Distribution')
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{int(height)}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom')

    # 4. Radar chart for cluster profiles
    ax4 = fig.add_subplot(2, 2, 4, projection='polar')
    features = ['Annual_Income', 'Spending_Score', 'Purchase_Frequency', 'Days_Since_Last_Purchase']

    # Normalize cluster means for radar chart
    cluster_means = df.groupby('Cluster')[features].mean()
    cluster_means_normalized = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    for cluster in sorted(df['Cluster'].unique()):
        values = cluster_means_normalized.loc[cluster].tolist()
        values += values[:1]
        ax4.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster}')
        ax4.fill(angles, values, alpha=0.1)

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(['Income', 'Spending', 'Frequency', 'Recency'])
    ax4.set_title('Cluster Profiles (Normalized)')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig('customer_segments.png', dpi=150, bbox_inches='tight')
    print("   Saved: customer_segments.png")

    return fig


def create_segment_heatmap(df):
    """Create heatmap of cluster characteristics."""
    print("\n   Creating segment heatmap...")

    features = ['Annual_Income', 'Spending_Score', 'Purchase_Frequency', 'Days_Since_Last_Purchase']

    # Calculate z-scores for each cluster
    cluster_means = df.groupby('Cluster')[features].mean()
    overall_means = df[features].mean()
    overall_stds = df[features].std()

    z_scores = (cluster_means - overall_means) / overall_stds

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(z_scores.values, cmap='RdYlGn', aspect='auto', vmin=-2, vmax=2)

    # Labels
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(['Income', 'Spending', 'Frequency', 'Recency'], rotation=45, ha='right')
    ax.set_yticks(range(len(z_scores)))
    ax.set_yticklabels([f'Cluster {i}' for i in z_scores.index])

    # Add text annotations
    for i in range(len(z_scores)):
        for j in range(len(features)):
            text = ax.text(j, i, f'{z_scores.iloc[i, j]:.2f}',
                           ha='center', va='center', color='black', fontsize=12)

    ax.set_title('Cluster Characteristics (Z-Scores)\nGreen = Above Average, Red = Below Average')
    plt.colorbar(im, ax=ax, label='Z-Score')

    plt.tight_layout()
    plt.savefig('segment_heatmap.png', dpi=150, bbox_inches='tight')
    print("   Saved: segment_heatmap.png")

    return fig


# =============================================================================
# PART 6: MAIN EXECUTION
# =============================================================================

def main():
    """Main function demonstrating K-Means and clustering."""
    print("=" * 70)
    print("DAY 13: K-MEANS & CLUSTERING")
    print("Customer Segmentation Case Study")
    print("=" * 70)

    # 1. Explain concepts
    explain_unsupervised_learning()
    explain_kmeans()

    # 2. Visualize K-Means algorithm
    visualize_kmeans_steps()

    # 3. Create sample data for finding K
    np.random.seed(42)
    X1 = np.random.randn(100, 2) * 0.5 + np.array([2, 2])
    X2 = np.random.randn(100, 2) * 0.5 + np.array([-2, 2])
    X3 = np.random.randn(100, 2) * 0.5 + np.array([0, -2])
    X_sample = np.vstack([X1, X2, X3])

    # 4. Elbow method
    inertias, elbow_k = elbow_method(X_sample)

    # 5. Silhouette analysis
    sil_scores, sil_k = silhouette_analysis(X_sample)

    # 6. Detailed silhouette plot
    visualize_silhouette_plot(X_sample, 3)

    # 7. Hierarchical clustering
    explain_hierarchical_clustering()
    hierarchical_clustering_demo(X_sample)

    # 8. Customer segmentation case study
    df = create_customer_data()
    df, kmeans, scaler, X_scaled, optimal_k = perform_customer_segmentation(df)
    profiles = interpret_clusters(df)
    visualize_customer_segments(df, X_scaled, kmeans, optimal_k)
    create_segment_heatmap(df)

    # Save customer data
    df.to_csv('customer_segments_data.csv', index=False)
    print("\n   Saved: customer_segments_data.csv")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    Key Concepts Covered:
    ---------------------
    1. UNSUPERVISED LEARNING
       - No labels, discover hidden patterns
       - Clustering groups similar points

    2. K-MEANS ALGORITHM
       - Initialize centroids
       - Assign points to nearest centroid
       - Update centroids as cluster means
       - Repeat until convergence

    3. CHOOSING K
       - Elbow Method: Look for "elbow" in inertia curve
       - Silhouette Score: Maximize cluster separation

    4. HIERARCHICAL CLUSTERING
       - Builds dendrogram (tree of clusters)
       - Linkage methods: Ward (best), Complete, Average, Single

    5. CUSTOMER SEGMENTATION
       - Practical application of clustering
       - Feature selection and scaling
       - Cluster interpretation for business insights

    Visualizations Saved:
    ---------------------
    - kmeans_steps.png
    - elbow_method.png
    - silhouette_scores.png
    - silhouette_plot_k3.png
    - hierarchical_clustering.png
    - customer_segments.png
    - segment_heatmap.png
    - customer_segments_data.csv
    """)

    plt.close('all')
    print("\nAll visualizations saved successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
