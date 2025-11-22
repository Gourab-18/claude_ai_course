# Week 1: Python & Exploratory Data Analysis

A comprehensive curriculum covering feature engineering, data transformation, and end-to-end EDA projects.

## Overview

This week covers essential data science fundamentals:

| Day | Topic | Assignment |
|-----|-------|------------|
| 6 | Feature Engineering Part 1 | Transform raw dataset to ML-ready format |
| 7 | Week 1 Project | Complete EDA report on Titanic dataset |

## Directory Structure

```
week1_python_eda/
├── day06_feature_engineering/
│   └── day06_feature_engineering.ipynb    # Feature scaling, encoding, datetime features
├── day07_week1_project/
│   └── day07_week1_project_eda.ipynb      # Complete EDA workflow and report
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
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

Then navigate to the day's notebook and run cells sequentially.

## Day-by-Day Content

### Day 6: Feature Engineering Part 1

**Topics Covered:**
- Feature Scaling
  - Standardization (Z-score normalization)
  - Min-Max Normalization
  - Robust Scaling
  - When to use which scaler
- Encoding Categorical Variables
  - Label Encoding
  - One-Hot Encoding
  - Ordinal Encoding
  - Target Encoding
- Feature Creation
  - Polynomial Features
  - Interaction Features
  - Aggregation Features
  - Domain-Specific Features
- Handling DateTime Features
  - Extracting components (year, month, day, hour)
  - Cyclical encoding (sin/cos)
  - Time-based features

**Key Code Examples:**
```python
# Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X[['category']])

# Cyclical encoding for datetime
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

**Assignment:**
Transform a raw e-commerce dataset with mixed data types (numerical, categorical, datetime) into ML-ready format.

---

### Day 7: Week 1 Project - Complete EDA

**Topics Covered:**
- Complete EDA Workflow
  - Data loading and overview
  - Data quality assessment
  - Missing value analysis
- Univariate Analysis
  - Distribution analysis
  - Statistical summaries
  - Outlier detection
- Bivariate Analysis
  - Correlation analysis
  - Target vs features
  - Statistical tests
- Multivariate Analysis
  - Correlation heatmaps
  - Pair plots
  - Feature interactions
- Data Preparation
  - Missing value treatment
  - Feature engineering
  - Encoding and scaling
- Documentation and Reporting
  - Key insights
  - Recommendations
  - Summary visualizations

**Key Code Examples:**
```python
# Missing value analysis
missing_df = pd.DataFrame({
    'Missing Count': df.isnull().sum(),
    'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
})

# Survival rate analysis
survival_rate = df.groupby('feature')['target'].mean()

# Feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
```

**Assignment:**
Create a comprehensive EDA report on the Titanic dataset with:
- Data quality assessment
- Pattern and anomaly identification
- Feature engineering
- Insights and recommendations

## Learning Outcomes

By completing this week, you will:

1. **Master Feature Scaling**
   - Understand when to use different scaling methods
   - Implement scalers manually and with scikit-learn
   - Handle outliers appropriately

2. **Handle Categorical Data**
   - Choose appropriate encoding methods
   - Avoid common pitfalls (dummy variable trap, target leakage)
   - Work with high-cardinality features

3. **Engineer Effective Features**
   - Create domain-specific features
   - Extract information from datetime columns
   - Use cyclical encoding for periodic data

4. **Conduct Professional EDA**
   - Follow systematic analysis workflow
   - Create publication-ready visualizations
   - Document findings clearly

5. **Prepare Data for ML**
   - Handle missing values strategically
   - Transform mixed data types
   - Create train-test splits properly

## Tips for Success

1. **Run each cell sequentially** to understand the flow
2. **Experiment** with different parameters and methods
3. **Read the comments** - they contain important explanations
4. **Create your own visualizations** to explore the data
5. **Document your findings** as you go
6. **Apply these techniques** to your own datasets

## Troubleshooting

### Jupyter Notebook Issues

```bash
# Reset kernel if stuck
Kernel > Restart Kernel

# Clear all outputs
Edit > Clear All Outputs
```

### Memory Issues

```python
# Reduce dataset size for testing
df_sample = df.sample(n=1000, random_state=42)

# Use chunked reading for large files
chunks = pd.read_csv('large_file.csv', chunksize=10000)
```

### Visualization Issues

```python
# If plots don't show
%matplotlib inline

# Increase figure size
plt.figure(figsize=(12, 8))
```

## Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Feature Engineering Book](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)

## Next Steps

After completing Week 1, you'll be ready for:
- Week 2: Classical Machine Learning (starting with Linear Regression)
- Applying EDA skills to real-world datasets
- Building end-to-end ML pipelines

---

Happy Learning!
