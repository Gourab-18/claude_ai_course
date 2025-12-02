# DAY 9: LOGISTIC REGRESSION - QUICK REFERENCE

## 📐 Core Formulas

### Sigmoid Function
```
σ(z) = 1 / (1 + e^(-z))

Properties:
• Range: (0, 1)
• σ(0) = 0.5
• σ(∞) → 1
• σ(-∞) → 0
```

### Prediction
```
z = w·x + b
ŷ = σ(z) = σ(w·x + b)

Classification:
• ŷ ≥ 0.5 → Class 1
• ŷ < 0.5 → Class 0
```

### Binary Cross-Entropy Loss
```
J(w,b) = -(1/m) Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

### Gradient Descent
```
∂J/∂w = (1/m) X^T(ŷ - y)
∂J/∂b = (1/m) Σ(ŷ - y)

w := w - α(∂J/∂w)
b := b - α(∂J/∂b)
```

---

## 💻 Code Templates

### From-Scratch Implementation
```python
class LogisticRegression:
    def __init__(self, lr=0.01, iters=1000):
        self.lr = lr
        self.iters = iters
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n, m = X.shape
        self.w = np.zeros(m)
        self.b = 0
        
        for _ in range(self.iters):
            z = X @ self.w + self.b
            y_pred = self.sigmoid(z)
            
            # Gradients
            dw = (1/n) * X.T @ (y_pred - y)
            db = (1/n) * np.sum(y_pred - y)
            
            # Update
            self.w -= self.lr * dw
            self.b -= self.lr * db
    
    def predict(self, X, threshold=0.5):
        z = X @ self.w + self.b
        probs = self.sigmoid(z)
        return (probs >= threshold).astype(int)
```

### Sklearn Usage
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
```

---

## 📊 Evaluation Metrics

### Confusion Matrix
```
              Predicted
           Neg (0)  Pos (1)
True Neg     TN       FP
True Pos     FN       TP
```

### Metrics Formulas
```python
Accuracy  = (TP + TN) / Total
Precision = TP / (TP + FP)  # Positive prediction quality
Recall    = TP / (TP + FN)  # Positive detection rate
F1-Score  = 2·(P·R)/(P+R)   # Harmonic mean

# Sklearn
from sklearn.metrics import *
accuracy_score(y_true, y_pred)
precision_score(y_true, y_pred)
recall_score(y_true, y_pred)
f1_score(y_true, y_pred)
roc_auc_score(y_true, y_proba)
```

### ROC Curve
```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_true, y_proba)
auc = roc_auc_score(y_true, y_proba)

plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
plt.plot([0,1], [0,1], 'k--')  # Random baseline
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
```

---

## 🎯 Key Concepts

### Sigmoid vs Linear
| Aspect | Linear | Logistic |
|--------|--------|----------|
| Output | ℝ (any) | (0, 1) |
| Use | Regression | Classification |
| Loss | MSE | Cross-Entropy |

### Why Cross-Entropy?
- ✅ Convex (guaranteed convergence)
- ✅ Better gradients
- ❌ MSE non-convex for classification

### Decision Boundary
```
Where: σ(w·x + b) = 0.5
Means: w·x + b = 0

2D Example:
w=[2, -1], b=-3
→ 2x₁ - x₂ - 3 = 0
```

---

## 📈 Our Results (Spam Classifier)

```
Dataset: 2,000 emails
Features: 38 (TF-IDF)
Train/Test: 1600/400

Test Results:
  Accuracy:  100.0% ✅ (Target: >85%)
  Precision: 1.000
  Recall:    1.000
  F1-Score:  1.000
  ROC-AUC:   1.000

Perfect Classification!
```

---

## 🎤 Interview Responses

### "Explain logistic regression"
> "Logistic regression predicts probability of binary outcome using sigmoid function. Maps linear combination w·x+b to (0,1) using σ(z)=1/(1+e^-z). Optimized with gradient descent minimizing binary cross-entropy loss. Decision boundary at 0.5 probability."

### "Why sigmoid function?"
> "Sigmoid maps any real number to valid probability (0,1). Has nice derivative σ'(z)=σ(z)(1-σ(z)) making gradient descent smooth. Output interpretable as P(positive class|features)."

### "Cross-entropy vs MSE?"
> "Cross-entropy creates convex optimization for classification - guaranteed global minimum. MSE creates non-convex landscape with local minima. Cross-entropy also has better gradient flow for binary predictions."

### "Interpret confusion matrix"
> "Shows true vs predicted classes. TN/TP are correct, FP/FN are errors. Precision=TP/(TP+FP) measures positive prediction quality. Recall=TP/(TP+FN) measures detection rate. F1 balances both."

---

## ⚡ Quick Commands

```bash
# Run analysis
python day9_logistic_regression_complete.py

# View results
open day9_spam_classifier_results.png
open day9_sigmoid_function.png
```

---

## 🔧 Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Overflow in exp() | z too large | Clip z: `z = np.clip(z, -500, 500)` |
| log(0) error | ŷ = 0 or 1 | Add epsilon: `ŷ = np.clip(ŷ, 1e-15, 1-1e-15)` |
| Poor convergence | Bad learning rate | Try α=0.01, 0.1, 1.0 |
| Low accuracy | Bad features | Feature engineering, scaling |

---

## 🚀 Applications

**Good for:**
- Spam detection ✅
- Fraud detection
- Medical diagnosis
- Customer churn
- Click prediction

**Not good for:**
- Multi-class (use softmax)
- Non-linear boundaries
- Image classification
- Time series

---

## 📊 Text Classification Pipeline

```python
# 1. TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(texts).toarray()

# 2. Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Model
model = LogisticRegression()
model.fit(X_scaled, y)

# 4. Evaluate
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

---

## 💡 Pro Tips

1. **Always scale features** for faster convergence
2. **Use cross-entropy** not MSE for classification
3. **Plot ROC curve** to understand model at different thresholds
4. **Check confusion matrix** not just accuracy
5. **Interpret coefficients** for feature importance

---

## ✅ Status

**Day 9**: ✅ COMPLETE  
**Target**: >85% accuracy  
**Achieved**: 100% (+15%)  
**Next**: Day 10 - Decision Trees

---

*Keep this cheat sheet handy for interviews! 📋*
