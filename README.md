# 🌳 Gini Impurity — Decision Tree Fundamentals

A hands-on Python notebook implementing **Gini Impurity** from scratch — covering the math, best-split search, tree visualization, and a comparison with Entropy, applied to synthetic data and the classic Iris dataset.

---

## 📐 Core Equations

### 1. Gini Index

Measures the impurity of a single node:
```
Gini(t) = 1 - Σ pᵢ²
```

Where `pᵢ` is the proportion of class `i` in node `t`.

| Value | Meaning |
|-------|---------|
| `0.0` | Perfectly pure — all samples are one class |
| `0.5` | Maximum impurity — classes equally mixed (binary) |

---

### 2. Gini Split

Weighted impurity after splitting into left and right children:
```
Gini_split = (n_left / n) × Gini(left) + (n_right / n) × Gini(right)
```

→ Choose the split with the **smallest** Gini Split value.

---

### 3. Gini Gain

How much impurity is reduced after a split:
```
Gini Gain = Gini(parent) - Gini_split
```

→ Choose the split with the **largest** Gini Gain.

---

### 4. Multi-class Gini Index

Extends to C classes:
```
Gini(t) = 1 - Σ pᵢ²    for i = 1 to C
```

Example — Iris root node with 3 equal classes (50 each):
```
Gini(root) = 1 - (1/3)² - (1/3)² - (1/3)² = 0.6667
```

---

### 5. Gini vs Entropy

Entropy (used in ID3 / C4.5):
```
Entropy(t) = - Σ pᵢ × log₂(pᵢ)
```

| Criterion | Speed  | Sensitivity | Default in sklearn |
|-----------|--------|-------------|-------------------|
| Gini      | Faster | Moderate    | ✅ Yes            |
| Entropy   | Slower | Higher      | ❌ No             |

---

## 📦 What's Inside

### Step 1 — Manual Gini Calculation
```python
def gini_index(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

def gini_split(y_left, y_right):
    n = len(y_left) + len(y_right)
    return (len(y_left) / n) * gini_index(y_left) + \
           (len(y_right) / n) * gini_index(y_right)
```

Output:
```
Gini Node  : 0.4800
Gini Left  : 0.3200
Gini Right : 0.4800
Gini Split : 0.4000
Gini Gain  : 0.0800
```

---

### Step 2 — Best Split Search

Scans all thresholds per feature and finds the one with the lowest Gini Split:
```
Best split → Fitur_A  |  Threshold: -0.0001  |  Gini Split: 0.2467
```

---

### Step 3 — Decision Tree Training
```python
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model.fit(X_train, y_train)
```

Result on synthetic data (200 samples):
```
Accuracy : 88.00%

              precision  recall  f1-score
           0     0.96    0.83      0.89
           1     0.79    0.95      0.86
```

---

### Step 4 — Iris Dataset

Best split per feature:
```
sepal length (cm)  →  threshold=5.40  gini=0.4389  gain=0.2278
sepal width (cm)   →  threshold=3.30  gini=0.5397  gain=0.1269
petal length (cm)  →  threshold=1.90  gini=0.3333  gain=0.3333  ✅
petal width (cm)   →  threshold=0.80  gini=0.3333  gain=0.3333  ✅
```

Petal length and petal width produce the lowest Gini Split → selected as root.

---

### Step 5 — Gini vs Entropy on Iris
```python
model_gini    = DecisionTreeClassifier(criterion='gini',    max_depth=4)
model_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=4)
```

| Criterion | Accuracy | Depth | Leaves |
|-----------|----------|-------|--------|
| Gini      | 100%     | 4     | 7      |
| Entropy   | 100%     | 4     | 7      |

---

### Step 6 — Overfitting Analysis (Depth vs Accuracy)
```
depth= 1  train=0.675  test=0.633
depth= 2  train=0.950  test=0.967
depth= 3  train=0.958  test=1.000  ← sweet spot
depth= 4  train=0.975  test=1.000
depth= 6  train=1.000  test=1.000
```

`max_depth=3` gives full test accuracy without overfitting.

---

## 📊 Summary

| Concept    | Formula                              | Rule                    |
|------------|--------------------------------------|-------------------------|
| Gini Index | `1 − Σ pᵢ²`                         | Lower = purer node      |
| Gini Split | `Σ (nₖ/n) × Gini(k)`               | Pick the smallest       |
| Gini Gain  | `Gini(parent) − Gini_split`         | Pick the largest        |
| Entropy    | `−Σ pᵢ × log₂(pᵢ)`                 | Alternative to Gini     |

---
---

## ▶️ Run
```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook Gini.ipynb
```
