***

# 📝 Complete Logistic Regression Project Documentation


***

# **Logistic Regression: From Scratch to Production**

## Project Overview

This project demonstrates the complete implementation of **Logistic Regression** from scratch using gradient descent, with comprehensive comparisons to industry-standard libraries. The journey includes two datasets to showcase both real-world challenges and model capabilities.

***

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset 1: E-Commerce Customer Retention](#dataset-1-e-commerce-customer-retention)
3. [Dataset 2: Breast Cancer Classification](#dataset-2-breast-cancer-classification)
4. [Key Learnings](#key-learnings)
5. [Technical Implementation](#technical-implementation)
6. [Results Summary](#results-summary)
7. [Conclusions](#conclusions)

***

## Introduction

**Objective**: Build a logistic regression classifier from scratch to deeply understand:

- Sigmoid activation function
- Binary cross-entropy loss
- Gradient descent optimization
- Classification metrics
- Real-world data challenges

**Tools Used**:

- Python, NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn (for comparison)
- SMOTE (for class imbalance)
- PCA (for dimensionality reduction)

***

## Dataset 1: E-Commerce Customer Retention

### Business Problem

Predict whether a customer will return for future purchases based on their transaction and behavioral data.

### Dataset Details

- **Size**: 22,049 transactions
- **Features**: 33 (after encoding)
    - Demographics: Age, Gender, City
    - Behavioral: Session_Duration_Minutes, Pages_Viewed, engagement_score
    - Transaction: Unit_Price, Quantity, Discount_Amount, Payment_Method
    - Post-purchase: Delivery_Time_Days, Customer_Rating
- **Target**: Is_Returning_Customer (Binary: 0/1)
- **Class Distribution**: 81% returning, 19% not returning (IMBALANCED)

***

### Exploratory Data Analysis (EDA)

#### Key Findings:

**1. Distribution Analysis**:

- Age: Normal distribution (18-75 years, mean=35)
- Unit_Price: EXTREME right skew (outliers up to \$7,900 - luxury items)
- Discount_Amount: 50% orders have NO discount (spike at zero)
- Customer_Rating: Left skew (most ratings 4-5 stars)

**2. Correlation Analysis**:

- **ALL features showed WEAK correlation with target** (< 0.07)
- Strongest: Quantity (0.069) - still very weak!
- Multicollinearity detected: Unit_Price ↔ Total_Amount (0.85) → Dropped Total_Amount

**3. Outlier Detection**:

- Unit_Price: 570 Z-score outliers
- Discount_Amount: 409 Z-score outliers
- **Domain decision**: KEPT outliers (real business patterns - bulk orders, luxury items, sales)

***

### Data Preprocessing Pipeline

```python
# 1. Drop unnecessary columns
df = df.drop(columns=["Order_ID", "Customer_ID", "Total_Amount"])

# 2. Feature Engineering
df['engagement_score'] = df['Session_Duration_Minutes'] * df['Pages_Viewed']
df['fast_delivery'] = (df['Delivery_Time_Days'] <= 3).astype(int)
df['day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek
df['month'] = pd.to_datetime(df['Date']).dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# 3. Encoding
- Binary encoding: Gender (Male=1, Female=0)
- One-hot encoding: City, Product_Category, Payment_Method, Device_Type (drop_first=True)

# 4. Variance Filtering
threshold = 0.05 (removed low-variance features)

# 5. Multicollinearity Removal
correlation > 0.8 → dropped

# 6. Train-Test Split (Stratified 80-20)

# 7. Scaling: Log-transform + Z-score normalization
logged = np.log1p(df)
scaled = (logged - mean) / std

# 8. SMOTE (balance classes to 50-50)
```


***

### PCA Analysis

```
Original features: 33
PCA components: 26 (retaining 95% variance)
Top 5 components explain: [6.89%, 6.19%, 5.10%, 4.92%, 4.58%]
```

**PCA Visualization Result**:

- Complete overlap between returning and non-returning customers
- No clear cluster separation
- **Conclusion**: Features do not linearly separate classes

***

### Model Results - E-Commerce Dataset

#### Manual Logistic Regression:

```
Epoch 900: Loss=0.6756, Accuracy=0.5813
```


#### Sklearn Logistic Regression:

```
Accuracy: 57.41%
```


#### Random Forest (Non-linear model):

```
              precision    recall  f1-score   support
       0       0.30      0.03      0.06       804
       1       0.82      0.98      0.89      3606
    accuracy                           0.81      4410
ROC AUC: 0.60
```


***

### Key Finding: Feature Limitations

**Despite applying industry-standard techniques:**

- Log-transformation and scaling
- SMOTE for class imbalance
- PCA for dimensionality reduction
- Multiple algorithms (Logistic Regression, Random Forest, XGBoost)

**Result**: Models achieved only ~58% accuracy (barely better than random guessing)

**Analysis**:
The dataset's transactional features showed **weak separation** between returning and non-returning customers:

- PCA visualization revealed complete class overlap
- All models resorted to majority-class prediction (81% baseline)
- Recall for minority class: only 3-5%

**Business Interpretation**:
Customer return behavior is likely driven by factors **NOT present** in transactional data:

- Product quality and satisfaction
- Customer service interactions
- Competitor offerings
- Personal circumstances
- Brand perception

**Recommendation**:
To improve predictions, collect additional features:

- Product review text (sentiment analysis)
- Customer service tickets
- Email engagement rates
- Social media sentiment
- Customer lifetime value (CLV)

***

## Dataset 2: Breast Cancer Classification (Redemption!)

### Why This Dataset?

To **validate that our manual implementation works correctly**, we tested on a well-established dataset with strong feature-target relationships.

### Dataset Details

- **Source**: Sklearn's breast cancer dataset
- **Size**: 569 samples
- **Features**: 30 (tumor measurements)
- **Target**: Malignant (0) vs Benign (1)
- **Class Distribution**: Balanced

***

### Preprocessing

```python
# 1. Train-test split (80-20)
# 2. StandardScaler (Z-score normalization)
```


***

### Results - Breast Cancer Dataset

#### Manual Logistic Regression:

```
Epoch 900: Loss=0.1107, Accuracy=98.24%

Test Set Performance:
Accuracy:  98.25%
Precision: 98.60%
Recall:    98.60%
F1-score:  98.60%
```


#### Sklearn Logistic Regression:

```
Accuracy:  97.37%
Precision: 97.22%
Recall:    98.60%
F1-score:  97.90%
```


***

### Model Comparison

| Metric | Manual Model | Sklearn Model | Difference |
| :-- | :-- | :-- | :-- |
| **Accuracy** | **98.25%** | 97.37% | +0.88% |
| **Precision** | **98.60%** | 97.22% | +1.37% |
| **Recall** | 98.60% | 98.60% | 0.00% |
| **F1-Score** | **98.60%** | 97.90% | +0.69% |

**Manual model OUTPERFORMED sklearn!** 🔥

***

## Key Learnings

### 1. **Not All Problems Are Solvable with Available Data**

- E-commerce dataset taught us that feature quality matters MORE than model complexity
- Real-world data often lacks the signal needed for accurate predictions
- Knowing when to collect more data is a crucial skill


### 2. **Manual Implementation Validates Understanding**

- Building from scratch revealed the math behind the "black box"
- Manual model matched and even exceeded sklearn performance on breast cancer data
- Proved our implementation is correct


### 3. **Class Imbalance Handling**

- SMOTE helped balance training data
- But didn't fix fundamental feature weakness in e-commerce dataset
- Showed that balancing techniques can't create signal that doesn't exist


### 4. **Dimensionality Reduction Insights**

- PCA visualization clearly showed feature limitations
- When classes don't separate in 2D PCA, linear models will struggle
- Useful diagnostic tool, not a magic fix

***

## Technical Implementation

### Manual Logistic Regression Class

```python
class LogisticRegressionManual:
    def __init__(self, learning_rate=0.01, epoch=1000, random_state=42):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.random_state = random_state
        self.cost_history = []
        self.weight = None
        self.bias = None

    def sigmoid(self, z):
        return 1/(1+np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).ravel()
        
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for i in range(self.epoch):
            # Forward pass
            z = X @ self.weight + self.bias
            y_pred = self.sigmoid(z)
            
            # Binary cross-entropy loss
            eps = 1e-15
            y_pred_clip = np.clip(y_pred, eps, 1 - eps)
            cost = -(y * np.log(y_pred_clip) + (1-y) * np.log(1-y_pred_clip))
            self.cost_history.append(np.mean(cost))

            # Gradients
            error = y_pred - y
            dw = (1/n_samples) * (X.T @ error)
            db = (1/n_samples) * (np.sum(error))

            # Update weights
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        X = np.array(X)
        z = X @ self.weight + self.bias
        y_pred = self.sigmoid(z)
        return (y_pred >= 0.5).astype(int)
```

**Key Components**:

- **Sigmoid**: Maps linear output to probability[^1]
- **Binary Cross-Entropy**: Penalizes wrong probability predictions
- **Gradient Descent**: Iteratively optimizes weights using derivatives
- **Threshold**: 0.5 for binary classification

***

## Results Summary

### E-Commerce Dataset

- **Challenge**: Weak feature-target relationship
- **Manual Model**: 58.13% accuracy
- **Sklearn Model**: 57.41% accuracy
- **Learning**: Feature engineering and additional data needed


### Breast Cancer Dataset

- **Success**: Strong feature-target relationship
- **Manual Model**: **98.25% accuracy** (BETTER than sklearn!)
- **Sklearn Model**: 97.37% accuracy
- **Learning**: Manual implementation is production-ready

***

## Conclusions

### What Worked:

✅ Manual logistic regression implementation is **correct and competitive**
✅ Comprehensive preprocessing pipeline (log-transform, SMOTE, PCA)
✅ Proper handling of imbalanced data
✅ Thorough EDA revealed dataset limitations early
✅ Domain knowledge applied to feature engineering

### What Didn't Work:

❌ E-commerce transactional features too weak for return prediction
❌ PCA and SMOTE couldn't compensate for missing signal
❌ Complex models (Random Forest, XGBoost) also failed → confirms feature issue

### Professional Takeaway:

> "The best model in the world can't extract signal that doesn't exist in the data. Knowing when to go back and collect better features is as important as building better models."

***

## Future Work

### For E-Commerce Dataset:

1. Collect customer service interaction data
2. Add product review sentiment scores
3. Track email engagement metrics
4. Include competitor pricing data
5. Consider time-series analysis (purchase patterns over time)

### For Model:

1. Implement mini-batch gradient descent
2. Add L1/L2 regularization
3. Implement learning rate scheduling
4. Add early stopping mechanism

***

## Repository Structure

```
Logistic-Regression-Project/
├── data/
│   ├── raw/
│   │   └── ecommerce_data.csv
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── notebooks/
│   ├── 01_EDA_Ecommerce.ipynb
│   ├── 02_Preprocessing.ipynb
│  
├── src/
│   └── LogisticRegressionManual.py
├
├── README.md

```


***

## How to Run

```bash
# Clone repository
git clone https://github.com/Harimhs/Logistic-Regression-Project.git
cd Logistic-Regression-Project

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook
```

***

## Requirements

```
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
imbalanced-learn==0.11.0
```

***

## Author

**Hariharasudhan M**
Data Science Enthusiast | Machine Learning Engineer
[GitHub](https://github.com/Harimhs) | [LinkedIn](#) | [Kaggle](https://www.kaggle.com/harimhs)

***

## License

MIT License

***


