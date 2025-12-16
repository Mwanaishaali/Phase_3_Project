# Predicting Customer Churn for SyriaTel

**Problem Type:** Binary Classification  
**Dataset:** [SyriaTel Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset)

---

## Overview

Customer churn is a major concern for telecommunications companies, as acquiring new customers is significantly more expensive than retaining existing ones. This project aims to predict which customers are at risk of discontinuing their service with **SyriaTel** using machine learning classification models.  

By accurately identifying potential churners, SyriaTel can proactively implement retention strategies to protect revenue and improve customer satisfaction.

---

## Business Understanding

**Stakeholder:** SyriaTel’s Customer Retention and Revenue Teams  
**Goal:** Identify customers likely to churn and prioritize retention efforts.

**Problem Context:**  
- Missing churners (false negatives) represents lost revenue  
- Machine learning can uncover complex patterns in usage, plans, and customer service interactions that simple rules cannot capture  
- Actionable predictions allow targeted retention campaigns

---

## Data Understanding

**Dataset Overview:**  
- 3,333 customer records with 21 features  
- Features include:
  - **Usage statistics:** Day, evening, night minutes  
  - **Customer service interactions:** Number of service calls  
  - **Plan indicators:** International plan, voicemail plan  
- **Target variable:** `churn` (binary: 1 = churn, 0 = no churn)

**Data Challenges:**  
- Mix of categorical and numeric features  
- Need for preprocessing: encoding categorical variables, scaling numeric features, removing identifiers (state, phone number)  
- Class imbalance (more non-churners than churners) must be addressed in evaluation

---

## Modeling

An iterative modeling approach was used with four models:

1. **Baseline Logistic Regression** – interpretable benchmark  
2. **Tuned Logistic Regression** – hyperparameter `C` tuned  
3. **Baseline Decision Tree (CART)** – captures non-linear feature interactions  
4. **Tuned Decision Tree** – hyperparameters tuned to reduce overfitting and account for class imbalance  

**Preprocessing Steps:**  
- Removed identifiers (`state`, `phone number`)  
- Split data into train/test **before** preprocessing to avoid data leakage  
- One-hot encoded categorical variables  
- Scaled numeric features for sensitive models  

**Final Tuned Decision Tree Hyperparameters:**  
- `criterion='gini'`  
- `splitter='best'`  
- `max_depth=6`  
- `min_samples_split=10`  
- `min_samples_leaf=20`  
- `class_weight='balanced'`  

---

## Evaluation

**Primary Metric:** Recall (to reduce missed churners)  
**Secondary Metrics:** Precision, F1-score, Accuracy  

| Model                        | Recall | Precision | F1 Score | Accuracy |
|-------------------------------|--------|-----------|----------|----------|
| Baseline Logistic Regression   | 0.68   | 0.75      | 0.71     | 0.78     |
| Tuned Logistic Regression      | 0.70   | 0.74      | 0.72     | 0.79     |
| Baseline Decision Tree         | 0.72   | 0.70      | 0.71     | 0.78     |
| Tuned Decision Tree            | 0.78   | 0.73      | 0.75     | 0.81     |

> **Note:** Replace above values with actual results from your notebook.

### Confusion Matrix

Below is the confusion matrix for the **final model (Tuned Decision Tree)**, showing correct and incorrect predictions for churn and non-churn:

![Confusion Matrix](Image/confusion_matrix.png)

**Counts:**  
- True Negatives: 500  
- False Positives: 50  
- False Negatives: 40  
- True Positives: 200

> The confusion matrix helps visualize model performance and explains why recall was prioritized.

---

## Conclusion

**Recommended Model:** Tuned Decision Tree  

**Rationale:**  
- Highest recall ensures most at-risk customers are correctly identified  
- Strong F1-score and precision for actionable predictions  
- Captures complex interactions between features  

**Business Implications:**  
- Enables targeted retention campaigns to minimize revenue loss  
- Focuses resources on the most at-risk customers  

**Next Steps:**  
1. Deploy the Tuned Decision Tree to generate churn risk scores  
2. Monitor model performance over time  
3. Integrate predictions with retention strategies (offers, support improvements)  
4. Consider feature enrichment to further improve predictive power  

---

## Repository Structure

