# Predicting Late Delivery Risk in Supply Chain Logistics

A machine learning project that predicts whether an order will be delivered **late** or **on time** using only the information available at the moment the order is placed — no post-shipment data, no leakage.

Built as the term project for **MIE 622: Predictive Analytics and Statistical Learning** (Spring 2026).

---

## Overview

Late deliveries quietly drain margins in supply chain operations — they trigger penalties, inflate expediting costs, and erode customer trust. Most organizations sit on rich operational data but use it reactively, for after-the-fact reporting, rather than proactively to flag at-risk orders before they ship.

This project asks a simple question: **using only order-time information (payment type, shipping mode, scheduled lead time, product, customer segment, geography, order economics), can we reliably predict which orders are going to be late?**

Three classification models — Logistic Regression, Random Forest, and XGBoost — are trained, tuned, and compared on the [DataCo Smart Supply Chain dataset](https://www.kaggle.com/datasets/evilspirit05/datasupplychain) (180,519 records, 53 columns).

---

## Dataset

- **Source:** DataCo Smart Supply Chain Dataset (Kaggle)
- **Size:** 180,519 records × 53 columns → 149,958 records after removing 30,561 duplicates
- **Target:** `Late_delivery_risk` (binary: 1 = late, 0 = on time)
- **Class balance:** ~56.7% late vs. 43.3% on time

From the original 53 columns, **12 features** were retained for modeling. Post-shipment fields like `Days for shipping (real)` and `Delivery Status` were deliberately dropped to prevent data leakage. High-cardinality identifiers (customer ID, email, order ID, etc.) were also excluded.

| #  | Feature                         | Notes                                              |
|----|---------------------------------|----------------------------------------------------|
| 1  | Type                            | Payment type (DEBIT, CASH, TRANSFER, PAYMENT)      |
| 2  | Days for shipment (scheduled)   | Scheduled lead time                                |
| 3  | Shipping Mode                   | Standard, First, Second, Same Day                  |
| 4  | Category Name                   | Top 10 categories retained, rest grouped as 'Other'|
| 5  | Department Name                 | Product department                                 |
| 6  | Customer Segment                | Consumer, Corporate, Home Office                   |
| 7  | Customer Country                | Customer's country                                 |
| 8  | Market                          | Africa, Europe, LATAM, Pacific Asia, USCA          |
| 9  | Order Region                    | 23 distinct regions                                |
| 10 | Order Item Quantity             | Order quantity                                     |
| 11 | Order Item Discount Rate        | Applied discount                                   |
| 12 | Product Price                   | Log-transformed during preprocessing               |

---

## Methodology

The pipeline runs end-to-end in a single Jupyter notebook:

1. **Feature selection** — 53 → 12 leak-free features
2. **Sanity checks** — duplicate removal (30,561 rows), null checks, dtype verification
3. **Exploratory Data Analysis** — distributions, boxplots, bivariate analysis against the target
4. **Feature engineering** — top-10 category bucketing, country standardization, log transform of skewed price
5. **Encoding & scaling** — one-hot encoding (`drop_first=True`) for categoricals, `StandardScaler` for numericals (fit on train only)
6. **Train/test split** — stratified 70/30 split, `random_state=42`
7. **Class balancing** — SMOTE applied **only to the training set** (test set keeps its real-world distribution)
8. **Hyperparameter tuning** — `GridSearchCV` for Random Forest, `RandomizedSearchCV` for XGBoost (5-fold CV)
9. **Evaluation** — accuracy, precision, recall, F1 on both train and test sets; confusion matrices; feature importance

---

## Results

| Model               | Accuracy   | Precision  | Recall   | F1 Score   |
|---------------------|------------|------------|----------|------------|
| Logistic Regression | 68.48%     | **83.88%** | 54.95%   | 66.40%     |
| **Random Forest**   | **68.55%** | 82.72%     | 56.26%   | 66.97%     |
| XGBoost             | 68.30%     | 81.75%     | **56.75%** | **66.99%** |

**Random Forest** came out on top by a hair on accuracy and was the strongest overall, though all three models clustered tightly in the 68–69% accuracy range. Logistic Regression took the precision crown; XGBoost edged out the others on recall and F1.

### Key findings

- **Shipping mode is by far the dominant predictor.** First Class orders are late **94.88%** of the time vs. **40.02%** for Standard Class — a counter-intuitive result driven by the fact that faster modes have tighter promised lead times and no slack to absorb upstream variance.
- **Scheduled shipping days** and **order region** are the next most important drivers.
- **Market region** barely matters (late rates only range 55–57% across all five markets).
- All three models are **high-precision, moderate-recall** — when they flag an order as late they're usually right, but they miss ~44% of actual late deliveries. Threshold tuning is the most obvious lever for improvement.

---

## Tech Stack

- **Python 3**
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — preprocessing, Logistic Regression, Random Forest, cross-validation
- `xgboost` — gradient boosting
- `imbalanced-learn` — SMOTE
- `kagglehub` — dataset download

---

## Repository Structure

```
.
├── A_Machine_Learning_Approach_to_Predicting_Late_Delivery_Risk_in_Supply_Chain_Logistics.ipynb
├── Term_Project_Final_Report.docx
├── Predictive_Analytics_Final_PPT.pptx
└── README.md
```

---

## How to Run

```bash
# Clone the repo
git clone <your-repo-url>
cd <your-repo-name>

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn kagglehub

# Launch the notebook
jupyter notebook A_Machine_Learning_Approach_to_Predicting_Late_Delivery_Risk_in_Supply_Chain_Logistics.ipynb
```

The notebook downloads the dataset automatically via `kagglehub` — you'll need a Kaggle account configured locally.

---

## Future Work

- **Threshold tuning** — use precision-recall curves to pick a threshold that maximizes F1 or hits a target recall (~75%) instead of defaulting to 0.5
- **Drop SMOTE** — the original 57/43 split was barely imbalanced, and forcing 50/50 may have hurt calibration
- **Cost-sensitive tuning** — optimize against a custom loss that penalizes missed late deliveries more than false alarms
- **Safe temporal features** — day of week, month, quarter, holiday flags to capture seasonality
- **Native categorical handling** — CatBoost or LightGBM to avoid the 12 → 59 column explosion from one-hot encoding
- **Lag features** — recent late rate by region/carrier to capture correlated patterns

---

## Author

**Shreyas Basavaraju Ravindra**

Term project for MIE 622: Predictive Analytics and Statistical Learning, Spring 2026.
Instructor: Michael Prokle, PhD.

---

## References

- Constante, F., Silva, F., & Pereira, A. (2019). *DataCo Smart Supply Chain for Big Data Analysis*. Mendeley Data / [Kaggle](https://www.kaggle.com/datasets/evilspirit05/datasupplychain).
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*, 785–794. https://doi.org/10.1145/2939672.2939785
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR*, 16, 321–357. https://doi.org/10.1613/jair.953
