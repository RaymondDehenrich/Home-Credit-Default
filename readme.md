# Home Credit Default Risk Prediction

This project tackles the **Home Credit Default Risk** Kaggle dataset, where the goal is to predict whether an applicant will default on a loan.  
The dataset is highly relational, consisting of multiple large CSV tables describing client demographics, bureau records, previous applications, and credit card behavior.  


## Highlights
- Designed and implemented a **data integration pipeline** to combine **multiple relational CSV tables** (e.g., `application_train`, `bureau`, `previous_application`, `POS_CASH_balance`, `credit_card_balance`) into a **single feature-rich table**.  
- Engineered features from categorical, numerical, and time-series data.  
- Trained and evaluated a **LightGBM Classifier (LGBMClassifier)** for binary classification.  
- Addressed challenges of **imbalanced data** and **high-cardinality categorical features**.  


## Dataset
- Source: [Kaggle – Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)  
- Tables include:  
  - **application_train/test**: main client application data  
  - **bureau/bureau_balance**: previous loans and credit history  
  - **previous_application**: past loan applications  
  - **POS_CASH_balance / credit_card_balance / installments_payments**: client repayment and spending patterns  


## Results
- **AUC score:** *0.774*  
- LightGBM outperformed baseline logistic regression and random forest models in both speed and accuracy.  
- Model shows significant result indicating possibilities in real world prediction.


## Next Steps
- Experiment with **Transformer-based embeddings** for tabular + sequential data.  
- Apply **SMOTE/oversampling** or focal loss for better handling of imbalance.  
- Explore **stacked ensemble models** (e.g., LGBM + XGBoost + NN).  
