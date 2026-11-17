Customer Churn Prediction – Final Report

1. Problem Statement

The objective is to predict whether a telecom customer is likely to leave the company. Early identification helps the business take retention actions.

2. Dataset

The Telco Customer Churn dataset contains 7,043 customer records and 21 original columns. It includes customer demographics, services, contract details, charges, and churn status.

3. Methodology

- Performed data cleaning and exploratory data analysis.
- Created features such as tenure group, average monthly spend, service count, and long-term contract.
- Compared Logistic Regression, Random Forest, and XGBoost.
- Tuned Logistic Regression and XGBoost using RandomizedSearchCV.
- Selected XGBoost and handled class imbalance using "scale_pos_weight".
- Selected a classification threshold of 0.59 to target approximately 75% recall.

4. Final Test Results

The selected XGBoost model was evaluated once on the untouched test dataset.

- ROC-AUC: 0.8469
- F1 Score: 0.6281
- Churn Recall: 75.4%

Confusion Matrix:

| Predicted No| Predicted Yes
Actual No| 793| 242
Actual Yes| 92| 282

5. Limitations

- The model is trained on historical telecom customer data.
- Customer behaviour and service conditions may change over time.
- Some customers may be incorrectly identified as likely to churn.
- Feature importance indicates predictive contribution, not causation.

6. Business Recommendations

- Focus on month-to-month contract customers.
- Offer suitable long-term contract plans.
- Improve online security and technical support services.
- Provide targeted retention offers to customers with high predicted churn probabilities.

Conclusion

The XGBoost model achieved a test ROC-AUC of 0.8469 and identified approximately 75% of actual churners. It can support proactive customer retention strategies.