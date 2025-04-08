# Telco-Customer-Churn-Analysis-Model-Insights
“A machine learning project to predict customer churn using logistic regression and other models.”
## Dataset
Telco Customer Churn dataset from IBM (Kaggle).

## Tools Used
- Python
- Pandas, Scikit-learn, Seaborn, Matplotlib
- Power BI (for visualization)
- Jupyter Notebook

import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
## Aim:
This project aims to build a machine learning model that predicts whether a customer is likely to leave, helping the business take proactive steps to retain them.

## Project Overview
We analyzed customer data using logistic regression to identify patterns associated with churn. By identifying key factors driving churn, we aimed to help the business reduce customer attrition.
## Data Summary
The dataset contains 7,043 customer records with 21 features, including demographics, account information, service usage, and customer status.

## Key Variables:

Target Variable: Churn (Yes/No)
Features: Contract, Internet Service, Payment Method, Monthly Charges, tenure, etc.

## The data set includes information about:
Customers who left within the last month — the column is called Churn
Services that each customer has signed up for — phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
Customer account information — how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
Demographic info about customers — gender, age range, and if they have partners and dependents.

## Exploratory Data Analysis(EDA)
![image](https://github.com/user-attachments/assets/f10a440b-21fc-49c8-a76f-e112e3db2fb5)

## Key Insights from the describe() Output:

# Numerical Features:
tenure: Ranges from 0 to 72 months (this is the customer's tenure with the company).
Monthly Charges: Ranges from 18.25 to 118.75. The mean is around 64.76, which might suggest the distribution could be skewed.
TotalCharges: Ranges from NaN (missing values) to ~8684. Perhaps it needs special handling due to missing data or incorrect formats.

# Categorical Features:
gender: 2 unique values (Male, Female).
SeniorCitizen: 2 unique values (0 = No, 1 = Yes).
Churn: 2 unique values (No, Yes), which is the target variable indicating customer churn.
Other categorical columns like Partner, Dependents, PhoneService, InternetService, etc.

## Data Preprocessing
1. Handling Missing Values

TotalCharges has missing values (6531 non-null values, with 7043 expected).
Other columns appear to have no missing values, as all counts are 7043.
![image](https://github.com/user-attachments/assets/c608179e-2ddb-4640-829c-b1cb796c2a31)
All columns now show 0 missing values.

## Visualization of Missing Data

# Visualize missing Data
All columns now show 0 missing values.
![image](https://github.com/user-attachments/assets/13a95580-4d5a-4c7a-8c75-822d0d9819d7)

## Converting Categorical Variables- Label Encoding for Binary Features & one-hot encoding

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
# Encoding binary categorical columns
binary_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']
for col in binary_cols:
 df[col] = label_encoder.fit_transform(df[col])
# Check the result
df.head()

#One-Hot Encoding for Multi-Category Features
df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=True)

# Check the result
df.head()

## Model Building
Splitting the data train/test sets- we are trying out Logistic Regression & Rain Forest.

#Split data into training/test sets
#Label Encoding for Binary Features
from sklearn.preprocessing import LabelEncoder
# Create a label encoder object
label_encoder = LabelEncoder()
# List of binary columns to encode
binary_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']
# Loop through each binary column and encode it
for col in binary_cols:
 df[col] = label_encoder.fit_transform(df[col])
# Check the result 
df.head()

![image](https://github.com/user-attachments/assets/03f8ba2b-2853-4d5e-b945-42d5bb03ed76)
![image](https://github.com/user-attachments/assets/89176c03-885e-410e-9f5e-f61f9529ff9e)
(5634, 22)This represents the training set (X_train) with the following dimensions:
5634: This is the number of samples (rows) in the training set.
22: This is the number of features (columns) in the training set.
So, the training set has 5634 samples, and for each sample, there are 22 features or variables used to predict the target.
(1409, 22)This represents the test set (X_test) with the following dimensions:
1409: This is the number of samples (rows) in the test set.
22: This is the number of features (columns) in the test set.
So, the test set has 1409 samples, and like the training set, each sample has 22 features.
(5634,)This represents the target variable for the training set (y_train).
5634: This is the number of samples in the training set’s target variable, which corresponds to the same number of samples in X_train. The target variable (y_train) is a 1-dimensional array with the churn values (0 or 1) for each customer in the training set.
(1409,)This represents the target variable for the test set (y_test).
1409: This is the number of samples in the test set’s target variable, corresponding to the same number of samples in X_test. The target variable (y_test) is also a 1-dimensional array with the churn values (0 or 1) for each customer in the test set.

## Logistic Regression Model

![image](https://github.com/user-attachments/assets/58b3d8eb-24eb-4241-9a04-171e28c4dd79)
![image](https://github.com/user-attachments/assets/fb74fb50-f5c1-4992-86cd-4906ed0dedfa)

1. Accuracy:

Accuracy: 82.04%
This means the model correctly predicted whether the customer would churn or not 82.04% of the time.
This is a solid result for a first attempt, but we can always improve it with more tuning or different models.

2. Classification Report:

The classification report gives us more detailed insights into the model’s performance for each class (Churn = 1 and Churn = 0).

Class 0 (No Churn):

Precision: 0.86
Of all the times the model predicted a customer would not churn, it was correct 86% of the time.
Recall: 0.91
Of all the actual customers who did not churn, the model correctly identified 91% of them.
F1-Score: 0.88
A good balance between precision and recall for the “no churn” class.
Class 1 (Churn):

Precision: 0.69
Of all the times the model predicted a customer would churn, it was correct 69% of the time.
Recall: 0.58
Of all the actual customers who did churn, the model only identified 58% of them.
F1-Score: 0.63
The F1-score here is lower, indicating the model has room for improvement in predicting churn.

3. Confusion Matrix:

The confusion matrix helps visualize the model’s true positives, false positives, true negatives, and false negatives.

[[938  98]  -> True negatives (938), False positives (98)
 [155 218]  -> False negatives (155), True positives (218)
True Negatives (938): These are customers who did not churn, and the model correctly predicted that.
False Positives (98): These are customers who did not churn, but the model predicted they would churn.
False Negatives (155): These are customers who churned, but the model predicted they would not.
True Positives (218): These are customers who churned, and the model correctly predicted that.
Key Takeaways:

Precision and Recall for Churn (1): The model seems to be more accurate in predicting customers who won’t churn (Class 0). The precision and recall for churn (Class 1) are lower, which suggests the model is struggling to predict customers who are likely to churn.

## Random Forest Model

Steps:

Import the Random Forest Classifier.
Train the model using the training data.
Evaluate the model on the test set.
Check the classification report, accuracy, and confusion matrix
# Import the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display the classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
![image](https://github.com/user-attachments/assets/da56d55a-d69d-45a8-991c-ea171d80be9c)

1. Accuracy:
Accuracy: 79.42%
This means that 79.42% of the time, the model correctly predicted whether a customer would churn or not.

2. Classification Report:

Class 0 (No Churn):

Precision: 0.83
Of all the times the model predicted a customer would not churn, it was correct 83% of the time.
Recall: 0.91
Of all the actual customers who did not churn, the model correctly identified 91% of them.
F1-Score: 0.87
The F1-score here is quite good, indicating that the model is doing well in identifying customers who will not churn.
Class 1 (Churn):

Precision: 0.65
Of all the times the model predicted a customer would churn, it was correct 65% of the time
Recall: 0.47
Of all the actual customers who churned, the model only identified 47% of them.
F1-Score: 0.55
The F1-score for churn (Class 1) is lower than for no churn (Class 0), which suggests that the model is struggling to correctly predict customers who will churn.

3. Confusion Matrix

[[943  93]  -> True negatives (943), False positives (93)
 [197 176]  -> False negatives (197), True positives (176)
True Negatives (943): Customers who did not churn, and the model correctly predicted this.
False Positives (93): Customers who did not churn, but the model predicted they would churn.
False Negatives (197): Customers who churned, but the model predicted they would not.
True Positives (176): Customers who churned, and the model correctly predicted this.
Key Takeaways:

No Churn (Class 0): The model performs well in predicting customers who won’t churn (high precision and recall).
Churn (Class 1): The model struggles to predict customers who will churn, as shown by the lower precision, recall, and F1-score. In particular, the recall for churn (Class 1) is 0.47, which means the model is missing a lot of customers who churn.

# Import Logistic Regression
from sklearn.linear_model import LogisticRegression

# Initialize the Logistic Regression model
log_reg_model = LogisticRegression(random_state=42, max_iter=1000)

# Train the Logistic Regression model on the training data
log_reg_model.fit(X_train, y_train)

# Now, make predictions using Logistic Regression
y_pred_log_reg = log_reg_model.predict(X_test)

# Evaluate the Logistic Regression model
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_log_reg))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log_reg))

# Plot confusion matrix for Logistic Regression
sns.heatmap(confusion_matrix(y_test, y_pred_log_reg), annot=True, fmt="d", cmap="Blues", xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title("Logistic Regression Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()
![image](https://github.com/user-attachments/assets/c4842395-f999-40b0-ac15-567eb9fc26e2)
![image](https://github.com/user-attachments/assets/12824c95-eed9-4cfe-a479-5b8735c948f2)

## Model Summary & Performance

# Logistic Regression Results

Accuracy: 82%
Precision (Churn): 0.69
Recall (Churn): 0.58
F1-score (Churn): 0.63
Random Forest Classifier Results

Accuracy: 79%
Precision (Churn): 0.65
Recall (Churn): 0.47
F1-score (Churn): 0.55
Model Evaluation & Interpretation

Which features contribute most to churn? Analyzing top insights from the model.

# Create a DataFrame for feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': log_reg_model.coef_[0]
})

# Sort by absolute importance
feature_importance['Absolute_Importance'] = feature_importance['Importance'].abs()
feature_importance = feature_importance.sort_values(by='Absolute_Importance', ascending=False)

# Plot top 10 important features
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature', palette='coolwarm')
plt.title('Top 10 Features Influencing Churn (Logistic Regression)')
plt.tight_layout()
plt.show()
![image](https://github.com/user-attachments/assets/5d3ebdf2-02e0-4710-9df5-17d523137f36)
![image](https://github.com/user-attachments/assets/2d58ee54-0365-4f4e-b6c8-ca509ed8f14a)

## 🔑 Top Insights from Model

Month-to-Month Contracts → Customers with flexible, month-to-month contracts churn the most. These customers are more likely to leave at any time without penalty.
Short Tenure = High Churn → Newer customers are much more likely to churn. It may signal dissatisfaction early on.
Fiber Optic Users Churn More → This could indicate issues with fiber service quality, pricing, or customer expectations
Value-Added Services Retain Customers → Customers with Tech Support and Online Security churn less, suggesting that bundled or supportive services increase loyalty.
High Monthly Charges = Risky → Customers paying more per month are more prone to churn, possibly due to perceived poor value.

## Business Recommendations
Introduce loyalty rewards for long-term customers.
Encourage annual or 2-year contracts via discounts.
Offer value-added services like Tech Support as bundles.
Investigate customer complaints related to fiber services.
Monitor high-bill customers and offer personalized discounts.


Here is a link to my documentation on Medium  https://medium.com/@peremobon/telco-customer-churn-analysis-model-insights-845623091d87
