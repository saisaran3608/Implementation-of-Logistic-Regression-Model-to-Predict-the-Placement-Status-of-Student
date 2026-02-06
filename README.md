# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Data Collection and Preprocessing Load the placement dataset and remove unnecessary columns. Check for missing and duplicate values, and convert all categorical variables into numerical form using Label Encoding.

Step 2: Feature Selection and Data Splitting Separate the dataset into independent variables (features) and the dependent variable (placement status). Split the data into training and testing sets.

Step 3: Model Training Apply the Logistic Regression algorithm on the training data to build the prediction model.

Step 4: Prediction and Evaluation Use the trained model to predict placement status on test data and evaluate the performance using accuracy score, confusion matrix, and classification report.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SRI SAI SARAN G
RegisterNumber:  212225220103
*/

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:/Users/91908/Downloads/Placement_Data.csv")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])
datal

x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
y_pred

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)


classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)
print("Prediction of LR:")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
<img width="485" height="43" alt="545021294-0686f60f-ce8a-435f-a815-897105ed2676" src="https://github.com/user-attachments/assets/47db8f7a-7095-454b-9d31-c5e5856dc8a9" />

<img width="949" height="184" alt="545021402-31db91d7-3c5d-402a-9ff6-b9e7c4c806a2" src="https://github.com/user-attachments/assets/623fcaf1-6ad3-4963-a3c0-fd251d474be9" />

<img width="410" height="47" alt="545021867-30705170-b28d-41e2-8715-1c4f56b26ddb" src="https://github.com/user-attachments/assets/c3a0cc1f-cc08-473d-bef8-7be15da61903" />

<img width="825" height="165" alt="545021981-a2f92597-08fa-4e9f-a253-fe9e39a63a85" src="https://github.com/user-attachments/assets/1bb34f3a-d729-4d95-bb82-c2ee3e9cf12b" />

<img width="479" height="178" alt="545022568-13ac366d-4031-410d-9f96-5b9018c0bb52" src="https://github.com/user-attachments/assets/aae2e46f-ed5c-408b-a292-5f771a3c1452" />

<img width="785" height="379" alt="545022750-2e4d23a1-693e-4f02-9749-63d6e8ddac72" src="https://github.com/user-attachments/assets/b1d7f755-59df-4813-9c2e-f8ef6b43b824" />

<img width="699" height="132" alt="545022888-1dce5daf-820a-4eb5-9bba-b2e3abe1730a" src="https://github.com/user-attachments/assets/4e06310a-8519-484a-b30e-9f8198ad7ef5" />

<img width="320" height="66" alt="545023925-ca8cabdf-a430-47ac-b622-9e25fab18357" src="https://github.com/user-attachments/assets/a773d3d9-47da-47bd-b0df-b3692307b309" />

<img width="331" height="94" alt="545024180-03b4d338-fb1f-44dd-8b97-954c06f15926" src="https://github.com/user-attachments/assets/2a79668e-15a1-4acd-95a7-21eafd45b066" />

<img width="720" height="268" alt="545024376-8a350fce-cad9-458e-92cf-d4e675e9815c" src="https://github.com/user-attachments/assets/64b7181d-922c-4e8a-b05b-2ffd7aa3d36f" />

<img width="753" height="210" alt="545024464-fe4e2d51-00c2-4304-911a-b0aedbc4296d" src="https://github.com/user-attachments/assets/084c98ba-3805-48b3-ab8d-d65b0dd94e89" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
