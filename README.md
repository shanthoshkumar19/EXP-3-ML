# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and load the dataset.
2. Remove unnecessary columns and convert categorical data using dummy variables.
3. Separate the dataset into features (X) and target variable (Y).
4. Split the data into training and testing sets.
5. Create and train the Linear Regression model using training data.
6. Evaluate the model using 5-fold cross-validation and calculate average R² score.
7. Predict test data values and compute MSE, MAE, and R²; plot actual vs predicted prices.
## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: SHANTHOSH KUMAR R
RegisterNumber:  212225040402
*/



import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,mean_absolute_error
import matplotlib.pyplot as plt
data=pd.read_csv('CarPrice_Assignment.csv')
#simple processing
data=data.drop(['car_ID','CarName'],axis=1)#removes unnecessary columns
data = pd.get_dummies(data, drop_first=True)# Handle categorical variables
# Split data
X = data.drop('price', axis=1)
Y = data['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# 3.Create and train model
model = LinearRegression()
model.fit(X_train, Y_train)
# 4.Evaluate with cross-validation (simple version)
print("Name: SHANTHOSH KUMAR R")
print("Reg. No: 212225040402")
print("\n=== Cross-validation ===")
cv_scores=cross_val_score(model,X,Y,cv=5)
print("Fold R2 scores:",[f"{score:.4f}" for score in cv_scores])
print(f"Average R2: {cv_scores.mean():.4f}")
# 5. Test set evaluation
Y_pred = model.predict(X_test)
print("\n=== Test Set Performance ===")
print(f"MSE: {mean_squared_error(Y_test, Y_pred):.2f}")
print(f"MAE: {mean_absolute_error(Y_test, Y_pred):.2f}")
print(f"R²: {r2_score(Y_test, Y_pred):.4f}")
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, Y_pred, alpha=0.6)
plt.plot([Y_test.min(), Y_test.max()],[Y_test.min(), Y_test.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)
plt.show()
```

## Output:
<img width="732" height="255" alt="image" src="https://github.com/user-attachments/assets/9b63925c-b7fb-43b2-823a-2381c7e33540" />
<img width="1138" height="666" alt="image" src="https://github.com/user-attachments/assets/559fc5b8-263d-4a26-91bb-0e8cb878e337" />



## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
