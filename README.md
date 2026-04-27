# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load dataset and clean column names
2.Select input features (Size, Bedrooms) and targets (Price, Occupants)
3.Scale input features using StandardScaler
4.Initialize two SGD regression models
5.Train models using scaled data
6.Take user input, scale it, and predict outputs

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: MOHAMMED GHUFRAN P 
RegisterNumber: 212225230178 
*/
#SGD-ex4
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("house.csv")
#print(data.columns)
data.columns = data.columns.str.strip()
# Features (inputs)
X = data[['Size', 'Bedrooms']]

# Targets (outputs)
y_price = data['Price']
y_occ = data['Occupants']

# Scaling (important for SGD)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Models
price_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
occ_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)

# Train models
price_model.fit(X_scaled, y_price)
occ_model.fit(X_scaled, y_occ)

# Input
size = float(input("Enter house size: "))
bed = int(input("Enter number of bedrooms: "))

# Scale input
new_data = scaler.transform([[size, bed]])

# Prediction
pred_price = price_model.predict(new_data)
pred_occ = occ_model.predict(new_data)

print("Predicted Price:", pred_price[0])
print("Predicted Occupants:", round(pred_occ[0]))
```
## Output:
<img width="1320" height="130" alt="image" src="https://github.com/user-attachments/assets/69520db9-bc3c-4336-a169-c9054a164178" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
