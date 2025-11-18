# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```
## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: K DHANUSRI POOJA
RegisterNumber:  24011393

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("student_scores.csv")

print(df.tail())
print(df.head())
df.info()

x = df.iloc[:, :-1].values  # Hours
y = df.iloc[:,:-1].values   # Scores

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

print("X_Training:", x_train)
print("X_Test:", x_test)
print("Y_Training:", y_train)
print("Y_Test:", y_test)

reg = LinearRegression()
reg.fit(x_train, y_train)

Y_pred = reg.predict(x_test)

print("Predicted Scores:", Y_pred)
print("Actual Scores:", y_test)

a = Y_pred - y_test
print("Difference (Predicted - Actual):", a)

plt.scatter(x_train, y_train, color="green")
plt.plot(x_train, reg.predict(x_train), color="red")
plt.title('Training set (Hours vs Scores)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test, y_test, color="blue")
plt.plot(x_test, reg.predict(x_test), color="green")
plt.title('Testing set (Hours vs Scores)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mae = mean_absolute_error(y_test, Y_pred)
mse = mean_squared_error(y_test, Y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
```

## Output:

HEAD:

<img width="270" height="225" alt="359573548-9de86369-938b-4030-a048-808337d517d4" src="https://github.com/user-attachments/assets/476fc479-d6e8-47cc-9dea-e5347c100495" />

TAIL:

<img width="310" height="219" alt="359573769-6a1861ee-dac9-408d-9f7a-faef16902dbd" src="https://github.com/user-attachments/assets/e51ef2ca-70a8-441d-ac3e-d160afe4d953" />

TRAINING:

<img width="250" height="384" alt="359570077-5b02582c-73c3-40e7-bca7-b53d44566ded" src="https://github.com/user-attachments/assets/695e3823-3cd4-4ad8-9515-6c3cebc397b0" />

<img width="271" height="380" alt="359570509-3b8a3dc9-a4c3-4c62-879e-fe9f1420eba7" src="https://github.com/user-attachments/assets/90828a82-5ed0-421d-a64b-e9ab3dc0cbb5" />

TEST:

<img width="337" height="235" alt="359571117-5b112037-841a-48c7-a45f-8529261351df" src="https://github.com/user-attachments/assets/119d3610-0264-485b-a622-7b72babaea43" />

<img width="271" height="380" alt="359570509-3b8a3dc9-a4c3-4c62-879e-fe9f1420eba7" src="https://github.com/user-attachments/assets/ada19ac2-c9d2-4598-929e-edaa344954bd" />

TRAINING SET:

<img width="743" height="570" alt="359572132-c550b4a3-7bc4-4365-ad61-03aa1fd3e53a" src="https://github.com/user-attachments/assets/300aeda4-e652-4475-a83c-3a8bd24989f2" />

TEST SET:

<img width="725" height="566" alt="359572417-68d8ecf9-f140-4677-a7f9-9b6609da5987" src="https://github.com/user-attachments/assets/d89f2d2f-1abb-4fee-a6c2-90b6f169b2bd" />

<img width="456" height="82" alt="359572824-2ecaced7-c4b7-448a-ab1a-126978202f4f" src="https://github.com/user-attachments/assets/796c3f95-842e-41a1-9236-d9323585cc9c" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
