# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required library and read the dataframe
   
2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the cost function using Gradient Descent and generate the required graph.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: 
RegisterNumber:  
*/


import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate=0.01, num_iters=1000):
    
 X=np.c_[np.ones(len(X1)), X1]

theta=np.zeros (X.shape[1]).reshape(-1,1)
    
for _ in range(num_iters):
         
 predictions=(X).dot(theta).reshape(-1,1)
         
 errors=(predictions-y).reshape(-1,1)
         
theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
return theta

data=pd.read_csv("50_Startups.csv" ,header = None) 

X=(data.iloc[1:, :-2].values)

X1=X.astype(float)

scaler=StandardScaler()

y=(data.iloc[1:,-1].values).reshape(-1,1)

X1_Scaled=scaler.fit_transform(X1)

Y1_Scaled=scaler.fit_transform(y)

theta=linear_regression(X1_Scaled, Y1_Scaled)

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)

new_Scaled=scaler.fit_transform(new_data)


prediction=np.dot (np.append(1, new_Scaled), theta)

prediction=prediction.reshape(-1,1)

pre=scaler.inverse_transform(prediction)

print (data.head)

print(f"Predicted value: {pre}")
```

## Output:
<img width="698" height="777" alt="image" src="https://github.com/user-attachments/assets/8c0d1098-3d5e-4946-a31a-b25a8af6c156" />

<img width="628" height="227" alt="image" src="https://github.com/user-attachments/assets/5bb98529-05e0-4a8a-8e8f-45b68e5fcbd2" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
