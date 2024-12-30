# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 00:03:34 2024

@author: Jonathan Gonzalez

Machine Learning Regression Masterclass in Python 
By: Dr. Ryan Ahmed 
Platform: Udemy
Type: Compilation of videos

A program that performs linear regression to predict Miles Per Gallon (MPG) 
based on a vehicle's horsepower. The regression model is developed using 
Pandas for data manipulation, Scikit-Learn for model building, and Keras 
for deep learning integration.

Last Updated: 12/28/2024
"""

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

FuelEconomyData = pd.read_csv("FuelEconomy.csv")

# Prints last 5 data points
print(FuelEconomyData.tail(5))

# Describes the data that is going to be used
print("\n")
print(FuelEconomyData.describe())

# Plotting to visualize data
plt.close('all')
sns.jointplot(x = "Horse Power", y = "Fuel Economy (MPG)", data = FuelEconomyData)
sns.pairplot(FuelEconomyData)
sns.lmplot(x = "Horse Power", y = "Fuel Economy (MPG)",data = FuelEconomyData)

x = FuelEconomyData[["Horse Power"]]
y = FuelEconomyData["Fuel Economy (MPG)"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state=42)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression(fit_intercept=True)

regressor.fit(x_train, y_train)

m = regressor.coef_

b = regressor.intercept_

y_predict = regressor.predict(x_test)

R2 = 1- np.sum((y_test-y_predict)**2)/np.sum((y_test - np.mean(y_test))**2)
plt.figure(figsize=(10,10))
plt.title("HP vs. MPG (Training dataset)")
plt.plot(x_train,y_train, 'o', color = 'gray', label = 'Data Points')
plt.plot(x_train,regressor.predict(x_train),'-', color = 'red', label = 'Model, R2 = 0.92')
plt.xlabel("Horse Power (HP)")
plt.ylabel("MPG")
plt.legend()
plt.grid()

plt.figure(figsize=(10,10))
plt.title("HP vs. MPG (Testing dataset)")
plt.plot(x_test,y_test, 'o', color = 'gray', label = 'Data Points')
plt.plot(x_test,y_predict,'-', color = 'red', label = 'Model, R2 = 0.92')
plt.xlabel("Horse Power (HP)")
plt.ylabel("MPG")
plt.legend()
plt.grid()


