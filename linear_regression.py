
# coding: utf-8

# In[3]:


import os
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv('Beton.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

# Spliting the dataset into the tranning set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =1/3,random_state=0)


#Fitting sample linear Regression to Tranning set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor=regressor.fit(X_train, y_train)


#Predicting the Test Result

y_pred=regressor.predict(X_test)

#Virtuallizing the tranning set result
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary VS Experience(Tranning set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print(y_pred)


print(plt.show())




