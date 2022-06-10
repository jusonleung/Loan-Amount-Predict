# -*- coding: utf-8 -*-
"""COMP4434 Big Data Analytics - Individual Project

#20037676D Leung Chun Kit

#Introduction
AlphaMoney is a platform for issuing loans. For people who want to borrow money from the AlphaMoney platform, the platform requires them to provide relevant personal information to decide the Loan Amount. The platform has collected personal information from users, including Gender, Age, Credit Score, Property Age, Profession, Loan Amount Request, Property Price and Income Stability, etc., and determined the Loan Amount based on them.


In this project, I will use two model (Linear Regression and DNN) to predict the Loan amount of the providing dataset.

# Data preprocessing/analytics

##Reading Data set
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

"""Drop the rows with 4 or more mssing values"""

train_df.dropna(subset=['Loan Amount'], inplace=True)
train_df.dropna(thresh=len(train_df.columns)-4, inplace=True)
train_df['Customer ID'].drop_duplicates(inplace=True)

"""Merge the traning and testing data set"""

df = pd.concat([train_df, test_df])

"""Drop the row with duplicate Customer ID"""

df['Customer ID'].drop_duplicates(inplace=True)

"""##Function of creating dummy variable"""

def creat_dummies(df, column_name):
  # get the dummies and store it in a variable
  dummies = pd.get_dummies(df[column_name])
  # Concatenate the dummies to original dataframe
  merged = pd.concat([df, dummies], axis=1)
  # drop the values
  merged.drop([column_name], axis=1, inplace=True)
  return merged

"""##Column - Name
Customer name is clearly not related to the loan amount, so drop this column
"""

df.drop("Name", axis=1, inplace=True)

"""##Column - Gender
There are 51 missing values
"""

df['Gender'].value_counts(dropna=False)

"""The gender distribution is nearly evenly distributed, so fill the missing value from randomly choose from other values.


First create a funtion of filling missing data by random choosing from non missing values in column.
"""

def randomiseMissingData(df2, column_name):
    "randomise missing data for DataFrame (within a column)"
    df = df2.copy()
    data = df[column_name]
    mask = data.isnull()
    samples = random.choices( data[~mask].values , k = mask.sum() )
    data[mask] = samples
    return df

"""Filling missing data by random choosing from non missing values"""

df = randomiseMissingData(df, 'Gender')

df['Gender'].value_counts(dropna=False)

"""Replace with dummy variable columns"""

df = creat_dummies(df,'Gender')

"""##Column - Age
No missing value in "Age", also no extreme value (too young or too old) in both data set.
"""

df['Age'].value_counts(dropna=False)

"""##Column - Income (USD)
Rename this column with "Income"

There are no negative value in this column
"""

df.rename(columns={'Income (USD)':'Income'}, inplace=True)

df.Income.isna().sum(), df['Income'][df['Income']<0]

"""Graph of Income againist Loan Amount"""

plt.scatter(df['Income'], df['Loan Amount'])
plt.xlabel('Income')
plt.ylabel('Loan Amount')
plt.show()

"""From the graph, There is a value with extremely high value.

This row the in training dataset, so frop this row
"""

print(df.loc[df['Income']==df['Income'].max(), ['Income', 'Loan Amount']])
df.drop(df['Income'].idxmax(), inplace=True)

"""Graph of Income againist Loan Amount again"""

plt.scatter(df['Income'], df['Loan Amount'],s=30)
plt.show()

"""Fill the missing value with the median of the two dataset"""

df.Income.fillna(df.Income.median(), inplace=True)

"""##Column - Income Stability
There are 2 type of value except "nan" 
Most of them is "Low"
"""

print(df["Income Stability"].value_counts(dropna=False))
df["Income Stability"].value_counts(dropna=False).plot(kind='bar')

"""Fill the missing value with the mode of the two dataset"""

df["Income Stability"].fillna(df["Income Stability"].mode()[0], inplace=True)

"""Turn the values to numeric by replacing 'Low' with 0 and 'High' with 1"""

df["Income Stability"].replace(['Low', 'High'], [0,1], inplace=True)

"""##Column - Profession
There are 8 types of categorical values and no missing value
"""

print(df["Profession"].value_counts(dropna=False))
df["Profession"].value_counts(dropna=False).plot(kind='barh')

"""Replace with dummy variable columns


"""

df = creat_dummies(df, "Profession")

"""##Column - Type of Employment
There are 18 types of of Employment

There are 1/4 of rows are missing value

"""

print(df["Type of Employment"].value_counts(dropna=False))
df["Type of Employment"].value_counts(dropna=False).plot(kind='pie')

"""Filling missing data by random choosing from non missing values

"""

df = randomiseMissingData(df, 'Type of Employment')

"""Replace with dummy variable columns"""

df = creat_dummies(df,'Type of Employment')

"""##Column - Location
There are 3 types of "Location", no missing value

"""

print(df["Location"].value_counts(dropna=False))
df["Location"].value_counts(dropna=False).plot(kind='pie')

"""Replace with dummy variable columns"""

df = creat_dummies(df, "Location")

"""##Column - Loan Amount Request (USD)
Rename this column with "Loan Amount Request"

There are no missing value

"""

df.rename(columns={'Loan Amount Request (USD)':'Loan Amount Request'}, inplace=True)

print(df['Loan Amount Request'].isna().sum())

plt.scatter(df['Loan Amount Request'], df['Loan Amount'])
plt.xlabel('Loan Amount Request')
plt.ylabel('Loan Amount')
plt.show()

"""##Column - Current Loan Expenses (USD)
Rename this column with "Current Loan Expenses"

There are missing values
"""

df.rename(columns={'Current Loan Expenses (USD)':'Current Loan Expenses'}, inplace=True)

print(df['Current Loan Expenses'].isna().sum())

df['Current Loan Expenses'][df['Current Loan Expenses']<0]

"""There are many '-999', replace them with median of othe positive values

Also replace the missing value with the 0
"""

df['Current Loan Expenses'].replace({-999, df[df['Current Loan Expenses']>0]['Current Loan Expenses'].median()}, inplace=True)

df['Current Loan Expenses'].fillna(0, inplace=True)

"""##Column - Expense Type 1
There are 2 types of value, no missing value
"""

print(df["Expense Type 1"].value_counts(dropna=False))
df["Expense Type 1"].value_counts(dropna=False).plot(kind='pie')

"""Turn the values to numeric by replacing 'N' with 0 , 'Y' with 1"""

df["Expense Type 1"].replace(['N', 'Y'], [0,1], inplace=True)

"""##Column - Expense Type 2
There are 2 types of value, no missing value
"""

print(df["Expense Type 2"].value_counts(dropna=False))
df["Expense Type 2"].value_counts(dropna=False).plot(kind='pie')

"""Turn the values to numeric by replacing 'N' with 0 , 'Y' with 1"""

df["Expense Type 2"].replace(['N', 'Y'], [0,1], inplace=True)

"""##Column - Dependents

There are missing values
"""

print(df["Dependents"].value_counts(dropna=False))
df["Dependents"].value_counts(dropna=False).plot(kind='bar')

"""Replace the missing value with 0"""

df['Dependents'].fillna(0, inplace=True)

"""##Column - Credit Score
There are no nagetive value

There are missing values
"""

print('range:', df['Credit Score'].min(), df['Credit Score'].max())
print(df['Credit Score'].isna().sum())

"""Replace missing values with median"""

df['Credit Score'].fillna(df['Credit Score'].median(), inplace=True)

"""##Column - No. of Defaults
There are only 2 type of value [0, 1], no missing value
"""

print(df["No. of Defaults"].value_counts(dropna=False))
df["No. of Defaults"].value_counts(dropna=False).plot(kind='bar')

"""##Column - Has Active Credit Card
There are 3 types of value excluding the missing values
"""

print(df["Has Active Credit Card"].value_counts(dropna=False))
df["Has Active Credit Card"].value_counts(dropna=False).plot(kind='bar')

"""Filling missing data by random choosing from non missing values"""

df = randomiseMissingData(df, 'Has Active Credit Card')

"""Replace by dummy variables"""

df = creat_dummies(df, "Has Active Credit Card")

"""##Column - Property ID
Obviously, Property ID will not affect the loan amount

So remove this column
"""

df.drop(['Property ID'], axis=1, inplace=True)

"""##Column - Property Age
There are no nagative value and 4850 missing values


Plot the graph of Property Age against Loan Amount

"""

print('range:', df['Property Age'].min(), df['Property Age'].max())
print(df['Property Age'].isna().sum())

plt.scatter(df['Property Age'], df['Loan Amount'])
plt.xlabel('Property Age')
plt.ylabel('Loan Amount')
plt.show()

"""We can see there is a row with Property Age over 120000, it's twice of the second highest value


This row is in training dataset, so drop this row
"""

print(df.loc[df['Property Age']==df['Property Age'].max(), ['Property Age', 'Loan Amount']])
df.drop(df['Property Age'].idxmax(), inplace=True)

"""Replace missing value with median"""

df['Property Age'].fillna(df['Property Age'].median(), inplace=True)

"""##Column - Property Type
There are 4 types of value, no missing value
"""

print(df["Property Type"].value_counts(dropna=False))
df["Property Type"].value_counts(dropna=False).plot(kind='bar')

"""To turn to numerical data, replace with 4 dummy variable columns ( 'Property Type 1', 'Property Type 2', 'Property Type 3', 'Property Type 4' )"""

def creat_dummies_Property_Type(df, column_name):
  # get the dummies and store it in a variable
  dummies = pd.get_dummies(df[column_name])
  # rename the dummies columns
  dummies.rename(columns={1:'Property Type 1', 2:'Property Type 2', 3:'Property Type 3', 4:'Property Type 4'}, inplace=True)
  # Concatenate the dummies to original dataframe
  merged = pd.concat([df, dummies], axis=1)
  # drop the values
  merged.drop([column_name], axis=1, inplace=True)
  return merged

df = creat_dummies_Property_Type(df, "Property Type")

"""##Column - Property Location
There are 3 types of value

There are 345 missing values
"""

print(df["Property Location"].value_counts(dropna=False))
df["Property Location"].value_counts(dropna=False).plot(kind='pie')

"""Replace the missing value with mode"""

df['Property Location'].fillna(df['Property Location'].mode(), inplace=True)

"""To turn to numerical data, replace with dummy variable columns"""

def creat_dummies_Property_Location(df, column_name):
  # get the dummies and store it in a variable
  dummies = pd.get_dummies(df[column_name])
  # rename the dummies columns
  dummies.rename(columns={'Semi-Urban':'Semi-Urban Property', 'Rural':'Rural Property', 'Urban':'Urban Property'}, inplace=True)
  # Concatenate the dummies to original dataframe
  merged = pd.concat([df, dummies], axis=1)
  # drop the values
  merged.drop([column_name], axis=1, inplace=True)
  return merged

df = creat_dummies_Property_Location(df, "Property Location")

"""##Column - Co-Applicant
There are 3 type of values [1, 0, -999], no missing values
"""

print(df['Co-Applicant'].value_counts(dropna=False))
df['Co-Applicant'].value_counts(dropna=False).plot(kind='pie')

"""Replace the nagative value (-999) with 0"""

df['Co-Applicant'].clip(lower=0, inplace=True)

"""##Column - Property Price
There are some negative values, no missing values
"""

print('range:', df['Property Price'].min(), df['Property Price'].max())
print(df['Property Price'].isna().sum())

"""Plot the graph of Property Price against Loan Amount"""

plt.scatter(df['Property Price'], df['Loan Amount'])
plt.xlabel('Property Price')
plt.ylabel('Loan Amount')
plt.show()

"""Replace the -999 with median of other positive values"""

df['Property Price'].replace({-999, df[df['Property Price']>0]['Property Price'].median()}, inplace=True)

"""Split the data back to training data and testing data"""

test_df = df[df['Loan Amount'].isna()]
test_df.drop('Loan Amount', axis=1, inplace=True)
train_df = df.dropna(subset=['Loan Amount'])

"""##Column - Loan Amount

There are some -999 in this column
"""

train_df['Loan Amount'][train_df['Loan Amount']<0]

"""Loan Amount is the value that we want to predict, it's so important, can't be easily replaced by mean or median

So just remove the rows with  value "-999" in training dataset

"""

train_df = train_df[train_df['Loan Amount']>=0]

"""##Column - Customer ID
This row is clearly not related to the Loan Amount, but for testing data, it use to identify each row, so remove this column from training data, pop out from testing data
"""

train_df.drop("Customer ID", axis=1, inplace=True)
test_ID = test_df.pop('Customer ID')

"""#Model design and implementation

Get the column - 'Loan Amount' from training dataset and let it be y

Also reshape it to (len(y),1)
"""

y = train_df['Loan Amount']
y = y.values.reshape(len(y),1)

"""Get the other columns from training dataset and let it be x"""

x = train_df.drop(['Loan Amount'], axis=1)

"""##Linear Regression Model
The Linear Regression Model is implemented using Gradient Descent Algorithm

Cost function:
$$J(\theta_0,\theta_1,...)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta (x^{(i)})-y^{(i)})^2$$

Gradient Descent Algorithm:
$$n=0, \theta[n]=0, x^{(0)}=0$$
REPEAT {
  $$\theta[n+1] = \theta[n]-\frac{\alpha}{m}\sum_{i=1}^{m}(h_\theta (x^{(i)})-y^{(i)})x^{(i)}$$
} UNTIL $$J($$
"""

class LinearRegression:
    """
    Parameters
    ----------
    eta : float
        Learning rate
    eps : float
        Convergence criteria
    n_iterations : int
        No of passes over the training set
   
    Attributes
    ----------
    theta : theta/ after fitting the model
    bias : bias / after fitting the model
    cost : total error of the model after each iteration
    """

    def __init__(self, eta, eps,  n_iterations):
        self.eta = eta
        self.eps = eps
        self.n_iterations = n_iterations

    def fit(self, x, y):
        """Fit the training data

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training samples
        y : array-like, shape = [n_samples, n_target_values]
            Target values

        Returns
        -------
        self : object
        """
          
        self.theta = np.zeros((x.shape[1], 1))
        self.bias = 0
        m = x.shape[0]
        self.cost = []
        self.cost.append(self.costfuntion(x,y))
        print('initial cost: ',self.cost)
    
        for i in range(self.n_iterations):
            y_pred = np.dot(x, self.theta)+self.bias
            residuals = y_pred - y
            gradient_vector = np.dot(x.T, residuals)
            self.bias -= (self.eta / m) * np.sum(residuals)
            self.theta -= (self.eta / m) * gradient_vector
            new_cost = self.costfuntion(x,y)
            cost_diff = self.cost[-1]-new_cost
            self.cost.append(new_cost)
            if (i%100 == 0):
              print('itr:',i)
              print('cost:', new_cost)
              print('cost_diff',cost_diff)
            if (cost_diff < self.eps):
              print('itr:',i)
              print('cost:', new_cost)
              print('cost_diff',cost_diff)
              break
        return self

    def costfuntion(self, x, y):
      m = x.shape[0]
      y_pred = np.dot(x, self.theta)+self.bias
      residuals = y_pred - y
      cost = np.sum((residuals ** 2)) / (2 * m)
      return cost


    def predict(self, x):
        """ Predicts the value after the model has been trained.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        return np.dot(x, self.theta)+self.bias

"""First lets try traning the model with a low learning rate 0.000005, Convergence criteria 100 and  number of iterations 1000

"""

LR = LinearRegression(0.000005, 1000, 1000)
LR.fit(x, y)

"""The cost increase after the first iteration, the learning rate is too high.

Now try with learning rate 0.00000005
"""

LR = LinearRegression(0.00000005, 1000, 1000)
LR.fit(x, y)

"""The cost increase after the first iteration, the learning rate is still too high.

Now try with learning rate 0.000000000005
"""

LR = LinearRegression(0.000000000005, 1000, 3000)
LR.fit(x, y)

"""Plot the graph of cost over iteration"""

plt.plot(range(len(LR.cost)), LR.cost)  
plt.title("Cost over iteration")
plt.grid(axis="y")
plt.xlabel('iterations')
plt.ylabel('Cost')
plt.show()

"""The linear regression model is trained, now use the model to predict the Loan Amount"""

y_pred_LR = LR.predict(test_df)
y_pred_LR

"""# Regularized Linear Regression
To prevent the overfitting of the Linear Regression model, we can use a Regularized Linear Regression model.

It's a Linear Regression model with a extra Regularization Regression penalized term.

The Regularized Gradient Descent Algorithm are represented as follow:
**Hypothesis Model:**
$$h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2+...$$

**Cost function:**
$$J(\theta_0,\theta_1,...)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta (x^{(i)})-y^{(i)})^2$$

**Gradient Descent Algorithm:**
$$n=0, \theta[n]=0, x_0=0$$
Repeat until convergence {
  $$\theta[n+1] = \theta[n]-\frac{\alpha}{m}\sum_{i=1}^{m}(h_\theta (x^{(i)})-y^{(i)})x^{(i)}$$
}
"""

class RegularizedLinearRegression:
    """
    Parameters
    ----------
    eta : float
        Learning rate
    eps : float
        Convergence criteria
    ld  : float
        lambda (control the tradeoff)
    n_iterations : int
        No of passes over the training set
   
    Attributes
    ----------
    theta : theta/ after fitting the model
    bias : bias / after fitting the model
    cost : total error of the model after each iteration
    """

    def __init__(self, eta, eps, ld,  n_iterations):
        self.eta = eta
        self.eps = eps
        self.ld = ld
        self.n_iterations = n_iterations

    def fit(self, x, y):
        """Fit the training data

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training samples
        y : array-like, shape = [n_samples, n_target_values]
            Target values

        Returns
        -------
        self : object
        """
          
        self.theta = np.zeros((x.shape[1], 1))
        self.bias = 0
        m = x.shape[0]
        self.cost = []
        self.cost.append(self.costfuntion(x,y))
        print('initial cost: ',self.cost)
    
        for i in range(self.n_iterations):
            y_pred = np.dot(x, self.theta)+self.bias
            residuals = y_pred - y
            gradient_vector = np.dot(x.T, residuals)
            self.bias = self.bias * (1 - self.ld * self.eta / m) - (self.eta / m) * np.sum(residuals)
            self.theta = self.theta * (1 - self.ld * self.eta / m) - (self.eta / m) * gradient_vector
            new_cost = self.costfuntion(x,y)
            cost_diff = self.cost[-1]-new_cost
            self.cost.append(new_cost)
            if (i%100 == 0):
              print('itr:',i)
              print('cost:', new_cost)
              print('cost_diff',cost_diff)
            if (cost_diff < self.eps):
              print('itr:',i)
              print('cost:', new_cost)
              print('cost_diff',cost_diff)
              break
        return self

    def costfuntion(self, x, y):
      m = x.shape[0]
      y_pred = np.dot(x, self.theta)+self.bias
      residuals = y_pred - y
      cost = (np.sum((residuals ** 2)) + self.ld * np.sum((self.theta ** 2)) )/ (2 * m)
      return cost


    def predict(self, x):
        """ Predicts the value after the model has been trained.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        return np.dot(x, self.theta)+self.bias

"""Train the model with learning rate 0.000000000005, Convergence criteria 1000, lambda 1 and number of iterations 3000


"""

RLR = RegularizedLinearRegression(0.000000000005, 1000, 1, 3000)
RLR.fit(x, y)

plt.plot(range(len(RLR.cost)), RLR.cost)  
plt.title("Cost over iteration")
plt.grid(axis="y")
plt.xlabel('iterations')
plt.ylabel('Cost')
plt.show()

"""The Regularized Linear Regression model is trained, now use the model to predict the Loan Amount"""

y_pred_RLR = RLR.predict(test_df)
y_pred_RLR

"""#Deep Neural Network


import the required package first
"""

from pandas.core.common import standardize_mapping
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout

"""Building DNN using keras

The DNN model has two hidden layer

Both hidden layer has 64 Neurons, activation function is 'relu'

The configuretion of model:


*   optimizer: adam
*   loss function: mean square error (mse)


Show the Mean Absolute Error and Mean Absolute Percentage Error of each epoch


"""

def create_DNN():
  DNN = Sequential()
  DNN.add(Dense(64, input_dim=x.shape[1], activation='relu'))
  DNN.add(Dense(64, activation='relu'))
  DNN.add(Dense(1)) 
  DNN.compile(loss='mse', 
                optimizer='adam',
                metrics=['mae','mape'])
  return DNN

DNN = create_DNN()

DNN.fit(x, y,epochs=40,batch_size=40)

"""Predict with the trained DNN model"""

y_pred_DNN = DNN.predict(test_df)
y_pred_DNN

"""##DNN2
I make another approach of training the model.


As the traning data set can split into two set, one is accepted loan (loan amount > 0), one is unaccepted loan (loan anount = 0), so we create an array "loan_accepted" to indicate that.
"""

loan_acceptance = y.copy()
loan_acceptance[loan_acceptance > 0] = 1

"""We first build the model of predict loan accepted


Standardize the dataset:
"""

df = pd.concat([x, test_df], ignore_index=True)
min_max_scaler = preprocessing.MinMaxScaler()
df_st = min_max_scaler.fit_transform(df)
x_std = df_st[:df_st.shape[0]-3000]
test_std = df_st[df_st.shape[0]-3000:]

"""Building DNN_acceptance using keras

The model has two hidden layer

Both hidden layer has 64 Neurons, activation function is 'relu'

The output layer's activation function is 'sigmoid'

The configuretion of model:


*   optimizer: adam
*   loss function: binary crossentropy


Show the accuracy of each epoch
"""

DNN_acceptance = Sequential()
DNN_acceptance.add(Dense(64, input_dim=x_std.shape[1], activation='relu'))
DNN_acceptance.add(Dense(64, activation='relu'))
DNN_acceptance.add(Dense(1, activation='sigmoid'))
DNN_acceptance.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

DNN_acceptance.fit(x_std, loan_acceptance,epochs=40,batch_size=40)

"""Predict the loan acceptance of the testing dataset"""

y_pred_acceptance = np.round_(DNN_acceptance.predict(test_std))
unique, counts = np.unique(y_pred_acceptance, return_counts=True)
dict(zip(unique, counts))

"""Build the DNN model (DNN2) of accepted loan customer's loan amount, the setting is same as the first DNN model."""

DNN2 = create_DNN()

"""Get the accepted customer data from the traing dataset"""

x_accepted = train_df[train_df["Loan Amount"] > 0]
y_accepted = x_accepted.pop("Loan Amount")
y_accepted = y_accepted.values.reshape(len(y_accepted),1)

"""Train the DNN2 model"""

DNN2.fit(x_accepted, y_accepted, epochs=40,batch_size=40)

y_pred_DNN2 = DNN2.predict(test_df) * y_pred_acceptance
y_pred_DNN2

"""#Performance evaluation and discussions 


First compare the Linear Regression (LR) and Regularized Linear Regression (RLR) model. The cost of this two model after traning are 539861934 and 539861934, it's very close. As RLR model cost function include an extra regularized cost, it can prevent overfitting of the model, so RLR is more prefer than LR. 


Then compare the two Deep Neural Network model - DNN, and the combination of DNN_acceptance and DNN2. DNN and DNN2 use the same model but different training data, as DNN use the all customers data and DNN2 only use the accepted customers data. We can see that all the mean square error (MSE), mean absolute error (MAE) and  mean absolute percentage error (MAPE) of DNN2 is much lower than DNN.

| Model | MSE | MAE | MAPE         
| ----- | ------: | ------: | -----:
| DNN | 1072337856 | 23102.255 | 11997613654016.0000
| DNN2 | 38637864  | 4224.297  | 6.3832

Especially the MAPE, the diffence is 1879560981015 times. We can clearly see that use the only use the accepted customers data to predict the loan amount is much more acurrate. I think the reason is because the unaccepted customers account for arround a quarter of all, it quite a large amonnt, as their loan ammount are all 0, it makes DNN bias to them when training and the error become large. DNN2 only consider the accepted customers, the model will not bias the unaccepted customers. DNN2 require an acceptance prediction, so we hane DNN_acceptance, and the acurracy is 0.9129 which means the model is nice. As a result DNN2 is much better than DNN.


Last we compare RLR and DNN2. As RLR's traing is not quite linear and have many outlier as it contain many unaccepted customers, the model will not acurrate. So I think that the predicted result of DNN2 (y_pred_DNN2) is the most acurrate.

Output the LR model presiction result to 20037676D_LR.csv
"""

result_LR = pd.DataFrame( np.concatenate((test_ID.values.reshape(len(test_ID), 1) ,y_pred_LR),axis=1), columns = [ 'Customer_ID', 'Loan Amount' ])
result_LR

result_LR.to_csv('20037676D_LR.csv', index = False)

"""Output the RLR model presiction result to 20037676D_RLR.csv"""

result_RLR = pd.DataFrame( np.concatenate((test_ID.values.reshape(len(test_ID), 1) ,y_pred_RLR),axis=1), columns = [ 'Customer_ID', 'Loan Amount' ])
result_RLR

result_RLR.to_csv('20037676D_RLR.csv', index = False)

"""Output the DNN model presiction result to 20037676D_DNN.csv"""

result_DNN = pd.DataFrame( np.concatenate((test_ID.values.reshape(len(test_ID), 1) ,y_pred_DNN),axis=1), columns = [ 'Customer_ID', 'Loan Amount' ])
result_DNN

result_DNN.to_csv('20037676D_DNN.csv', index = False)

"""Output the DNN2 model presiction result to 20037676D_DNN2.csv"""

result_DNN2 = pd.DataFrame( np.concatenate((test_ID.values.reshape(len(test_ID), 1) ,y_pred_DNN2),axis=1), columns = [ 'Customer_ID', 'Loan Amount' ])
result_DNN2

result_DNN2.to_csv('20037676D_DNN2.csv', index = False)

"""#Summary

We have built 4 model - Linear Regression, Regularized Linear Regression, Deep Neural Network and a special DNN model which separate as two parts. And we found that the last model has the least error and is most accurate.

# Future work 
The idea of DNN2 can be also implement by regression model. The acceptance prediction can implemented by Logistic Regression and the accepted customers' loan amount prediction just use Linear Regression. This model can be include in the furtue work.

#Reference


*   API reference — pandas 1.4.2 documentation https://pandas.pydata.org/docs/reference/index.html
*   NumPy Reference — NumPy v1.22 Manual https://numpy.org/doc/stable/reference/index.html
*   Keras API reference https://keras.io/api/
"""