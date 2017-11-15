
#############################################################################################
##################### Simple Linear Regression - Python  ####################################
#############################################################################################

#---------------------------------------------------------------------------------
# Step : 1 Importing the libraries
#---------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#---------------------------------------------------------------------------------
# Step : 2 Data Preprocessing
#--------------------------------------------------------------------------------
         #2(a) Importing the dataset
dataset = pd.read_csv('Employee_Salary.csv')
Var_Independent  = dataset.iloc[:, :-1].values
Var_dependent = dataset.iloc[:, 1].values

         #2(b) Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
Var_I_train, Var_I_test, Var_D_train, Var_D_test = train_test_split(Var_Independent, Var_dependent, test_size = 1/3.0, random_state = 0)

#--------------------------------------------------------------------------------
# Step : 3 Data modelling
#--------------------------------------------------------------------------------
        #3(a) Fitting Naive Bayes to the Training set
		
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Var_I_train, Var_D_train)

        #3(b) Predicting the Test set results
Var_D_pred = regressor.predict(Var_I_test)


#--------------------------------------------------------------------------------
# Step : 4 Data Visualising 
#--------------------------------------------------------------------------------
         #4(a) for the Training set results
plt.scatter(Var_I_train, y_train, color = 'red')
plt.plot(Var_I_train, regressor.predict(Var_I_train), color = 'blue')
plt.title('Employee Salary vs Experience (Training set)')
plt.xlabel('Employee Years of Experience')
plt.ylabel('Employee Salary')
plt.show()

      #4(b) for the Training set results
	  
plt.scatter(Var_I_test, Var_D_test, color = 'red')
plt.plot(Var_I_train, regressor.predict(Var_I_train), color = 'blue')
plt.title('Employee Salary vs Experience (Test set)')
plt.xlabel('Employee Years of Experience')
plt.ylabel('Employee Salary')
plt.show()