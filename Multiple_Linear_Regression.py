#################################################################################
##################### Multiple Linear Regression - Python  ######################
#################################################################################

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
dataset = pd.read_csv('50_Startup Orgnizations.csv')
Var_Independent = dataset.iloc[:, :-1].values
Var_dependent = dataset.iloc[:, 4].values

         #2(b) Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
Var_Independent[:, 3] = labelencoder.fit_transform(Var_Independent[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
Var_Independent = onehotencoder.fit_transform(Var_Independent).toarray()

         #2(c) Avoiding the Dummy Variable Trap
Var_Independent = Var_Independent[:, 1:]

         #2(d) Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
Var_I_train, Var_I_test, Var_D_train, Var_D_test = train_test_split(Var_Independent, Var_dependent, test_size = 0.2, random_state = 0)

#--------------------------------------------------------------------------------
# Step : 3 Data modelling
#--------------------------------------------------------------------------------
        #3(a) Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Var_I_train,  Var_D_train)

        #3(b) Predicting the Test set results
VAR_D_pred = regressor.predict(Var_I_test)