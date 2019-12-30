#!/usr/bin/env python
# coding: utf-8

# #  <h1><center>Grocery Sales Forcesting for Supermarket</center></h1>
# 
# <img src="header.png">
# 
# Img Source: Kaggle.com
# 
# ## Table of Contents
# 
# 1. Abstract
# 
# 2. Introduction
# 
# 3. Data Sources
# 
#    3.a. Data Loading
#    
#    3.b. Anamoly Detection
#    
#    3.c. Data Preparation and Data Cleaning
#    
# 4. Analyzing Impact of Oil on the Sales
# 
# 5. Data Blending
# 
# 6. Product Purchase Trend
# 
#    6.a Fetching N most purchased products
# 
# 7. Exploratory Data Analysis
# 
# 8. Data Transformation
# 
#    8.a. One Hot Encoding
#    
# 9. Regression Techniques
#    9.a Linear Regression
#    
#    9.b Decision Tree Regressors
#    
#    9.c Extra Tree Regressors
#    
#    9.d Random Forest Regressors
#    
#    9.e Gradient Boosting Regressors
#    
#    9.f XGBoost
#    
# 10. Light Gradient Boosting Method (LGBM)
# 
# 11. Creating Neural Network
# 
# 12. Conclusion
# 
# 13. References and Attributions
# 
# 

# # 1. Abstract
# 
# <img src="Forecast.jpg">
# 
# Product sales forecasting is a major aspect of purchasing management. Forecasts are crucial in
# determining inventory stock levels, and accurately estimating future demand for goods has been an
# ongoing challenge, especially in the Supermarkets and Grocery Stores industry. If goods are not readily
# available or goods availability is more than demand overall profit can be compromised. As a result, sales
# forecasting for goods can be significant to ensure loss is minimized. Additionally, the problem becomes
# more complex as retailers add new locations with unique needs, new products, ever transitioning
# seasonal tastes, and unpredictable product marketing. In this analysis, a forecasting model is developed
# using machine learning algorithms to improve the accurately forecasts product sales. The proposed
# model is especially targeted to support the future purchase and more accurate forecasts product sales
# and is not intended to change current subjective forecasting methods. A model based on a real grocery
# store's data is developed in order to validate the use of the various machine learning algorithms. In the
# case study, multiple regression methods are compared. The methods impact on forecast product
# availability in store to ensure they have just enough products at right time.
# 
# ## The aim of this project is to forecast more accurate product sales for the Ecuadorian supermarket chain based on certain features.
# 

# # 2. Introduction
# 
# In this project, we are trying to forecasts product sales based on the items, stores, transaction and other
# dependent variables like holidays and oil prices.
# 
# This is a Kaggle Competition called "Corporación Favorita Grocery Sales Forecasting" where the task is to
# predict stocking of products to better ensure grocery stores please customers by having just enough of
# the right products at the right time.
# 
# For this particular problem, we have analyzed the data as a supervised learning problem. In order to
# forecasts the sales we have compared different regression models like Linear Regression, Decision Tree,
# ExtraTreeRegressor, Gradient Boosting, Random Forest and XgBoost. Further to optimize the results we
# have used multilayer perception (MLP: a class of feed forward artificial neural network) and LightGBM (
# gradient boosting framework that uses tree based learning algorithms).
# 
# The data comes in the shape of multiple files. First, the training data (train.csv) essentially contains the sales by date, store, and item. The test data (test.csv) contains the same features without the sales information, which we are tasked to predict. The train vs test split is based on the date. In addition, some test items are not included in the train data.
# 
# <img src="background.jpg">

# In[2]:


pip install lightgbm


# In[2]:


#Loading all the relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder,minmax_scale,PolynomialFeatures,StandardScaler,Normalizer
from sklearn.model_selection import KFold,GridSearchCV,train_test_split
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
#from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
from datetime import date, timedelta
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import keras
import sys
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import lightgbm as lgb


# # 3. Data Sources
# There are 5 additional data files that provide the following information:
# 
# -- stores.csv : Details about the stores, such as location and type.
# 
# -- items.csv: Item metadata, such as class and whether they are perishable. Note, that perishable items have a higher scoring weight than others.
# 
# -- transactions.csv: Count of sales transactions for the training data
# 
# -- oil.csv: Daily oil price. This is relevant, because “Ecuador is an oil-dependent country and its economical health is highly vulnerable to shocks in oil prices.” (source)
# 
# -- holidays_events.csv: Holidays in Ecuador. Some holidays can be transferred to another day (possibly from weekend to weekday).
# 
# 
# <img src="Capture1.jpg">
# 
# 
# The text in the document by Source Kaggle is licensed under CC BY 3.0 https://creativecommons.org/licenses/by/3.0/us/
# 

# # 3.a. Data Loading

# In[4]:


#Loading the data
dtypes = {'store_nbr': np.dtype('int64'),
          'item_nbr': np.dtype('int64'),
          'unit_sales': np.dtype('float64'),
          'onpromotion': np.dtype('O')}

Sales = pd.read_csv('trainset.csv', dtype=dtypes)
test = pd.read_csv('test.csv', dtype=dtypes)
stores = pd.read_csv('stores.csv')
items = pd.read_csv('items.csv')
trans = pd.read_csv('transactions.csv')
#oil = pd.read_csv('../input/oil.csv') #we upload this database later
holidays = pd.read_csv('holidays_events.csv')


# In[5]:


#sampling the data, since the data is too huge to carry put any operations
date_mask = (Sales['date'] >= '2017-07-15') & (Sales['date'] <= '2017-08-15')

Salesdf = Sales[date_mask]

#Print the size
len(Salesdf)


# # 3.b. Anamoly Detection

# In[6]:


#Load the data
oil = pd.read_csv('oil.csv')

#add missing date
min_oil_date = min(Salesdf.date)
max_oil_date = max(Salesdf.date)

calendar = []

d1 = datetime.datetime.strptime(min_oil_date, '%Y-%m-%d')  # start date
d2 = datetime.datetime.strptime(max_oil_date, '%Y-%m-%d')  # end date

delta = d2 - d1         # timedelta

for i in range(delta.days + 1):
    calendar.append(datetime.date.strftime(d1 + timedelta(days=i), '%Y-%m-%d'))

calendar = pd.DataFrame({'date':calendar})

oil = calendar.merge(oil, left_on='date', right_on='date', how='left')


# In[7]:


oil.head()


# # 3.c Data Preparation and Data Cleaning

# In[8]:


#Check how many NA
print(oil.isnull().sum(), '\n')

#Type


print('Type : ', '\n', oil.dtypes)

#Print the 3 first line
oil.head(5)


# # 4. Analyzing Impact of Oil on the sales of other products

# In[9]:


#Check index to apply the formula
na_index_oil = oil[oil['dcoilwtico'].isnull() == True].index.values

#Define the index to use to apply the formala
na_index_oil_plus = na_index_oil.copy()
na_index_oil_minus = np.maximum(0, na_index_oil-1)

for i in range(len(na_index_oil)):
    k = 1
    while (na_index_oil[min(i+k,len(na_index_oil)-1)] == na_index_oil[i]+k):
        k += 1
    na_index_oil_plus[i] = min(len(oil)-1, na_index_oil_plus[i] + k )

#Apply the formula
for i in range(len(na_index_oil)):
    if (na_index_oil[i] == 0):
        oil.loc[na_index_oil[i], 'dcoilwtico'] = oil.loc[na_index_oil_plus[i], 'dcoilwtico']
    elif (na_index_oil[i] == len(oil)):
        oil.loc[na_index_oil[i], 'dcoilwtico'] = oil.loc[na_index_oil_minus[i], 'dcoilwtico']
    else:
        oil.loc[na_index_oil[i], 'dcoilwtico'] = (oil.loc[na_index_oil_plus[i], 'dcoilwtico'] + oil.loc[na_index_oil_minus[i], 'dcoilwtico'])/ 2   


# In[10]:


oil.isnull().sum()


# In[11]:


#Plot the oil values
oil_plot = oil['dcoilwtico'].copy()
oil_plot.index = oil['date'].copy()
oil_plot.plot()
plt.show()


# # By the end of the analysis it is evident that the sale of oil as an important commodity has a significant impact on the unit sales of other products.

# The text in the document by Source Wikipedia is licensed under CC BY 3.0 https://creativecommons.org/licenses/by/3.0/us/
# 
# The text in the document by Analytics Vidhya is licensed under CC BY 3.0 https://creativecommons.org/licenses/by/3.0/us/
# 
# The code in the document by Source Kaggle is licensed under the MIT License https://opensource.org/licenses/MIT

# # 5. Data Blending

# In[12]:


#Merge train
Salesdf = Salesdf.drop('id', axis = 1)
Salesdf = Salesdf.merge(stores, left_on='store_nbr', right_on='store_nbr', how='left')
Salesdf = Salesdf.merge(items, left_on='item_nbr', right_on='item_nbr', how='left')
Salesdf = Salesdf.merge(holidays, left_on='date', right_on='date', how='left')
Salesdf = Salesdf.merge(oil, left_on='date', right_on='date', how='left')
Salesdf = Salesdf.drop(['description', 'state', 'locale_name', 'class'], axis = 1)


# In[14]:


Salesdf.to_csv('datsettrain.csv',index=False)


# In[15]:


Salesdf.isnull().sum().sort_values(ascending=False)


# In[16]:


#Shape
print('Shape : ', Salesdf.shape, '\n')

#Type
print('Type : ', '\n', Salesdf.dtypes)

#Summary
Salesdf.describe()


# In[17]:


#5 random lines
Salesdf.sample(10)


# In[18]:


sns.countplot(x='store_nbr', data=Salesdf);


# # The the above plot shows the number of stores according to each store type.

# In[33]:


Salesdf.item_nbr.unique


# In[34]:


x=itemfreq(Salesdf.item_nbr)
x=x[x[:,1].argsort()[::-1]]
x


# In[35]:


x[:,0][0:10]


# In[36]:


s=Salesdf.item_nbr.as_matrix()
s


# In[37]:


t=s.reshape(s.shape[0],1)
t


# In[38]:


t1=np.where(t==x)
t1


# # 6. Product Purchase Trend

# ##  6.a. Fetching the N most purchased products

# In[148]:


#Fetching the N most purchased products#
def N_most_labels(data, variable , N , all='TRUE'):
    labels_freq_pd = itemfreq(data[variable])
    labels_freq_pd = labels_freq_pd[labels_freq_pd[:, 1].argsort()[::-1]] #[::-1] ==> to sort in descending order
    
    if all == 'FALSE':
        main_labels = labels_freq_pd[:,0][0:N]
    else: 
        main_labels = labels_freq_pd[:,0][:]
        
    labels_raw_np = data[variable].as_matrix() #transform in numpy
    labels_raw_np = labels_raw_np.reshape(labels_raw_np.shape[0],1)

    labels_filtered_index = np.where(labels_raw_np == main_labels)
    
    return labels_freq_pd, labels_filtered_index

label_freq, labels_filtered_index = N_most_labels(data = Salesdf, variable = "item_nbr", N = 20, all='FALSE')
print("labels_filtered_index[0].shape = ", labels_filtered_index[0].shape)

Salesdf_filtered = Salesdf.loc[labels_filtered_index[0],:]


# The number of most purchased products gives us the idea of looking for the sales for those products.

# The code in the document by Source Kaggle is licensed under the MIT License https://opensource.org/licenses/MIT

# In[149]:


label_freq[0:10]


# In[150]:


Salesdf_filtered.sample(3)


# In[151]:


#Fill in cells if there is no holiday by the value : "no_holyday"
na_index_Salesdf = Salesdf_filtered[Salesdf_filtered['type_y'].isnull() == True].index.values
print("Size of na_index_Salesdf : ", len(na_index_Salesdf), '\n')

Salesdf_filtered.loc[Salesdf_filtered['type_y'].isnull(), 'type_y'] = "no_holyday"
Salesdf_filtered.loc[Salesdf_filtered['locale'].isnull(), 'locale'] = "no_locale"
Salesdf_filtered.loc[Salesdf_filtered['transferred'].isnull(), 'transferred'] = "no_holyday"
    
#check is there is NA
Salesdf_filtered.isnull().sum()


# In[152]:


def get_month_year(df):
    df['month'] = df.date.apply(lambda x: x.split('-')[1])
    df['year'] = df.date.apply(lambda x: x.split('-')[0])
    
    return df

get_month_year(Salesdf_filtered);


# In[153]:


Salesdf_filtered['date'] = pd.to_datetime(Salesdf_filtered['date'])
Salesdf_filtered['day'] = Salesdf_filtered['date'].dt.weekday_name
Salesdf_filtered = Salesdf_filtered.drop('date', axis=1)


# In[154]:


Salesdf_filtered.sample(10)


# In[155]:


strain = Salesdf.sample(frac=0.01,replace=True)


# # 7. Exploratory Data Analysis

# In[156]:


# Plotting Sales per Item Family
fig, (axis1) = plt.subplots(1,1,figsize=(30,4))
sns.barplot(x='family', y='unit_sales', data=strain, ax=axis1)


# # The above plot shows the sales of products per Item family. The average unit_sales ranges somewhere between 0 to 20 with the maximum for the Food and Beverages item family.

# In[157]:


# Plotting Sales per Store Type
fig, (axis1) = plt.subplots(1,1,figsize=(15,4))
sns.barplot(x='type_x', y='unit_sales', data=strain, ax=axis1)


# # The bar plot of Total Sales per store type shows that Store type B has the maximum sales over other store types.

# In[158]:


# Plotting Stores in Cities
fig, (axis1) = plt.subplots(1,1,figsize=(30,4))
sns.countplot(x=stores['city'], data=stores, ax=axis1)


# # The City of Quito has the most number of Stores and the second being the city of Guayaquil.

# In[51]:


# Plotting Stores in States
fig, (axis1) = plt.subplots(1,1,figsize=(30,4))
sns.countplot(x=stores['state'], data=stores, ax=axis1)


# # The States of Pichincha Santo and the state of Guayas has the most number of Stores.

# In[52]:


# Stacked Barplots of Types against clusters
plt.style.use('seaborn-white')
#plt.style.use('dark_background')
type_cluster = stores.groupby(['type','cluster']).size()
type_cluster.unstack().plot(kind='bar',stacked=True, colormap= 'PuBu', figsize=(13,11),  grid=False)
plt.title('Stacked Barplot of Store types and their cluster distribution', fontsize=18)
plt.ylabel('Count of clusters in a particular store type', fontsize=16)
plt.xlabel('Store type', fontsize=16)
plt.show()


# # The above bar plot shows the cluster distribution across the store types.

# In[53]:


# Stacked barplot of types of stores across the different cities
plt.style.use('seaborn-white')
city_cluster = stores.groupby(['city','type']).store_nbr.size()
city_cluster.unstack().plot(kind='bar',stacked=True, colormap= 'viridis', figsize=(13,11),  grid=False)
plt.title('Stacked Barplot of Store types opened for each city')
plt.ylabel('Count of stores for a particular city')
plt.show()


# In[54]:


# Holiday Events data
plt.style.use('seaborn-white')
# plt.style.use('dark_background')
holiday_local_type = holidays.groupby(['locale_name', 'type']).size()
holiday_local_type.unstack().plot(kind='bar',stacked=True, colormap= 'magma_r', figsize=(12,10),  grid=False)
plt.title('Stacked Barplot of locale name against event type')
plt.ylabel('Count of entries')
plt.show()


# # 8. Data Transformation Techniques

# # 8.a One hot Encoding

# ## What is One Hot Encoding?
# A one hot encoding is a representation of categorical variables as binary vectors.
# This first requires that the categorical values be mapped to integer values.
# Then, each integer value is represented as a binary vector that is all zero values except the index of the integer, which is marked with a 1.
# 
# ## Why Use a One Hot Encoding?
# A one hot encoding allows the representation of categorical data to be more expressive.
# Many machine learning algorithms cannot work with categorical data directly. The categories must be converted into numbers. This is required for both input and output variables that are categorical.
# 
# 
# ## The text in the document by Source Wikipedia is licensed under CC BY 3.0 https://creativecommons.org/licenses/by/3.0/us/
# 

# In[55]:


## One hot encoding using get_dummies on pandas dataframe.
dummy_variables = ['onpromotion','city','type_x','cluster','store_nbr','item_nbr',
                'family','perishable','type_y', 'locale', 'transferred', 'month', 'day']

for var in dummy_variables:
    dummy = pd.get_dummies(Salesdf_filtered[var], prefix = var, drop_first = False)
    Salesdf_filtered = pd.concat([Salesdf_filtered, dummy], axis = 1)

Salesdf_filtered = Salesdf_filtered.drop(dummy_variables, axis = 1)
Salesdf_filtered = Salesdf_filtered.drop(['year'], axis = 1)


# In[56]:


Salesdf_filtered.info()
#Salesdf_filtered.sample(10)


# In[57]:


Salesdf_filtered.head()


# # The above dataframe contains data after the one hot encoding technique is applied to the data.

# In[58]:


#Re-scale
#We keep this value to re-scale the predicted unit_sales values in the following lines of code.
min_train, max_train = Salesdf_filtered['unit_sales'].min(), Salesdf_filtered['unit_sales'].max()


# In[59]:


scalable_variables = ['unit_sales','dcoilwtico']

for var in scalable_variables:
    mini, maxi = Salesdf_filtered[var].min(), Salesdf_filtered[var].max()
    Salesdf_filtered.loc[:,var] = (Salesdf_filtered[var] - mini) / (maxi - mini)


# In[60]:


print('Shape : ', Salesdf_filtered.shape)
Salesdf_filtered.sample(10)


# In[61]:


Salesdf_filtered.isnull().sum()


# In[62]:


#train database without unit_sales
Salesdf_filtered = Salesdf_filtered.reset_index(drop=True)  #we reset the index
y = Salesdf_filtered['unit_sales']
X = Salesdf_filtered.drop(['unit_sales'], axis = 1)

print('Shape X :', X.shape)
print('Shape y :', y.shape)


# In[63]:


num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=num_test, random_state=15)
print('X_train shape :', X_train.shape)
print('y_train shape :', y_train.shape)
print('X_test shape :', X_test.shape)
print('y_test shape :', y_test.shape)


# # 9. Regression Model fitting techniques.

# # 9.a. Linear Regression

# ## Linear Regression
# 
# Linear Regression is a linear approach for modelling the relationship between a scalar dependent variable y and one or more explanatory variables (or independent variables) denoted X. The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression.
# 
# Linear regression models are often fitted using the least squares approach, but they may also be fitted in other ways, such as by minimizing the "lack of fit" in some other norm (as with least absolute deviations regression), or by minimizing a penalized version of the least squares cost function as in ridge regression (L2-norm penalty) and lasso (L1-norm penalty). 
# 
# 
# ## The text in the document by Source Wikipedia is licensed under CC BY 3.0 https://creativecommons.org/licenses/by/3.0/us/

# In[64]:


# Fit the linear model
model = linear_model.LinearRegression()
results = model.fit(X_train, y_train)
print(results)


# In[65]:


# Print the coefficients
print (results.intercept_, results.coef_)


# In[67]:


import statsmodels.api as sm
import statsmodels.regression.linear_model as sm
model = sm.OLS(y_train, X_train)
results = model.fit()
# Statsmodels gives R-like statistical output
results.summary()
# Here the Ordinalry Least Squares method has given us the r^2 value of 0.354 which is poor for model to be predicted on these variables.


# # The Linear regression model gives a low R- square value of 0.354

# ## Checking for VIF for eliminating multicollinearity and overfitting

# In[68]:


#Implementing VIF (Variance Inflation Factor) to check whether the selected independent variables are correct for prediction 
# or not. Also, 'item_nbr', 'perishable' and 'dcoilwtico' had very close levels of co-relation with price which makes us to investigate whether all 
# three are important or not.
indep=['dcoilwtico','perishable','item_nbr','store_nbr','cluster']
X=Salesdf[indep]


# In[69]:


from statsmodels.stats.outliers_influence import variance_inflation_factor  
thresh=10 #Setting a threshold of 10 as a sign of serious and sever multi-collinearity
for i in np.arange(0,len(indep)):
    vif=[variance_inflation_factor(X[indep].values,ix)
    for ix in range(X[indep].shape[1])]
    maxloc=vif.index(max(vif))
    if max(vif) > thresh:
        print ("vif :", vif)
        print( X[indep].columns[maxloc] )
        del indep[maxloc]
    else:
        break
        
    print ('Final variables: ', indep)


# In[70]:


X[indep].head(5)


# In[71]:


X=Salesdf[['perishable', 'item_nbr', 'store_nbr', 'cluster']]
y=Salesdf[["unit_sales"]]


# ## Cross Validation using Scikit Learn
# 
# R^2 value is basically dependant on the way the data is split. Hence, there may be times when the R^2 value may not be able to
# represent the model's ability to generalize. For this we perform cross validation.

# In[72]:


reg=linear_model.LinearRegression()
cv_results=cross_val_score(reg,X_train,y_train,cv=5)
print(cv_results)
print(np.mean(cv_results))
print(np.std(cv_results))
#Using cross validation of score 5


# ## Regularization
# 
# We perform regularization in order to alter the loss function to penalize it for having higher coefficients for each feature variable. And as we know, large coefficients leads to overfitting.

# In[73]:


ridge = Ridge(alpha=0.1, normalize = True)
ridge.fit(X_train,y_train)
ridge_pred=ridge.predict(X_test)
ridge.score(X_test,y_test)
#The score is pretty much similar to the linear model built which ensures that the model has passed the Ridge regression test
# for regularization
#Ridge is used to penalize the loss function by adding the OLS loss function to the square of each coefficient multiplied by alpha.


# ## 9.b DecisionTree Regressor
# 
# A decision tree is a decision support tool that uses a tree-like graph or model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.
# 
# 

# In[74]:


dtr=DecisionTreeRegressor(max_depth=10,min_samples_leaf=5,max_leaf_nodes=5)


# In[75]:


dtr.fit(X_train,y_train)
y_pred=dtr.predict(X_test)

print('R2 score = ',r2_score(y_test, y_pred), '/ 1.0')
print('MSE score = ',mean_squared_error(y_test, y_pred), '/ 0.0')

##using a decision tree greatly improves the accurancy of model prediction.


# ## 9.c. ExtraTreesRegressor

# Extra-trees differ from classic decision trees in the way they are built. When looking for the best split to separate the samples of a node into two groups, random splits are drawn for each of the max_features randomly selected features and the best split among those is chosen.

# In[76]:


etr = ExtraTreesRegressor()

# Choose some parameter combinations to try

parameters = {'n_estimators': [5,10,100],
              'criterion': ['mse'],
              'max_depth': [5,10,15], 
              'min_samples_split': [2,5,10],
              'min_samples_leaf': [1,5]
             }
#We have to use RandomForestRegressor's own scorer (which is R^2 score)

#Determines the cross-validation splitting strategy /to specify the number of folds in a (Stratified)KFold

grid_obj = GridSearchCV(etr, parameters,
                        cv=3, 
                        n_jobs=-1, #Number of jobs to run in parallel
                        verbose=1)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
etr = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
etr.fit(X_train, y_train)


# In[77]:


y_pred = etr.predict(X_test)

print('R2 score = ',r2_score(y_test, y_pred), '/ 1.0')
print('MSE score = ',mean_squared_error(y_test, y_pred), '/ 0.0')


# ## 9.d Random Forest Regressor
# 
# Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.Random decision forests correct for decision trees' habit of overfitting to their training set.

# In[78]:


# Choose the type of classifier. 
RFR = RandomForestRegressor()

# Choose some parameter combinations to try
parameters = {'n_estimators': [5, 10, 100],
              'min_samples_leaf': [1,5]
             }


#We have to use RandomForestRegressor's own scorer (which is R^2 score)

#Determines the cross-validation splitting strategy /to specify the number of folds in a (Stratified)KFold
grid_obj = GridSearchCV(RFR, parameters,
                        cv=5, 
                        n_jobs=-1, #Number of jobs to run in parallel
                        verbose=1)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
RFR = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
RFR.fit(X_train, y_train)


# In[79]:


y_pred = RFR.predict(X_test)

print('R2 score = ',r2_score(y_test, y_pred), '/ 1.0')
print('MSE score = ',mean_squared_error(y_test, y_pred), '/ 0.0')


# In[80]:


RFR = RandomForestRegressor()

# Choose some parameter combinations to try
parameters = {'n_estimators': [5,10,100],
              'criterion': ['mse'],
              'max_depth': [5,10,15], 
              'min_samples_split': [2,5,10],
              'min_samples_leaf': [1,5]
             }


#We have to use RandomForestRegressor's own scorer (which is R^2 score)

#Determines the cross-validation splitting strategy /to specify the number of folds in a (Stratified)KFold
grid_obj = GridSearchCV(RFR, parameters,
                        cv=5, 
                        n_jobs=-1, #Number of jobs to run in parallel
                        verbose=1)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
RFR = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
RFR.fit(X_train, y_train)


# In[81]:


y_pred = RFR.predict(X_test)

print('R2 score = ',r2_score(y_test, y_pred), '/ 1.0')
print('MSE score = ',mean_squared_error(y_test, y_pred), '/ 0.0')


# ## 9.e. Gradient Boosting Regressor
# 
# The idea of boosting came out of the idea of whether a weak learner can be modified to become better.
# A weak hypothesis or weak learner is defined as one whose performance is at least slightly better than random chance.
# Hypothesis boosting was the idea of filtering observations, leaving those observations that the weak learner can handle and focusing on developing new weak learns to handle the remaining difficult observations.
# 
# ## How Gradient Boosting Works
# 
# Gradient boosting involves three elements:
# 
# -- A loss function to be optimized.
# 
# -- A weak learner to make predictions.
# 
# -- An additive model to add weak learners to minimize the loss function.

# In[82]:


#gbr = GradientBoostingRegressor(loss='huber',learning_rate=0.3,n_estimators=100,max_depth=5,min_samples_split=3)
gbr=GradientBoostingRegressor()

parameters = {'n_estimators': [5,10],
              'loss':['huber'],
              'criterion': ['mse'],
              'max_depth': [5,10], 
              'min_samples_split': [2,5],
              'min_samples_leaf': [1,5]
             }

#Determines the cross-validation splitting strategy /to specify the number of folds in a (Stratified)KFold
grid_obj = GridSearchCV(gbr, parameters,
                        cv=5, 
                        n_jobs=-1, #Number of jobs to run in parallel
                        verbose=1)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
gbr = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
gbr.fit(X_train, y_train)


# In[83]:


y_pred = gbr.predict(X_test)

print('R2 score using Gradient Boosting= ',r2_score(y_test, y_pred), '/ 1.0')
print('MSE score using Gradient Boosting= ',mean_squared_error(y_test, y_pred), '/ 0.0')


# In[84]:


gbr = GradientBoostingRegressor(loss='ls',learning_rate=0.1,n_estimators=150,max_depth=10,min_samples_split=5)


parameters = {'n_estimators': [5,15,150],
              'loss':['ls','huber'],
              'criterion': ['mse'],
              'max_depth': [10,15], 
              'min_samples_split': [2,5],
              'min_samples_leaf': [1,5]
             }

#Determines the cross-validation splitting strategy /to specify the number of folds in a (Stratified)KFold
grid_obj = GridSearchCV(gbr, parameters,
                        cv=5, 
                        n_jobs=-1, #Number of jobs to run in parallel
                        verbose=1)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
gbr = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
gbr.fit(X_train, y_train)


# In[85]:


y_pred = RFR.predict(X_test)

print('R2 score using Gradient Boosting= ',r2_score(y_test, y_pred), '/ 1.0')
print('MSE score using Gradient Boosting= ',mean_squared_error(y_test, y_pred), '/ 0.0')


# ## 9.f. XGBOOST
# 
# XGBoost (eXtreme Gradient Boosting) is a direct application of Gradient Boosting for decision trees
# 
# Main advantages are as follows:
# 1. Easy to use
# 2. Computational efficiency
# 3. Model Accuracy
# 4. Feasibility — easy to tune parameters and modify objectives.

# In[86]:


model=XGBRegressor(max_depth=5)


# In[87]:


model.fit(X_train,y_train)


# In[88]:


y_pred=model.predict(X_test)


# In[89]:


print('R2 score using XG Boost= ',r2_score(y_test, y_pred), '/ 1.0')
print('MSE score using XG Boost= ',mean_squared_error(y_test, y_pred), '/ 0.0')


# ## 10. LGBM
# 
# Light GBM is a fast, distributed, high-performance gradient boosting framework based on decision tree algorithm, used for ranking, classification and many other machine learning tasks.
# 
# Since it is based on decision tree algorithms, it splits the tree leaf wise with the best fit whereas other boosting algorithms split the tree depth wise or level wise rather than leaf-wise. So when growing on the same leaf in Light GBM, the leaf-wise algorithm can reduce more loss than the level-wise algorithm and hence results in much better accuracy which can rarely be achieved by any of the existing boosting algorithms. 

# In[90]:



from lightgbm import LGBMRegressor
lgbm=LGBMRegressor(max_depth=5)
lgbm.fit(X_train,y_train)


# In[91]:


y_pred=lgbm.predict(X_test)


# In[92]:


print('R2 score using LGBM = ',r2_score(y_test, y_pred), '/ 1.0')
print('MSE score using LGBM = ',mean_squared_error(y_test, y_pred), '/ 0.0')


# In[93]:


Salesdf.corr()


# ## 11. Creating a neural network
# 
# Multi-layer Perceptron (MLP) is a supervised learning algorithm that learns a function  by training on a dataset, where  is the number of dimensions for input and  is the number of dimensions for output.
# 
# The advantages of Multi-layer Perceptron are:
# 
# -- Capability to learn non-linear models.
# 
# -- Capability to learn models in real-time (on-line learning) using partial_fit.

# In[94]:


# Convert data as np.array
features = np.array(X_train)
#targets = np.array(y_train.reshape(y_train.shape[0],1))
targets = np.array(y_train.values.reshape(y_train.shape[0],1))
features_validation= np.array(X_test)
#targets_validation = np.array(y_test.reshape(y_test.shape[0],1))
targets_validation = np.array(y_test.values.reshape(y_test.shape[0],1))

print(features[:10])
print(targets[:10])


# In[95]:


# Building the model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(1))

# Compiling the model
model.compile(loss = 'mse', optimizer='adam', metrics=['mse']) #mse: mean_square_error
model.summary()


# In[101]:


# Training the model
epochs_tot = 1000
epochs_step = 250
epochs_ratio = int(epochs_tot / epochs_step)
hist =np.array([])

for i in range(epochs_ratio):
    history = model.fit(features, targets, epochs=epochs_step, batch_size=100, verbose=0)
    
    # Evaluating the model on the training and testing set
    print("Step : " , i * epochs_step, "/", epochs_tot)
    score = model.evaluate(features, targets)
    print("Training MSE:", score[1])
    score = model.evaluate(features_validation, targets_validation)
    print("Validation MSE:", score[1], "\n")
    hist = np.concatenate((hist, np.array(history.history['mse'])), axis = 0)#mse: mean_square_error
    
# plot metrics
plt.plot(hist)
plt.show()


# In[102]:


y_pred = model.predict(features_validation, verbose=0)

print('R2 score = ',r2_score(y_test, y_pred), '/ 1.0')
print('MSE score = ',mean_squared_error(y_test, y_pred), '/ 0.0')


# In[103]:


#Lets plot the  first 50 predictions
plt.plot(y_test.as_matrix()[0:50], '+', color ='blue', alpha=0.7)
plt.plot(y_pred[0:50], 'ro', color ='red', alpha=0.5)
plt.show()


# ## Light GBM

# In[104]:


df_train = pd.read_csv(
    'train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 66458909)  # 2016-01-01
)


# In[105]:


df_train.head()


# In[106]:


df_train.shape


# In[107]:


df_train.tail()


# In[108]:


df_test = pd.read_csv(
    "test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]  # , date_parser=parser
).set_index(
    ['store_nbr', 'item_nbr', 'date']
)


# In[109]:


#we want to predict unit sales for last 15 days of Aug 2017
df_test.head()


# In[110]:


items = pd.read_csv(
    "items.csv",
).set_index("item_nbr")


# In[111]:


items.shape


# In[112]:


df_2017 = df_train[df_train.date.isin(
    pd.date_range("2017-05-31", periods=7 * 11))].copy()
del df_train
#dates range from May 31st to 15th Aug~ 77 days


# In[113]:


df_2017.shape


# In[114]:


df_2017.head()


# In[115]:


df_2017.isnull().sum()


# In[116]:


promo_2017_train = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)
#pivots the table and we want last level of index which is date in our case


# In[117]:


promo_2017_train.head()


# In[118]:


promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)


# In[119]:


promo_2017_train.columns


# In[120]:


#Repeat same steps for test dataset
promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)


# In[121]:


promo_2017_test.head()
promo_2017_test.shape


# In[122]:


promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
#Reseting the index same as promo_2017_train
promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)


# In[123]:


del promo_2017_test,promo_2017_train


# In[124]:


df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
df_2017.columns = df_2017.columns.get_level_values(1)


# In[125]:


promo_2017.head()


# In[126]:


items = items.reindex(df_2017.index.get_level_values(1))


# In[127]:


items.head()


# In[128]:


def get_timespan(df, dt, minus, periods):
    return df[
        pd.date_range(dt - timedelta(days=minus), periods=periods)
    ]


# In[129]:


def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({
        "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
        "mean_7_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
        "mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values
    })
    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[
            t2017 + timedelta(days=i)].values.astype(np.uint8)
    if is_train:
        y = df_2017[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    return X


# In[130]:


print("Preparing dataset...")
t2017 = date(2017, 6, 21)
X_l, y_l = [], []
for i in range(4):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(
        t2017 + delta
    )
    X_l.append(X_tmp)
    y_l.append(y_tmp)
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l
X_val, y_val = prepare_dataset(date(2017, 7, 26))
X_test = prepare_dataset(date(2017, 8, 16), is_train=False)


# In[131]:


print("Training and predicting models...")
params = {
    'num_leaves': 2**5 - 1,
    'objective': 'regression_l2',
    'max_depth': 8,
    'min_data_in_leaf': 50,
    'learning_rate': 0.05,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'bagging_freq': 1,
    'metric': 'l2',
    'num_threads': 4
}


# In[132]:


MAX_ROUNDS = 1000
val_pred = []
test_pred = []
cate_vars = []


# In[133]:


for i in range(16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        categorical_feature=cate_vars,
        weight=pd.concat([items["perishable"]] * 4) * 0.25 + 1
    )
    dval = lgb.Dataset(
        X_val, label=y_val[:, i], reference=dtrain,
        weight=items["perishable"] * 0.25 + 1,
        categorical_feature=cate_vars)
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=50
    )
    print("\n".join(("%s: %.2f" % x) for x in sorted(
        zip(X_train.columns, bst.feature_importance("gain")),
        key=lambda x: x[1], reverse=True
    )))
    val_pred.append(bst.predict(
        X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
    test_pred.append(bst.predict(
        X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))


# In[134]:


print("Validation mse:", mean_squared_error(
    y_val, np.array(val_pred).transpose()))


# In[135]:


print("Making submission...")
y_test = np.array(test_pred).transpose()
df_preds = pd.DataFrame(
    y_test, index=df_2017.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)


# In[136]:


submission = df_test[["id"]].join(df_preds, how="left").fillna(0)


# In[137]:


submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.to_csv('lgb.csv', float_format='%.4f', index=None)


# # 12. Conclusion
# We are getting following results on applying data set on different models:
# 
# Model                                    
# R2 Score
# 
# Linear Regression                  
# 0.354
# 
# Decision Tree Regression           
# 0.705
# 
# Extra Tree Regression              
# 0.825
# 
# Random Forest Regression           
# 0.836
# 
# Gradient Boosting Regression       
# 0.836
# 
# XG Boost                           
# 0.797
# 
# LGBM                               
# 0.759

# # 13. References and Attributions
# 
# [1] Cui, G., Wong, M. L., & Lui, H. K. (2006). Machine learning for direct marketing response models:
# Bayesian networks with evolutionary programming.Management Science, 52(4), 597-612
# 
# [2] Taylor, E. L. (2014). Predicting Consumer Behavior. Research World, 2014(46), 67-68
# 
# [3] Morwitz, V. G., Steckel, J. H., & Gupta, A. (2007). When do purchase intentions predict sales?.
# International Journal of Forecasting, 23(3), 347-364
# 
# [4] https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data
# 
# [5] https://en.wikipedia.org/wiki/Xgboost
# 
# [6] https://en.wikipedia.org/wiki/Random_forest
# 
# [7] https://en.wikipedia.org/wiki/Decision_tree
# 
# [8]https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vsxgboost/
# 
# [9] https://www.tutorialspoint.com/sales_forecasting/sales_forecasting_discussion.html
# 
# The text in the document by Source Wikipedia is licensed under CC BY 3.0 https://creativecommons.org/licenses/by/3.0/us/
# 
# The text in the document by Analytics Vidhya is licensed under CC BY 3.0 https://creativecommons.org/licenses/by/3.0/us/
# 
# The text in the document by towards data science is licensed under CC BY 3.0 https://creativecommons.org/licenses/by/3.0/us/
# 
# The code in the document by Source Kaggle is licensed under the MIT License https://opensource.org/licenses/MIT
