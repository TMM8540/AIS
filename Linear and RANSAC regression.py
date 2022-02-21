#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import sklearn


# In[3]:


df = pd.read_csv('ml_python_2019/data/housing_data',delim_whitespace = True, header = None)


# In[4]:


#df.head()


# In[5]:


col_name = ['CRIM', 'ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df.columns = col_name


# In[6]:


df.head()


# In[9]:


df.describe()
#gives good data to begin statistical analysis
#when std. deviation is much larger than the mean it tells you the data is very noisey
#small std. deviation lets you have more confidence in analysis
#similar means and std. deviations across features tends to be good


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


sns.pairplot(df, height = 1.5)


# In[15]:


col_study = ['CRIM','ZN','INDUS','NOX','RM']
sns.pairplot(df[col_study], height = 3.5)


# In[31]:


col_study2 = ['PTRATIO', 'B','LSTAT','MEDV','ZN']
sns.pairplot(df[col_study2], height = 2.5)
#graphs along diagonal is a plot against itself
#select graph of features plotted against others that look like they might exhibit a pattern for prediction
#feature selection allaows you to run an ML algorithm that will make sense


# Correlation Analysis + Feature Selection

# In[21]:


df.corr()


# In[26]:


plt.figure(figsize = (16,10))
sns.heatmap(df.corr(),annot = True,cmap="YlGnBu")


# In[34]:


sns.heatmap(df[col_study2].corr(),annot = True ,cmap="YlGnBu")
#annot places the values on the chart as well


# In[35]:


pd.options.display.float_format = '{:,.2f}'.format


# In[37]:


df.corr()


# Linear Regression with SciKit Learn

# In[38]:


df.head()


# In[39]:


x=df['RM'].values.reshape(-1,1)
y=df['MEDV'].values


# x- features
# y-target value


# In[41]:


from sklearn.linear_model import LinearRegression


# In[42]:


model = LinearRegression()


# In[43]:


model.fit(x,y)


# In[45]:


print(model.coef_) 
print(model.intercept_)


# In[46]:





# In[50]:


plt.figure(figsize = (10,8))
sns.regplot(x=x,y=y)
plt.xlabel('average rooms per dwelling')
plt.ylabel('average cost per dwelling')


# In[52]:


sns.jointplot(x='RM', y='MEDV', data=df, kind = 'reg', height = 10)


# In[55]:


model.predict(np.array(5).reshape(-1,1))
#prediction for a five room house


# how machine learning works 
# 
#  1. choose a class of a model by importing the correct estimator class 
#  2. choose model hyperparameters by instantiating the class with desired values
#  3. arrange data into features matrix and traget vectors
#  4. fit the data by 
#  5. apply the model to new data
#      predict method is used for supervised learning

# In[57]:


ml2 = LinearRegression()

x = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV'].values

ml2.fit(x,y)

sns.regplot(x=x,y=y)
plt.xlabel('percentile income')
plt.ylabel('average cost per dwelling')

sns.jointplot(x='LSTAT', y='MEDV', data=df, kind = 'reg', height = 10)

model.predict(np.array(17).reshape(-1,1))


# Robust regression
#     code often violates the conditions of linear regression
#     uses RANSAC algorithim

# In[58]:


x = df['RM'].values.reshape(-1,1)
y = df['MEDV'].values


# In[59]:


from sklearn.linear_model import RANSACRegressor


# In[64]:


ransac = RANSACRegressor()


# In[65]:


ransac.fit(x,y)


# In[66]:


inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)


# In[68]:


np.arange(3, 10, 1)


# In[71]:


line_x = np.arange(3,10,1)
line_y = ransac.predict(line_x.reshape(-1,1))


# In[81]:


plt.figure(figsize = (12,8))
plt.scatter(x[inlier_mask],y[inlier_mask],c = 'blue', marker = 'o', label = 'Inliers')
plt.scatter(x[outlier_mask],y[outlier_mask], c = 'brown', marker = 'o', label = 'Outliers')   
plt.xlabel('average rooms per dwelling')
plt.ylabel('Median value of home in 1000s of dollars')
plt.plot(line_x, line_y, color = 'red')


# In[83]:


x = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV'].values

ransac2 = RANSACRegressor()

ransac2.fit(x,y)


# In[103]:


inlier_arr = ransac2.inlier_mask_
outlier_arr = np.logical_not(inlier_arr)

x_axis = np.arange(0,35,1)
y_axis = ransac2.predict(x_axis.reshape(-1,1))


# In[104]:


plt.figure(figsize = (12,8))
plt.scatter(x[inlier_arr],y[inlier_arr],c = 'blue', marker = 'o', label = 'Inliers')
plt.scatter(x[outlier_arr],y[outlier_arr], c = 'brown', marker = 'o', label = 'Outliers')   
plt.xlabel('LSTAT')
plt.ylabel('Median value of home in 1000s of dollars')
plt.plot(x_axis, y_axis, color = 'red')


# Evaluating the performance of a regression model
# 

# In[105]:


from sklearn.model_selection import train_test_split


# In[106]:


x = df.iloc[: , :-1].values


# In[107]:


y = df['MEDV'].values


# In[108]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = .2, random_state = 0)
#settting random_state to zero ensures that data will have same starting points
#allows for the comparison of data across different training models


# In[111]:


lr = LinearRegression()
lr.fit(x_train,y_train)
LinearRegression(fit_intercept = True, n_jobs = None, normalize = False)

y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)


# 1. Residual Analysis
# first look at the residuals to see if there is an underlying pattern to the data
# 
# a good residual analysis will show randomly distributed points with no significant patterns

# In[115]:


plt.figure(figsize = (12,8))
plt.scatter(y_train_pred, y_train_pred-y_train, c='blue', marker = 'o', label = 'Training Data')
plt.scatter(y_test_pred, y_test_pred-y_test, c = 'orange', marker = 'o', label = 'Testing data')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = -10, xmax = 50, lw = 2, color = 'k')
plt.xlim([-10,50])


# Mean Squared Errors

# In[118]:


from sklearn.metrics import mean_squared_error

mean_squared_error(y_train, y_train_pred)


# In[117]:


mean_squared_error(y_test, y_test_pred)


# Coefficient of determination

# In[119]:


from sklearn.metrics import r2_score

r2_score(y_train, y_train_pred)


# In[120]:


r2_score(y_test, y_test_pred)


# 
