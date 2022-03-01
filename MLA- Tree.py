#!/usr/bin/env python
# coding: utf-8

# ![MLU Logo](../data/MLU_Logo.png)

# # <a name="0">Machine Learning Accelerator - Tabular Data - Lecture 1</a>
# 
# 
# ## Final Project 
# 
# In this notebook, we build a ML model to predict the __Time at Center__ field of our final project dataset.
# 
# 1. <a href="#1">Read the dataset</a> (Given) 
# 2. <a href="#2">Train a model</a> (Implement)
#     * <a href="#21">Exploratory Data Analysis</a>
#     * <a href="#22">Select features to build the model</a>
#     * <a href="#23">Data processing</a>
#     * <a href="#24">Model training</a>
# 3. <a href="#3">Make predictions on the test dataset</a> (Implement)
# 4. <a href="#4">Write the test predictions to a CSV file</a> (Given)
# 
# __Austin Animal Center Dataset__:
# 
# In this exercise, we are working with pet adoption data from __Austin Animal Center__. We have two datasets that cover intake and outcome of animals. Intake data is available from [here](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes/wter-evkm) and outcome is from [here](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes/9t4d-g238). 
# 
# In order to work with a single table, we joined the intake and outcome tables using the "Animal ID" column and created a training.csv, test_features.csv and y_test.csv files. Similar to our review dataset, we didn't consider animals with multiple entries to the facility to keep it simple. If you want to see the original datasets, they are available under data/review folder: Austin_Animal_Center_Intakes.csv, Austin_Animal_Center_Outcomes.csv.
# 
# __Dataset schema:__ 
# - __Pet ID__ - Unique ID of pet
# - __Outcome Type__ - State of pet at the time of recording the outcome
# - __Sex upon Outcome__ - Sex of pet at outcome
# - __Name__ - Name of pet 
# - __Found Location__ - Found location of pet before entered the center
# - __Intake Type__ - Circumstances bringing the pet to the center
# - __Intake Condition__ - Health condition of pet when entered the center
# - __Pet Type__ - Type of pet
# - __Sex upon Intake__ - Sex of pet when entered the center
# - __Breed__ - Breed of pet 
# - __Color__ - Color of pet 
# - __Age upon Intake Days__ - Age of pet when entered the center (days)
# - __Time at Center__ - Time at center (0 = less than 30 days; 1 = more than 30 days). This is the value to predict. 
# 

# In[109]:


#%pip install -q -r ../requirements.txt


# ## 1. <a name="1">Read the datasets</a> (Given)
# (<a href="#0">Go to top</a>)
# 
# Let's read the datasets into dataframes, using Pandas.

# In[110]:


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")
  
training_data = pd.read_csv('../data/final_project/training.csv')
test_data = pd.read_csv('../data/final_project/test_features.csv')
test_data = test_data.drop(labels=range(0,1), axis=0)

print('The shape of the training dataset is:', training_data.shape)
print('The shape of the test dataset is:', test_data.shape)


# ## 2. <a name="2">Train a model</a> (Implement)
# (<a href="#0">Go to top</a>)
# 
#  * <a href="#21">Exploratory Data Analysis</a>
#  * <a href="#22">Select features to build the model</a>
#  * <a href="#23">Data processing</a>
#  * <a href="#24">Model training</a>
# 
# ### 2.1 <a name="21">Exploratory Data Analysis</a> 
# (<a href="#2">Go to Train a model</a>)
# 
# We look at number of rows, columns and some simple statistics of the dataset.

# In[111]:


print(training_data.info())
training_data.head(5)
      


# In[112]:


model_features = training_data.columns.drop('Time at Center')
model_target = 'Time at Center'

print('Model features: ', model_features)
print('Model target: ', model_target)


# In[113]:


import numpy as np
numerical_features_all = training_data[model_features].select_dtypes(include=np.number).columns
print('Numerical columns:',numerical_features_all)

print('')

categorical_features_all = test_data[model_features].select_dtypes(include='object').columns
print('Categorical columns:',categorical_features_all)


# In[114]:


print(training_data[model_target].value_counts())

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

training_data[model_target].value_counts().plot.bar()
plt.show()
#heavily concentrated towards case 1

for c in categorical_features_all:
    if len(training_data[c].value_counts()) < 50:
        print(c)
        training_data[c].value_counts().plot.bar()
        plt.show()


# In[115]:


training_data.isna().sum()


# In[116]:


training_data.describe()


# In[ ]:





# ### 2.2 <a name="22">Select features to build the model</a> 
# (<a href="#2">Go to Train a model</a>)
# 

# In[117]:


numerical_features = ['Age upon Intake Days']

categorical_features = ['Outcome Type', 'Sex upon Outcome', 'Intake Type', 'Intake Condition', 'Pet Type', 
                                     'Sex upon Intake','Age upon Intake Days']

text_features = ['Breed']

for c in numerical_features: 
    print(c)
    print(training_data[c].value_counts(bins=10, sort=False))
    plt.show()


# In[118]:


threshold = 2/10
print((training_data.isna().sum()/len(training_data.index)))
columns_to_drop = training_data.loc[:,list(((training_data.isna().sum()/len(training_data.index))>=threshold))].columns    
print(columns_to_drop)

training_data_columns_dropped = training_data.drop(columns_to_drop, axis = 1)  
training_data_columns_dropped.head()


# In[119]:


training_data_cl = training_data_columns_dropped.dropna()
training_data_cl.isna().sum()


# ### 2.3 <a name="23">Data Processing</a> 
# (<a href="#2">Go to Train a model</a>)
# 

# In[ ]:





# ### 2.4 <a name="24">Model training</a> 
# (<a href="#2">Go to Train a model</a>)
# 

# In[120]:


from sklearn.utils import shuffle

class_0_no = training_data[training_data[model_target] == 0]
class_1_no = training_data[training_data[model_target] == 1]

upsampled_class_0_no = class_0_no.sample(n=len(class_1_no), replace=True, random_state=42)

training_data = pd.concat([class_1_no, upsampled_class_0_no])
training_data = shuffle(training_data)

print('Training set shape:', training_data.shape)

print('Class 1 samples in the training set:', sum(training_data[model_target] == 1))
print('Class 0 samples in the training set:', sum(training_data[model_target] == 0))


# In[121]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


numerical_processor = Pipeline([
    ('num_imputer', SimpleImputer(strategy='mean')),
    ('num_scaler', MinMaxScaler()) 
                                ])
                  

categorical_processor = Pipeline([
    ('cat_imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
    ('cat_encoder', OneHotEncoder(handle_unknown='ignore')) 
                                ])

text_processor_0 = Pipeline([
    ('text_vect_0', CountVectorizer(binary=True, max_features=50))])
 
# Combine all data preprocessors from above (add more, if you choose to define more!)
# For each processor/step specify: a name, the actual process, and finally the features to be processed
data_preprocessor = ColumnTransformer([
    ('numerical_pre', numerical_processor, numerical_features),
    ('categorical_pre', categorical_processor, categorical_features),
    ('text_pre_0', text_processor_0, text_features[0]),
                                    ]) 

### PIPELINE ###
################

# Pipeline desired all data transformers, along with an estimator at the end
# Later you can set/reach the parameters using the names issued - for hyperparameter tuning, for example
pipeline = Pipeline([
    ('data_preprocessing', data_preprocessor),
    ('dt', DecisionTreeClassifier())
                    ])

# Visualize the pipeline
# This will come in handy especially when building more complex pipelines, stringing together multiple preprocessing steps
from sklearn import set_config
set_config(display='diagram')
pipeline


# ## 3. <a name="3">Make predictions on the test dataset</a> (Implement)
# (<a href="#0">Go to top</a>)
# 
# Use the test set to make predictions with the trained model.

# In[122]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


X_train = training_data[model_features]
y_train = training_data[model_target]


pipeline.fit(X_train, y_train)


train_predictions = pipeline.predict(X_train)
print(confusion_matrix(y_train, train_predictions))
print(classification_report(y_train, train_predictions))
print("Accuracy (training):", accuracy_score(y_train, train_predictions))


X_test = test_data[model_features]
y_test = pd.read_csv('../data/final_project/y_test.csv')


test_predictions = pipeline.predict(X_test)
print(confusion_matrix(y_test, test_predictions))
print(classification_report(y_test, test_predictions))
print("Accuracy (test):", accuracy_score(y_test, test_predictions))


# In[ ]:





# In[ ]:




