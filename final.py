
# coding: utf-8

# To remove randomness and generate reproducible results, we use Theano backend instead of TensorFlow and set environment variables as well as all the random seeds as following

# In[ ]:


import os
#print(os.listdir("../input"))
import random as rn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

os.environ['CUDA_VISIBLE_DEVICE'] = ''
os.environ['PYTHONHASHSEED'] = '0'
os.environ['MKL_CBWR'] = 'AUTO'
os.environ['KERAS_BACKEND'] = 'theano'

np.random.seed(1)
rn.seed(1)


# Afterwards, import packages for later use

# In[ ]:


from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import metrics
from keras.layers.normalization import BatchNormalization 
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn import preprocessing
from keras.utils import to_categorical


# Load data

# In[ ]:


train = pd.read_csv('Train_full.csv')


# In[ ]:


test = pd.read_csv('Test_small_features.csv')


# In[ ]:


train.columns.values


# As we can see, "return_1" is missing from the feature set. As it might be useful so we will generate this attribute

# In[ ]:


# Generate return_1
def rate_return(price):
    return price[1:]/price[:-1].values - 1

train['return_1'] = rate_return(train['Close'])
train.loc[0, 'return_1'] = train.loc[1, 'lag_return_1']
test['return_1'] = rate_return(test['Close'])
test.loc[0, 'return_1'] = test.loc[1, 'lag_return_1']


# Intuitively, Close price represent only one point, while High and Low represent the whole interval, hence contain more information than only the Close price. Plus, prediction on moving direction of average high low (AVG = (High + Low)/2) outperforms prediction on Close price. 
# Empirically, we have predicted AVG movement, the accuracy rate is outstanding, around 75% to 80%. 
# Therefore, High and Low are important and should be included.
# 
# The work of Choudhry et.al., 2012 has motivated us incorporating lag returns of order upto 2 into the classification model. Here we choose "return_2" and "return_3" instead of "lag_return_1" and "lag_return_2" because the formers nest the latters and contain more information.

# In[ ]:


feature_set = ['Close', 'High', 'Low', 'return_1', 'return_2', 'return_3']


# In[ ]:


# Checking for missing values
train.isnull().values.sum()


# There is no missing value in the data set

# In[ ]:


train[feature_set].info()


# Outliers must be removed because they can badly affect the model's performance.
# 
# We used the Tukey method (Tukey JW., 1977) to detect ouliers.
# 
# From the feature set above, outliers are detected as observations with at least 2 outlied numerical values.

# In[ ]:


# Outlier detection 
def outliers(df, features, n):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[ (df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step) ].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outlier = list(k for k, v in outlier_indices.items() if v > n)
    return multiple_outlier


# In[ ]:


outlier_to_drop = outliers(train, feature_set, 2)
train = train.drop(outlier_to_drop, axis=0).reset_index(drop=True)


# The range of the data should not be neither too large nor to narrow. Furthermore, data normalization helps to improve the model's performance.

# In[ ]:


# Data preparation

train_y = to_categorical(train['up_down']) # change the last column to one-hot-encoding for training
train_X = train[feature_set]

train_X = preprocessing.scale(train_X)   # normalized data now has zero mean and unit variance

test_X = test[feature_set]

test_X = preprocessing.scale(test_X)   # normalized data now has zero mean and unit varianc


# Applying Neural Network with 2 hidden layers, hyper parameters are chosen by trial and error

# In[ ]:


# Building the model

def create_model(dense1, dense2, batch_size, epochs):
    n_cols = train_X.shape[1] # no of features
    model = Sequential()
    np.random.seed(1)

    model.add(Dense(units = dense1, input_shape = (n_cols,)))  # input layer & first hidden layer
    model.add(BatchNormalization())                          # Batch Normalization for the first hidden layer
    model.add(Activation('relu'))                            # use the activation function 'relu' for the nodes
    model.add(Dense(units = dense2, activation = 'relu'))      # second hidden layer
    model.add(BatchNormalization())                          # Batch Normalization for the second hidden layer
    model.add(Activation('relu'))                            
    model.add(Dense(2, activation = 'softmax'))                # The output layer has two nodes, (1 0) and (0 1) 

    # compile the model
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    #model training
    history = model.fit(train_X[cv_train], train_y[cv_train], batch_size = batch_size, epochs = epochs)
    return model


# Using time series k-fold cross validation

# In[ ]:


# Time series k-fold cross validation

k = 5
kfold = TimeSeriesSplit(n_splits = k)
for cv_train, cv_val in kfold.split(train_X, train_y):
    model = create_model(dense1 = 62, dense2 = 22, batch_size = 256, epochs = 5)
    


# In[ ]:


# Prediction

def prediction(model, X):
    pred = model.predict_classes(X)
    return pred[0]


# Row number of the test set is fed line by line and prediction are written out right after

# In[ ]:


# Main

row = '0'
while True:
    row = input()
    if (row == ''):
        break
    else:
        result = prediction(model, test_X[int(row) - 1][np.newaxis])
        print(result)


# Link to the Github account:
# https://github.com/ducnm24/dsc/tree/master

# References:
# 1. Choudhry, Taufiq & Mcgroarty, Frank & Peng, Ke & Wang, Shiyun. (2012). High-Frequency Exchange-Rate Prediction With An Artificial Neural Network. International Journal of Intelligent Systems in Accounting and Finance Management. 19. 170-178. 10.1002/isaf.1329. 
# 2. Tukey, J. W. (1977). Exploratory data analysis. Reading, Mass: Addison-Wesley Pub. Co.
# 3. Yu, Lean & Wang, Shouyang & Lai, Kin Keung. (2007). Foreign-Exchange-Rate Forecasting With Artificial Neural Networks. 10.1007/978-0-387-71720-3. 
