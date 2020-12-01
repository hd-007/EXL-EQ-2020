import numpy as np
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
# import dask.dataframe as pd
# from dask_ml.xgboost import XGBClassifier
import seaborn as sns 
import matplotlib.pyplot as plt 
from scipy.stats import norm

dataset = pd.read_csv('EXL_EQ_2020_Train_datasets.csv', delimiter=",")

dataset.head()

#Feature Engineering on Categorical features

dataset.var33.describe()
dataset["var33"].unique()
dataset["var33"].isnull().sum()/(dataset["var33"].count()+dataset["var33"].isnull().sum())
dataset.var33=dataset.var33.astype('category').cat.codes

dataset["var34"].describe()
dataset["var34"].unique()
dataset.var34=dataset.var34.astype('category').cat.codes

dataset["var35"].describe()
dataset["var35"].unique()
dataset.var35=dataset.var35.astype('category').cat.codes

dataset["var36"].describe()
dataset["var36"].unique()
dataset.var36=dataset.var36.astype('category').cat.codes

dataset["var37"].describe()
dataset["var37"].unique()
dataset["var37"].isnull().sum()/(dataset["var37"].count()+dataset["var37"].isnull().sum())
dataset.var37=dataset.var37.replace(np.NaN,'N')
dataset.var37=dataset.var37.astype('category').cat.codes

dataset["var38"].describe()
dataset["var38"].unique()
dataset["var38"].isnull().sum()/(dataset["var38"].count()+dataset["var38"].isnull().sum())
#0.6171733333333334
dataset=dataset.drop(['var38'], axis = 1)

dataset["var39"].describe()
dataset["var39"].unique()
dataset["var39"].isnull().sum()/(dataset["var39"].count()+dataset["var39"].isnull().sum())
#7.333333333333333e-05
dataset.var39=dataset.var39.replace(np.NaN,'Single Housing')
dataset.var39=dataset.var39.astype('category').cat.codes

dataset["var40"].describe()
dataset["var40"].unique()
dataset.var40=dataset.var40.astype('category').cat.codes


dataset["self_service_platform"].describe()
dataset["self_service_platform"].unique()

def score_to_numeric(x):
    if x=='Desktop':
        return 1
    if x=='Mobile App':
        return 2
    if x=='Mobile Web':
        return 3
    if x=='STB':
        return 4
        
dataset['self_service_platform'] = dataset['self_service_platform'].apply(score_to_numeric)
dataset

x = dataset["self_service_platform"].plot.hist(bins=12, alpha=0.5)

dataset.head()

#Feature Engineering on numeric features

dataset["var24"].describe()
dataset.var24=dataset.var24.replace(np.NaN,0)
x = dataset["var24"].plot.hist(bins=12, alpha=0.5)

dataset=dataset.drop(["var30"],axis=1)

corrmat = dataset.corr() 
  
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1)

X = dataset.drop(['self_service_platform'], axis=1)
y = dataset.self_service_platform
from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=5)

categorical_features_indices = np.where(X.dtypes != np.float)[0]


#from dask_ml.xgboost import XGBClassifier

#Import libraries:
import pandas as pd
import numpy as np
# import xgboost as xgb
from xgboost import XGBClassifier
# from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV  #Perforing grid search

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

#XGBOOST
model = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
model.fit(X_train, y_train)

prediction =model.predict(X_validation)
prediction

accuracy = accuracy_score(y_validation, prediction)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#Accuracy: 73.28%

#RANDOM FOREST

import math
from sklearn.ensemble import RandomForestClassifier 
  
# create classifier object 
classifier = RandomForestClassifier( random_state = 0)

classifier.fit(X_train, y_train)

prediction =regressor.predict(X_validation)
prediction

accuracy = accuracy_score(y_validation, prediction)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

Accuracy: 74.64%

dataset_test["prediction"]=prediction
dataset_test.to_csv(r'result.csv', index = False)
