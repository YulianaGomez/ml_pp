
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import os
import sys
import datetime
import glob
import re
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

"""
    Homework 2: ML Pipeline
    Looking at data regarding credit distress and trying to predict who will
    have credit distress in the next two years. Below is a pipeline of various
    ml tools that can be used to analyze, explore, and clean data.
    
    author: Yuliana Zamora
    Date: April 17, 2018
    """

# Reading csv data from file - must be in same directory
def load_data(csv_file):
    return pd.read_csv(csv_file)

#converts a string that is camelCase into snake_case
#https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
def camel_case(column_name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

#Give data with specific column
def histogram(data_frame):
    sns.distplot(data_frame)
    plt.show()

#Given specific column or row, returns statistical summary
def summary(data_frame):
    return data_frame.describe()

#Creating a correlation heat map from data set where var_name is the
#variable which has the most correlation
def cor_heat(data_frame,var_name):
    corrmat = data_frame.corr()
    k = 12
    cols = corrmat.nlargest(k, var_name)[var_name].index
    cm = np.corrcoef(data_frame[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()

#Scatter plots of desired variables in list
def plotCorr(dataFrame, list):
    sns.set()
    sns.pairplot(dataFrame[list], size = 2.5)
    return plt.show()

#Shows data is missing, we should delete the corresponding variable and pretend it never existed - threshold as parameter
def miss_data(data_frame):
    total = data_frame.isnull().sum().sort_values(ascending=False)
    percent = (data_frame.isnull().sum()/data_frame.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data.head(20)

#Dealing with missing data
def clean_miss(data_frame):
    missing_data = miss_data(data_frame)
    data_frame = data_frame.drop((missing_data[missing_data['Total'] > 1]).index,1)
    data_frame.isnull().sum().max() #just checking that there's no missing data missing...
    return data_frame

#Univariate analysis - scaling data, prints out low range and high range
def scale(data_frame, var_scale):
    data_scaled = StandardScaler().fit_transform(data_frame[var_scale][:,np.newaxis]);
    low_range = data_scaled[data_scaled[:,0].argsort()][:10]
    high_range= data_scaled[data_scaled[:,0].argsort()][-10:]
    print('outer range (low) of the distribution:')
    print(low_range)
    print('\nouter range (high) of the distribution:')
    print(high_range)

#Bivariate analysis
def bivariate(data_frame, var_1,var_2):
    varx = var_1
    vary = var_2
    data = pd.concat([data_frame[varx], data_frame[vary]], axis=1)
    data.plot.scatter(x=varx, y=vary, ylim=(0,100));
    plt.show()

#histogram and normal probability plot
def norm_plot(data_frame,var_name):
    sns.distplot(data_frame[var_name], fit=norm);
    fig = plt.figure()
    res = stats.probplot((data_frame)[var_name], plot=plt)
    plt.show()

#Fill in empty values
def fill_empty(data_frame,var, new_var):
    return data_frame[var].fillna(new_var)

#Discretize continuous variables
def descretize(data_frame, var, num):
    return pd.cut(data_frame[var],num,retbins=True)

#Creating dummy variables from categorical variables
def dummy_var(data_frame, var):
    return pd.get_dummies(data_frame[var])

#Logistic regression = iv, independent variable, var_list - dependent variables
def logReg(data_frame, IV, var_list):
    #organizing variable list to independent and dependent variables
    #taking care of hyphen if first word contains it
    if '-' in var_list[0]:
        formula = IV + "~"+'Q("'+var_list[0]+'")'
    else:
        formula = IV + "~"+var_list[0]
    #taking care of the rest of the potential hyphens
    for i in range(1, len(var_list)):
        if '-' in var_list[i]:
            formula = formula + "+"+'Q("'+var_list[i]+'")'
        else:
            formula = formula + "+"+ var_list[i]
    y, X = dmatrices(formula,data_frame, return_type="dataframe")
    y = np.ravel(y)
    model = LogisticRegression()
    model = model.fit(X, y)
    print (pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_)))))
    return model.score(X,y)

