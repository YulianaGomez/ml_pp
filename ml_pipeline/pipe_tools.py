
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


# Reading csv data from file - must be in same directory
def load_data(data):
    return pd.read_csv(data)

#converts a string that is camelCase into snake_case
#https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
def camel_case(column_name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

#Give data with specific column
def histogram(data):
    sns.distplot(data)
    plt.show()

#Given specific column or row, returns statistical summary
def summary(data):
    return data.describe()

#Creating a correlation heat map from data set where var_name is the the
#variable which has the most correlation
def cor_heat(input_data,var_name):
    corrmat = input_data.corr()
    k = 12
    cols = corrmat.nlargest(k, var_name)[var_name].index
    cm = np.corrcoef(input_data[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()

#Shows if more than 15% of the data is missing, we should delete the corresponding variable and pretend it never existed
def miss_data(input_data):
    total = input_data.isnull().sum().sort_values(ascending=False)
    percent = (input_data.isnull().sum()/input_data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data.head(20)

#Dealing with missing data
def clean_miss(input_data):
    missing_data = miss_data(input_data)
    input_data = input_data.drop((missing_data[missing_data['Total'] > 1]).index,1)
    input_data.isnull().sum().max() #just checking that there's no missing data missing...
    return input_data

#Univariate analysis - scaling data, prints out low range and high range
def scale(input_data, var_scale):
    data_scaled = StandardScaler().fit_transform(input_data[var_scale][:,np.newaxis]);
    low_range = data_scaled[data_scaled[:,0].argsort()][:10]
    high_range= data_scaled[data_scaled[:,0].argsort()][-10:]
    print('outer range (low) of the distribution:')
    print(low_range)
    print('\nouter range (high) of the distribution:')
    print(high_range)

#Bivariate analysis
def bivariate(input_data, var_scale,var_2):
    varx = var_scale
    vary = var_2
    data = pd.concat([input_data[varx], input_data[vary]], axis=1)
    data.plot.scatter(x=varx, y=vary, ylim=(0,16000));
    plt.show()

#histogram and normal probability plot
def norm_plot(input_data,var_name):
    sns.distplot(input_data[var_name], fit=norm);
    fig = plt.figure()
    res = stats.probplot((input_data)[var_name], plot=plt)
    plt.show()
