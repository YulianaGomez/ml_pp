import pandas as pd
from pipe_tools import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


read_d = load_data("credit-data.csv")
#print (read_d)

new_string = camel_case("helloPerson")
#print (new_string)
"""
#print (read_d['age'].describe())
print (summary(read_d['age']))

cor_heat(read_d, 'age')

histogram(read_d['age'])

#missing data
missing_data = miss_data(read_d)

#dealing with missing data
read_d = clean_miss(read_d )
print (miss_data(read_d))

#Scaling
scale(read_d,'age')


#Bivariate
bivariate(read_d,'age', 'DebtRatio')


#histogram and normal probability plot
norm_plot(read_d,'age')
"""
#filling in empty data
print (fill_empty(read_d,read_d.mean()))





