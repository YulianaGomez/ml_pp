import pandas as pd
from pipe_tools import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from datetime import datetime
import random
warnings.filterwarnings('ignore')

"""
    Homework 2: ML Pipeline
    Looking at data regarding credit distress and trying to predict who will
    have credit distress in the next two years. Below is testing the functions
    created in pipe_tools script
    
    author: Yuliana Zamora
    Date: April 17, 2018
"""



"""
new_string = camel_case("helloPerson")
print (new_string)

#print (data_frame["SeriousDlqin2yrs"].describe())
print (summary(data_frame["DebtRatio"]))

cor_heat(data_frame, "SeriousDlqin2yrs")

categories = ["SeriousDlqin2yrs","NumberOfOpenCreditLinesAndLoans","NumberRealEstateLoansOrLines","age"]
plotCorr(data_frame,categories)

histogram(data_frame["age"])

#missing data
missing_data = miss_data(data_frame)
print (missing_data)

#dealing with missing data
data_frame = clean_miss(data_frame )
print (miss_data(data_frame))

#Discretize data
print(descretize(data_frame, "MonthlyIncome", 4))

#Creating dummy variable
print(dummy_var(data_frame,'MonthlyIncome'))

#Scaling
scale(data_frame,"SeriousDlqin2yrs")


#Bivariate
bivariate(data_frame,'age','DebtRatio')


#histogram and normal probability plot
norm_plot(data_frame,"SeriousDlqin2yrs")


#filling in empty data
print (data_frame["MonthlyIncome"])
print (fill_empty(data_frame,"MonthlyIncome", data_frame["MonthlyIncome"].mean()))


column_variables = ['RevolvingUtilizationOfUnsecuredLines',"SeriousDlqin2yrs",'NumberOfTime30-59DaysPastDueNotWorse',
       'DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']
print (logReg(data_frame, "SeriousDlqin2yrs", column_variables))


knearest(data_frame,'DebtRatio','SeriousDlqin2yrs')
"""
#testing merging data
#print(len(data_frame_1))
#merged_data = merging_data(data_frame_1,data_frame_2)
#merged_data = merging_data2(data_frame_1,data_frame_2)
#my_data = pd.DataFrame(np.random.random((1000,5)),columns=feature_cols)
#print (merged_data.dtypes)
#print (len(merged_data))


data_frame_1 = load_data("projects.csv")
data_frame_1['date_posted'] = pd.to_datetime(data_frame_1['date_posted'])
#print(data_frame_1.columns)
data_frame_2 = load_data("outcomes.csv")
#data_frame_2['date_posted'] = pd.to_datetime(data_frame_2['date_posted'])
#print (data_frame)
#train_val_splits = [
#        (list(range(100)),list(range(100,200)))]

#features =['Poverty','grade_level','students_reached','is_exciting','teacher_referred_count]
features = ['school_latitude', 'school_longitude']

#data_frame_1['fully_funded'] = np.random.random((30000))>.5
data_frame_2['fully_funded'] = data_frame_2['fully_funded'] == 't'
merged_data = merging_data(data_frame_1,data_frame_2)
small_df =merged_data.head(100000)
target = ['fully_funded']
temp_val(small_df,target,features)

#class_comp(LogisticRegression, train_val_splits, my_data, target,features)


