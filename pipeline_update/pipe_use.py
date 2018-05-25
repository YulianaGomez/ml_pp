import pandas as pd
from pipe_tools import *
from credit_pipe import kfold_eval
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
data_frame = clean_miss(data_frame)
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

data_frame_1 = load_data("credit-data.csv")
"""
data_frame_1 = load_data("projects.csv")
data_frame_1['date_posted'] = pd.to_datetime(data_frame_1['date_posted'])
#print(data_frame_1.columns)
data_frame_2 = load_data("outcomes.csv")"""
#data_frame_2['date_posted'] = pd.to_datetime(data_frame_2['date_posted'])
#print (data_frame)
#train_val_splits = [
#        (list(range(100)),list(range(100,200)))]

#data_frame_1['fully_funded'] = np.random.random((30000))>.5
#data_frame_2['fully_funded'] = data_frame_2['fully_funded'] == 't'
"""
print(data_frame_1.columns)
merged_data = merging_data(data_frame_1,data_frame_2)
small_df =merged_data.head(100000)
#print(small_df.head())
data_specific_df = small_df[(small_df['date_posted'] > '2011-01-01') & (small_df['date_posted']<'2013-12-31')]
binarycols =['at_least_1_teacher_referred_donor','fully_funded','at_least_1_green_donation','great_chat','three_or_more_non_teacher_referred_donors','one_non_teacher_referred_donor_giving_100_plus','donation_from_thoughtful_donor', 'school_charter', 'school_magnet','school_year_round', 'school_nlns','teacher_teach_for_america', 'teacher_ny_teaching_fellow', 'is_exciting', 'eligible_double_your_impact_match','school_kipp', 'school_charter_ready_promise', 'eligible_almost_home_match']
#print (small_df.columns)
#print (data_specific_df.head())
#PersonID,SeriousDlqin2yrs,RevolvingUtilizationOfUnsecuredLines,age,zipcode,NumberOfTime30-59DaysPastDueNotWorse,DebtRatio,MonthlyIncome,NumberOfOpenCreditLinesAndLoans,NumberOfTimes90DaysLate,NumberRealEstateLoansOrLines,NumberOfTime60-89DaysPastDueNotWorse,NumberOfDependents
features =['is_exciting','teacher_referred_count','donation_from_thoughtful_donor', 'school_charter', 'school_magnet','school_year_round','at_least_1_green_donation','great_chat','at_least_1_teacher_referred_donor','at_least_1_green_donation','great_chat','three_or_more_non_teacher_referred_donors','one_non_teacher_referred_donor_giving_100_plus','donation_from_thoughtful_donor', 'school_charter', 'school_magnet','school_year_round', 'school_nlns','teacher_teach_for_america', 'teacher_ny_teaching_fellow', 'is_exciting', 'eligible_double_your_impact_match','school_kipp', 'school_charter_ready_promise', 'eligible_almost_home_match']
#features = ['school_latitude', 'school_longitude']

new_db = to_binary(data_specific_df,binarycols)
print (new_db)
#print(to_binary)
new_db['teacher_referred_count'].fillna(new_db['teacher_referred_count'].mean(),inplace=True)
new_db['great_messages_proportion'].fillna(new_db['great_messages_proportion'].mean(),inplace=True)

new_db['non_teacher_referred_count'].fillna(new_db['non_teacher_referred_count'].mean(),inplace=True)
"""
features=['RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']
#features=['RevolvingUtilizationOfUnsecuredLines','SeriousDlqin2yrs']
data_frame_1['MonthlyIncome'].fillna(data_frame_1['MonthlyIncome'].mean(),inplace=True)
data_frame_1['NumberOfDependents'].fillna(data_frame_1['NumberOfDependents'].median(),inplace=True)
#print (data_frame_1.columns)

target = ['SeriousDlqin2yrs']
#print (data_frame_1['SeriousDlqin2yrs'])
kfold_eval(data_frame_1,target,features)
#target = ['fully_funded']
#temp_val(small_df,target,features)
#temp_val(new_db,target,features)
#temp_val(data_frame_1,target,features)



#class_comp(LogisticRegression, train_val_splits, my_data, target,features)


