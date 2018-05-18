
import numpy as np
import pdb
import itertools as it
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from sklearn.metrics import f1_score
import pandas as pd
import os
import sys
import datetime
import glob
import re
import graphviz
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from patsy import dmatrices
from sklearn.metrics import roc_auc_score
from sklearn import tree
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import json

"""
    Homework 2: ML Pipeline
    Looking at data regarding credit distress and trying to predict who will
    have credit distress in the next two years. Below is a pipeline of various
    ml tools that can be used to analyze, explore, and clean data.
    
    author: Yuliana Zamora
    Date: April 17, 2018
    """

# Reading csv data from file - must be in same directory
def load_data(csv_file,nrows=None):
    
    return pd.read_csv(csv_file,nrows=nrows)

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

#Creating dictionary with no repeated column items
def column_dic(data_frame):
    dict = {line[:1]:line[1:].split()[0] for line in data_frame}
    print (dict)



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

#Nearest Neighbors -
def knearest(data_frame,train, test):
    #data_frame = data_frame.reshape(-1,1)
    X = data_frame[train].reshape(-1,1)
    Y = data_frame[test].reshape(-1,1)
    X_train = X[:100]
    Y_train = Y[:100]
    X_validate = X[100:]
    Y_validate = Y[100:]
    neighbor = KNeighborsClassifier(n_neighbors = 2, weights ='uniform')
    neighbor.fit(X_train, Y_train)
    predicted = neighbor.predict(X_validate)
    print (predicted)

def merging_data(dataframe_1,dataframe_2):
    return pd.merge(dataframe_1,dataframe_2)

def merging_data2(dataframe_1,dataframe_2):
    dataframe_1['fully_funded'] = 1
    return dataframe_1

def get_combos(param_grid_dict):

    all = sorted(param_grid_dict)
    all_combos=[]
    combinations = it.product(*(param_grid_dict[Name] for Name in all))
    for i in combinations:
        lil_combo = {}
        for iter,key in enumerate(all):
           lil_combo[key] = i[iter]
        all_combos.append(lil_combo)

    return (all_combos)

def get_metrics(y_pred, val_Y):
    metric_results ={}
    
    #loss = f1_score(y_pred,val_Y)
    perf_metrics = [.01,.02,.05,.10,.20,.30,.50]
    for i in perf_metrics:
        #pdb.set_trace()
        metric_results["precision at" + str([i])] = precision_score(val_Y, y_pred[:,0] > 1 - i)
        metric_results["recall at" + str([i])] = recall_score(val_Y, y_pred[:,0] > 1 - i)
        metric_results["F1 at" + str([i])] = f1_score(val_Y, y_pred[:,0] > 1 - i)
        
    metric_results["ROC"] = roc_auc_score(val_Y, y_pred[:,0])
    prec,rec,thresh = precision_recall_curve(val_Y, y_pred[:,0])
    metric_results["PREC"] = prec.tolist()
    metric_results["REC"] = rec.tolist()
    metric_results["THRESH"] = thresh.tolist()
    return (metric_results)

#plotting precisison and recal graphs, input one column for y_pred in class_comp method
def plot_precision_recall(val_Y,y_pred,model_name,output_type):
    #pdb.set_trace()
    prec,rec,thresh = precision_recall_curve(val_Y, y_pred)
    prec = prec[:-1]
    recall_curve = rec[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_pred)
    for value in thresh:
        num_above_thresh = len(y_pred[y_pred>=value])
        pct_above_thresh = num_above_thresh / float(len(y_pred))

        if pct_above_thresh <= 1:
            pct_above_per_thresh.append(pct_above_thresh)
        else:
            pdb.set_trace()

    
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, prec, 'b')
    print("PLOTTING STUFF")
    print(pct_above_per_thresh)
    print(prec[:-1])
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    if (output_type == 'save'):
        plt.savefig(name)
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()




def temp_val(data_frame,target,features):
    
    models_params = {
        LogisticRegression: {'C':[10**-1,10**-2,10**-3],'penalty':['l1','l2']},
        KNeighborsClassifier:{'n_neighbors':[5,10,25,100], 'p':[1,2,3],'n_jobs':[2]},
        DecisionTreeClassifier:{'max_depth': [5,10,15],'min_samples_leaf':[2,5,10]},
        RandomForestClassifier:{'n_estimators':[100] , 'criterion':['gini','entropy'], 'max_features':['sqrt','log2'] , 'max_depth':[5,10],'n_jobs':[4], 'min_samples_leaf':[10,50,100]},
        GradientBoostingClassifier:{'learning_rate':[.1,.01],'n_estimators':[100] ,'max_features':['sqrt','log2'] , 'max_depth':[1,2,3]},
        BaggingClassifier:{'max_samples':[.1,.25,.65], 'n_jobs':[4]},
        #SVC:{'kernel':['linear','rbf'],'gamma':[10,1,.1,.01], 'C':[10,1,.1,.01], 'probability':[True]}
        }
    # start time of our data
    #start_time = '2002-09-13'
    start_time_date = data_frame['date_posted'].min()

    #last date of data including labels and outcomes that we have
    #end_time = '2014-05-12'
    end_time_date = data_frame['date_posted'].max()
    
    #how far out do we want to predict (let's say in months for now)
    prediction_windows = [1]

    #how often is this prediction being made? every day? every month? once a year?
    update_window = 12

    from datetime import date, datetime, timedelta
    from dateutil.relativedelta import relativedelta

    #start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
    #end_time_date = datetime.strptime(end_time, '%Y-%m-%d')

    for prediction_window in prediction_windows:
        print(start_time_date,end_time_date)
        test_end_time = end_time_date
        while (test_end_time >= start_time_date + 2 * relativedelta(months=+prediction_window)):
            test_start_time = test_end_time - relativedelta(months=+prediction_window)
            train_end_time = test_start_time  - relativedelta(days=+1) # minus 1 day
            train_start_time = train_end_time - relativedelta(months=+prediction_window)
            while (train_start_time >= start_time_date ):
                #pdb.set_trace()
                print (train_start_time,train_end_time,test_start_time,test_end_time, prediction_window)
                train_start_time -= relativedelta(months=+prediction_window)
                # call function to get data
                train_set, test_set = extract_train_test_sets(train_start_time, train_end_time, test_start_time, test_end_time,data_frame)
                #pdb.set_trace()
                class_comp(train_set,test_set,target,features,models_params)
                # fit on train data
                # predict on test data
            test_end_time -= relativedelta(months=+update_window)

def extract_train_test_sets(train_start_time, train_end_time, test_start_time, test_end_time, df):
    train_set = df[(df['date_posted'] > train_start_time) & (df['date_posted']<train_end_time)]
    test_set = df[(df['date_posted'] > test_start_time) & (df['date_posted']<test_end_time)]
    return train_set, test_set


def class_comp(train_set,test_set,target,features,models_params):
    out = open("out.txt","a")
    X = train_set[features]
    y = train_set[target]
    metrics = {}
    #validation
    val_X = test_set[features]
    val_Y = test_set[target]
    for m, m_param in models_params.items():
       listofparam = get_combos(m_param)
       print("start training for {0}".format(m))
       out.write("start training for {0}\n".format(m))
       for params in listofparam:
           print (params)
           out.write(json.dumps(params))
           model = m(**params)
           model.fit(X,y)
           #y_pred vector of prob estimates
           #val_y are true values
           y_pred = model.predict_proba(val_X)
           metrics[m] = get_metrics(y_pred,val_Y)
           print("this is valy")
           print (val_Y)
           print("this is y_pred")
           print (y_pred)
           plot_precision_recall(val_Y, y_pred[:,0],model,'show')
           out.write("----------------------------\n")
           out.write("Using %s classifier \n" % models_params)
           out.write(json.dumps(metrics[m]))





