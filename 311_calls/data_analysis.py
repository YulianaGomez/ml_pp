
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import os
import sys
import datetime
import glob

def plot_data():
    cwd = os.getcwd()
    #graffiti = pd.read_csv("/home/parallels/Desktop/Parallels Shared Folders/ml_pp/311_Service_Requests_Graffiti_Removal.csv",dtype=object)
    #graffiti = pd.read_csv("311_Service_Requests_Graffiti_Removal.csv",dtype=object)
    #lights = pd.read_csv("/home/parallels/Desktop/Parallels Shared Folders/ml_pp/311_Service_Requests_Alley_Lights_Out.csv",dtype=object)
    #vacancy = pd.read_csv("/home/parallels/Desktop/Parallels Shared Folders/ml_pp/311_Service_Requests_Vacant.csv",dtype=object)

    #graffiti['Creation Date'] =  pd.to_datetime(graffiti['Creation Date'])
    #graffiti['Completion Date'] =  pd.to_datetime(graffiti['Completion Date'])


###--Graffiti over time--##
#graf_rate = graffiti['Type of Service Request'].groupby([graffiti['Creation Date'].dt.year]).agg({'count'})
#print(graf_rate)
    #graf_rate.hist()
    #plt.show()
###--Vacancy over time--##
#vac_rate = vacancy['SERVICE REQUEST TYPE'].groupby([vacancy['DATE SERVICE REQUEST WAS RECEIVED'].dt.month]).agg({'count'})


    start_date = '2017-01-01'
    end_date = '2017-12-31'
    #idx = pd.date_range('2017-01-01','2017-12-31')
    
    ####---Vacancy---###
    vacancy = pd.read_csv("311_Service_Requests_Vacant.csv",dtype=object)
    vprocessed = glob.glob("vacancy_count.csv")
    if len(vprocessed) > 0:
        vac_rate = pd.read_csv(vprocessed[0])
    else:
        # Get 2017 only (https://stackoverflow.com/questions/29370057/select-dataframe-rows-between-two-dates/41802199)
        vacancy['DATE SERVICE REQUEST WAS RECEIVED'] = pd.to_datetime(vacancy['DATE SERVICE REQUEST WAS RECEIVED'])
        mask = (vacancy['DATE SERVICE REQUEST WAS RECEIVED'] >= start_date) & (vacancy['DATE SERVICE REQUEST WAS RECEIVED'] <= end_date)
        vacancy = vacancy.loc[mask]
        vac_rate = vacancy['SERVICE REQUEST TYPE'].groupby([vacancy['DATE SERVICE REQUEST WAS RECEIVED'].dt.month]).agg({'count'})
        vac_rate = pd.DataFrame(vac_rate).reset_index()
        vac_rate.to_csv("vacancy_count.csv")
    ####---Lights---###
    lights = pd.read_csv("311_Service_Requests_Alley_Lights_Out.csv",dtype=object)
    lprocessed = glob.glob("lights_count.csv")
    if len(lprocessed) > 0:
        lights_rate = pd.read_csv(lprocessed[0])
    else:
        # Get 2017 only
        lights['Creation Date'] = pd.to_datetime(lights['Creation Date'])
        mask = (lights['Creation Date'] >= start_date) & (lights['Creation Date'] <= end_date)
        lights = lights.loc[mask]
        lights_rate = lights['Type of Service Request'].groupby([lights['Creation Date'].dt.month]).agg({'count'})
        lights_rate = pd.DataFrame(lights_rate).reset_index()
        lights_rate.to_csv("lights_count.csv")

    ####---Graffiti---###
    graffiti = pd.read_csv("311_Service_Requests_Graffiti_Removal.csv",dtype=object)
    gprocessed = glob.glob("graffiti_count.csv")
    if len(gprocessed) > 0:
        graffiti_rate = pd.read_csv(gprocessed[0])
    else:
        # Get 2017 only
        graffiti['Creation Date'] = pd.to_datetime(graffiti['Creation Date'])
        mask = (graffiti['Creation Date'] >= start_date) & (graffiti['Creation Date'] <= end_date)
        graffiti = graffiti.loc[mask]
        graffiti_rate = graffiti['Type of Service Request'].groupby([graffiti['Creation Date'].dt.month]).agg({'count'})
        graffiti_rate = pd.DataFrame(graffiti_rate).reset_index()
        graffiti_rate.to_csv("graffiti_count.csv")


    ax = vac_rate.plot(x='DATE SERVICE REQUEST WAS RECEIVED',y='count', label='Vacancy')
    lights_rate.plot(x='Creation Date',y='count',ax=ax, label='Lights')
    graffiti_rate.plot(x='Creation Date',y='count',ax=ax, label='Graffiti')
    ax.set_title('Rate of calls over time in 2017')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    plt.show()


"""
###--Graffiti year vs zipcode--##
    graf_dates = graffiti['Creation Date'].groupby([graffiti['Creation Date'].dt.year, graffiti['ZIP Code']]).agg({'count'})

###--Graffiti Response Time--##
    graffiti['Response Time'] = (graffiti['Completion Date'] -graffiti['Creation Date'])
    graf_completion_rate = graffiti['Response Time'].groupby([graffiti['Response Time'], graffiti['ZIP Code']]).agg({'count'})

###--General summary--###
    
    graf_desc = graffiti.describe()
    lights_desc = lights.describe()
    vac_desc = vacancy.describe()

    #print graf_desc
    #print graf_completion_rate
    print(graf_rate)
"""

if __name__=='__main__':
	plot_data()
