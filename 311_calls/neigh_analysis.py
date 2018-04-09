
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

    zip_dict = {"Far North" : [60626,60645, 60659, 60660,60640,60625,60630,60631,60656], \
                "Northwest" : [60618,60634, 60641,60607,60639], \
                "North" : [60618, 60613,60657, 60613,60614, 60610,60647], \
                "West" :[60651, 60622,60612, 60623, 60642,60639, 60644,60624,60612,60607,60608,60616], \
                "Central" : [60610,60601, 60602, 60603, 60604, 60605,60606, 60607, 60661,60616], \
                "South" : [60609,60616,60653,60615,60637,60649,60608,60620,60619], \
                "Southwest" :[60632,60608, 60609,60629,60638,60621,60636], \
                "Far Southwest" : [60652,60620,60643,60655], \
                "Far Southeast" : [60619,60617,60628,60643,60633,60827,60633,60638] }

    start_date = '2016-01-01'
    end_date = '2016-12-31'
    #idx = pd.date_range('2017-01-01','2017-12-31')

    temp_dict = {"Far North" : 0, \
                "Northwest" : 0, \
                "North" : 0, \
                "West" : 0, \
                "Central" : 0, \
                "South" : 0, \
                "Southwest" : 0, \
                "Far Southwest" : 0, \
                "Far Southeast" : 0 }
    vac_dict = temp_dict.copy()
    lights_dict = temp_dict.copy()
    graffiti_dict = temp_dict.copy()

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
        #vac_rate = vacancy['SERVICE REQUEST TYPE'].groupby([vacancy['DATE SERVICE REQUEST WAS RECEIVED'].dt.month,vacancy['ZIP CODE']]).agg({'count'})
        vac_rate = vacancy['SERVICE REQUEST TYPE'].groupby([vacancy['ZIP CODE']]).agg({'count'})
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
        #lights_rate = lights['Type of Service Request'].groupby([lights['Creation Date'].dt.month,lights['ZIP Code']]).agg({'count'})
        lights_rate = lights['Type of Service Request'].groupby([lights['ZIP Code']]).agg({'count'})
        lights_rate = pd.DataFrame(lights_rate).reset_index()
        lights_rate.to_csv("lights_count.csv")

    """
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
        #graffiti_rate = graffiti['Type of Service Request'].groupby([graffiti['Creation Date'].dt.month,lights['ZIP Code']]).agg({'count'})
        graffiti_rate = graffiti['Type of Service Request'].groupby([lights['ZIP Code']]).agg({'count'})
        graffiti_rate = pd.DataFrame(graffiti_rate).reset_index()
        graffiti_rate.to_csv("graffiti_count.csv")
    """

    # Compile info about vacancies
    for index, row in vac_rate.iterrows():
        for key, val in zip_dict.items():
            if row['ZIP CODE'] in val:
                vac_dict[key] += int(row['count'])

    # Compile info about lights
    for index, row in lights_rate.iterrows():
        for key, val in zip_dict.items():
            if row['ZIP Code'] in val:
                lights_dict[key] += int(row['count'])
    """
    # Copile info about graffiti
    for index, row in graffiti_rate.iterrows():
        for key, val in zip_dict.items():
            if row['ZIP Code'] in val:
                graffiti_dict[key] += int(row['count'])
    """

    fig, ax = plt.subplots()
    N = len(vac_dict.keys())
    ind = np.arange(N)
    width = 0.35
    rects1 = ax.bar(ind, vac_dict.values(), width)
    rects2 = ax.bar(ind + width, lights_dict.values(), width)
    #rects3 = ax.bar(ind + 2*width, graffiti_dict.values(), width)

    ax.set_ylabel('Number of Occurrences')
    ax.set_title('2016 Vacancy and Light-outage Reports')
    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(vac_dict.keys())
    ax.set_yscale('log')
    #ax.legend((rects1[0], rects2[0], rects3[0]), ('Vacancy', 'Lights', 'Graffiti'))
    ax.legend((rects1[0], rects2[0]), ('Vacancy', 'Lights'))

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    #autolabel(rects1)
    #autolabel(rects2)

    plt.show()


    #print(vac_rate)
    #ax = vac_rate.plot(x='ZIP CODE',y='count', label='Vacancy')
    #ax = vac_rate['ZIP CODE'].plot.hist(label='Vacancy', alpha=0.5, bins=10000)
    #ax = vac_rate.plot(x='ZIP CODE',y='count', label='Vacancy', kind='bar')
    #lights_rate.plot(x='Creation Date',y='count',ax=ax, label='Lights')
    #graffiti_rate.plot(x='Creation Date',y='count',ax=ax, label='Graffiti')
    #ax.set_title('Vacant and Abandoned Buildings Reported in 2017 by ZIP Code')
    #ax.set_ylabel('Count')
    #ax.set_xlim(60600,60800)
    #ax.set_xlim(60600,60720)
    #ax.set_ylim(0,80)
    #ax.set_yscale('log')
    #plt.show()


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
