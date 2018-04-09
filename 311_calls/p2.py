import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import json
import requests
import sys

"""
    Homework 1: Diagnostic
    Looking at 311 requests from the Chicago Open Data Portal and
    census API's for analysis of Chicago communities
    
    author: Yuliana Zamora
    Date: April 3, 2018

"""
class dataCounter():
    def __init__(self):
        self.child15 = {"Far North":0,"Northwest":0,"North":0,"West":0,"Central":0,"South":0,"Southwest":0,"Far Southwest":0,"Far Southeast":0}
        self.child16 = {"Far North":0,"Northwest":0,"North":0,"West":0,"Central":0,"South":0,"Southwest":0,"Far Southwest":0,"Far Southeast":0}
        self.bach15 = {"Far North":0,"Northwest":0,"North":0,"West":0,"Central":0,"South":0,"Southwest":0,"Far Southwest":0,"Far Southeast":0}
        self.bach16 = {"Far North":0,"Northwest":0,"North":0,"West":0,"Central":0,"South":0,"Southwest":0,"Far Southwest":0,"Far Southeast":0}
        self.mom15 = {"Far North":0,"Northwest":0,"North":0,"West":0,"Central":0,"South":0,"Southwest":0,"Far Southwest":0,"Far Southeast":0}
        self.mom16 = {"Far North":0,"Northwest":0,"North":0,"West":0,"Central":0,"South":0,"Southwest":0,"Far Southwest":0,"Far Southeast":0}

def main():
    ####--Populating demo data--####
    #Number of children on govt assistant, bachelors degrees, children in single mom homes
    processed15 = glob.glob("2015.json")
    processed16 = glob.glob("2016.json")
    if len(processed15) > 0 and len(processed16) > 0:
        json_data= open("2015.json", "r")
        demo_15 = json.load(json_data)
        json_data.close()

        json_data= open("2016.json", "r")
        demo_16 = json.load(json_data)
        json_data.close()
    else:
        for year in range(2015,2017):
            url = "https://api.census.gov/data/"+str(year)+"/acs/acs5/subject?get=NAME,S0901_C01_031E,S1501_C01_012E,S0901_C04_001E&for=zip%20code%20tabulation%20area:*"
            demo_data = requests.get(url,allow_redirects=True)
            file_name = str(year) +".json"
            open(file_name, 'wb').write(demo_data.content)
            if year == 2015:
                json_data= open("2015.json", "r")
                demo_15 = json.load(json_data)
                json_data.close()
            else:
                json_data= open("2016.json", "r")
                demo_16 = json.load(json_data)
                json_data.close()

    ###--setting specific regions with their corresponding zipcodes--###
    #http://chicago-zone.blogspot.com/2014/03/chicago-zip-code-map-locate-chicago.html
    zip_dict = {"Far North" : [60626,60645, 60659, 60660,60640,60625,60630,60631,60656], \
                "Northwest" : [60618,60634, 60641,60607,60639], \
                "North" : [60618, 60613,60657, 60613,60614, 60610,60647], \
                "West" :[60651, 60622,60612, 60623, 60642,60639, 60644,60624,60612,60607,60608,60616], \
                "Central" : [60610,60601, 60602, 60603, 60604, 60605,60606, 60607, 60661,60616], \
                "South" : [60609,60616,60653,60615,60637,60649,60608,60620,60619], \
                "Southwest" :[60632,60608, 60609,60629,60638,60621,60636], \
                "Far Southwest" : [60652,60620,60643,60655], \
                "Far Southeast" : [60619,60617,60628,60643,60633,60827,60633,60638] }

    # Create object to store the counters
    datacnt = dataCounter()

    #Populate data for 2015
    for key, val in zip_dict.items():
        for i in range(1, len(demo_15)):
            zipCode = int(demo_15[i][4])
            if zipCode in val:
                addval=[0, 0, 0]
                for j in range(1,4):
                    if demo_15[i][j] != None:
                        if j==1: addval[j-1] = float(demo_15[i][j])
                        else: addval[j-1] = int(demo_15[i][j])
                datacnt.child15[key] += addval[0]
                datacnt.bach15[key] += addval[1]
                datacnt.mom15[key] += addval[2]

    #Populate data for 2016
    for key, val in zip_dict.items():
        for i in range(1, len(demo_16)):
            zipCode = int(demo_16[i][4])
            if zipCode in val:
                addval=[0, 0, 0]
                for j in range(1,4):
                    if demo_16[i][j] != None:
                        if j==1: addval[j-1] = float(demo_16[i][j])
                        else: addval[j-1] = int(demo_16[i][j])
                datacnt.child16[key] += addval[0]
                datacnt.bach16[key] += addval[1]
                datacnt.mom16[key] += addval[2]

    fig, ax = plt.subplots()
    N = len(datacnt.child16.keys())
    ind = np.arange(N)
    width = 0.35
    setting='mom'
    if setting == 'child':
        rects1 = ax.bar(ind, datacnt.child15.values(), width)
        rects2 = ax.bar(ind + width, datacnt.child16.values(), width)
    elif setting == 'bach':
        rects1 = ax.bar(ind, datacnt.bach15.values(), width)
        rects2 = ax.bar(ind + width, datacnt.bach16.values(), width)
    elif setting == 'mom':
        rects1 = ax.bar(ind, datacnt.mom15.values(), width)
        rects2 = ax.bar(ind + width, datacnt.mom16.values(), width)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Chicago Communities')
    ax.set_title('Number of Children in Single mom Households in City of Chicago Community (2015-2016)')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(datacnt.mom16.keys())
        
    ax.legend((rects1[0], rects2[0]), ('2015', '2016'))

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

if __name__ == '__main__':
    main()
