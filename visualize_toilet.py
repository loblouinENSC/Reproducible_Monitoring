#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from scipy import stats


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy import stats

import seaborn as sns

import tkinter as tk
from tkinter import filedialog
from pandas import DataFrame


#Below script for toilet activity

log_entry = 'rule-toilet.csv'
hole_csv = 'days-toilet_failure.csv'
out_file = 'out.csv'
result_file = 'out.csv'
graph_title = 'Visiting the toilet activity'
label_y1 = 'Average duration in seconds'
label_y2 = 'Number of passage per month'
figure = 'toilet.pdf'
folder = 'result'


activity = pd.read_csv(log_entry, delimiter = ';', decimal="," ,
                          names=["date", "annotation", "activity_count", "duration"],
                          parse_dates=["date"], index_col="date")




#days_number_on = [0,0,0,0,0.05,0,0.74,0.85,0.88,0.74,0.74,0.68]

#activity.duration = activity.duration/60
#activity.rename(columns={'duration':'durationMin'}, inplace=True)


#out = activity.resample('M').agg({"activity_count":'sum', "durationMin":['mean', 'sem']})
out = activity.resample('M').agg({"activity_count":'sum', "duration":['mean', 'sem']})


export_csv = out.to_csv (out_file, index = True, header=True, sep = ';')

activity_dataset = pd.read_csv(out_file, delimiter =';', decimal=",",
                       names=["date", "sum", "mean", "sem"],
                       parse_dates=["date"], index_col="date")



activity_dataset = activity_dataset.drop('date')
activity_dataset = activity_dataset.loc[activity_dataset.index.dropna()]

activity_dataset.index = pd.to_datetime(activity_dataset.index)


#idx = pd.date_range('2017-08-31', '2018-08-31', freq='m')

idx = pd.date_range('2017-01-31', '2017-12-31', freq='m')



activity_dataset = activity_dataset.reindex(idx, fill_value=0)


activity_dataset.index = activity_dataset.index.strftime("%Y-%m")
activity_dataset = pd.read_csv(out_file, delimiter =';', decimal=",",
                       names=["date", "sum", "mean", "sem"],
                       parse_dates=["date"], index_col="date")

toilet_hole = pd.read_csv(hole_csv, delimiter =';', decimal=","
                              ,names=["date", "toilet_hole"],
                              parse_dates=["date"], index_col="date")


toilet_hole.index =  toilet_hole.index.strftime("%Y-%m")

#number of days where there was no failure
#toilet_hole = (30 - door_toilet)/35

activity_dataset = activity_dataset.drop('date')
activity_dataset = activity_dataset.loc[activity_dataset.index.dropna()]

activity_dataset.index = pd.to_datetime(activity_dataset.index)

#idx = pd.date_range('2017-08-31', '2018-08-31', freq='m')
idx = pd.date_range('2017-01-31', '2017-12-31', freq='m')


activity_dataset = activity_dataset.reindex(idx, fill_value=0)


activity_dataset.index = activity_dataset.index.strftime("%Y-%m")

result = pd.concat([activity_dataset, toilet_hole], axis=1, join_axes=[activity_dataset.index])


result.index.rename('date', inplace=True)

export_csv = result.to_csv (result_file, index = True, header=False, sep = ';')

result_activity = pd.read_csv(result_file, delimiter =';',
                         names=["date", "sum", "mean", "sem", "width"])
result_activity = result_activity.fillna(0)
result_activity['width'] = (31 - result_activity['width'])/35

#for exits devide the value by 60 to convert from seconds to minutes
result_activity['mean'] = result_activity['mean'] #/60
#result_activity['sem'] = result_activity['sem']/60


result_activity.loc[result_activity['width'] == 0, 'on'] = 0
result_activity.loc[result_activity['width'] > 0, 'on'] = -10
plt.rcParams['figure.figsize'] = [9, 7]
x = np.arange(12)
plt.bar(x , result_activity['mean'], width=result_activity['width'], yerr=result_activity['sem'],
         color="#1f77b4"
        )
plt.bar(x, result_activity['on'], width=result_activity['width'], color='green')
plt.gcf().autofmt_xdate()
plt.title(graph_title)
plt.xticks(x, result_activity.date)
plt.xlabel('Date')
plt.ylabel(label_y1, color="#1f77b4")
plt.tick_params(axis="y", labelcolor="#1f77b4")
plt.twinx()
plt.ylabel(label_y2, color="r")
plt.tick_params(axis="y", labelcolor="r", labelsize = 8)
plt.plot(result_activity['sum'], "-r")
plt.xticks(x, result_activity.date)

sns.set_style("dark")
result_activity
#plt.savefig('/path/'+folder+'/'+figure)
#plt.savefig('/path/toilet.pdf')
#plt.show()
