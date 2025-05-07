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


# Script to visualize sleeping activity
log_entry = 'rule-sleep_quiet.csv'
hole_csv = 'days-bed_failure.csv'
out = 'outSleep.csv'
result = 'result.csv'
graph_title = 'Sleeping activity'
label_y = 'Average duration by day in hour'
figure = 'sleep.png'
folder = 'result'


sleep = pd.read_csv(log_entry, delimiter =';', decimal=",",
                  names=["date", "annotation", "sleep_count", "duration"],
                        parse_dates=["date"], index_col="date")

sleep.duration = sleep.duration/3600
sleep.rename(columns={'duration':'durationHr'}, inplace=True)
sleep.resample('M').agg({"durationHr":['mean', 'sem']})

outSleep = sleep.resample('M').agg({"durationHr":['mean', 'sem']})

export_csv = outSleep.to_csv (out, index = True,
                         header=True, sep = ';')

rest = pd.read_csv(out, delimiter =';', decimal=",",
                       names=["date", "mean", "sem"], parse_dates=["date"], index_col="date")


rest = rest.drop('date')
rest = rest.loc[rest.index.dropna()]

rest.index = pd.to_datetime(rest.index)

#idx = pd.date_range('2017-08-31', '2018-08-31', freq='m')
idx = pd.date_range('2017-01-31', '2017-12-31', freq='m')
rest = rest.reindex(idx, fill_value=0)

rest.index = rest.index.strftime("%Y-%m")
rest = pd.read_csv(out, delimiter =';', decimal=",",
                       names=["date", "mean", "sem"],
                       parse_dates=["date"], index_col="date")

bed_failure = pd.read_csv(hole_csv,delimiter =';', decimal=","
                              ,names=["date", "bed_failure"],
                              parse_dates=["date"], index_col="date")

bed_failure.index =  bed_failure.index.strftime("%Y-%m")

#number of days where there was no failure
#bed_failure = (30 - bed_failure)/35

rest = rest.drop('date')
rest = rest.loc[rest.index.dropna()]

rest.index = pd.to_datetime(rest.index)


#idx = pd.date_range('2017-08-31', '2018-08-31', freq='m')


idx = pd.date_range('2017-01-31', '2017-12-31', freq='m')



rest = rest.reindex(idx, fill_value=0)


rest.index = rest.index.strftime("%Y-%m")

resultSleep = pd.concat([rest, bed_failure], axis=1, join_axes=[rest.index])


resultSleep.index.rename('date', inplace=True)

export_csv = resultSleep.to_csv (result, index = True, header=False, sep = ';')

result = pd.read_csv(result, delimiter =';',
                         names=["date", "mean", "sem", "width", "on"])

result = result.fillna(0)
result['width'] = (30 - result['width'])/35

if result['width'].empty:
    result['on'] = 0
else:
    result['on'] = -1


plt.rcParams['figure.figsize'] = [9, 7]
x = np.arange(12)
plt.bar(x, result['mean'], width=result['width'], yerr=result['sem'])
plt.bar(x, result['on'], width=result['width'], color='green')
plt.gcf().autofmt_xdate()
plt.title(graph_title)
plt.xticks(x, result.date)
plt.xlabel('Date')
plt.ylabel(label_y)
plt.tick_params(axis="y")






result
#plt.savefig('/path/'+folder+'/'+figure)
#plt.show()
#plt.savefig('/path/sleep.png')
#sleep13012.index
#df[df.sleep != 1].resample('M').agg({"dureeH":'mean'})
