#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
from pandas import DataFrame

# Below script for toilet activity
log_entry = 'rule-toilet.csv'
hole_csv = 'days-toilet_failure.csv'
out_file = 'out.csv'
result_file = 'out.csv'
graph_title = 'Visiting the toilet activity'
label_y1 = 'Average duration in seconds'
label_y2 = 'Number of passage per month'
figure = 'toilet.pdf'
folder = 'result'

# Read initial data
activity = pd.read_csv(log_entry, delimiter=';', decimal=",",
                      names=["date", "annotation", "activity_count", "duration"],
                      parse_dates=["date"], index_col="date")

# Process the data
out = activity.resample('ME').agg({"activity_count": 'sum', "duration": ['mean', 'sem']})  

# Export to CSV
export_csv = out.to_csv(out_file, index=True, header=True, sep=';')

# Read processed data back - specify date format to avoid warnings
activity_dataset = pd.read_csv(out_file, delimiter=';', decimal=",",
                             names=["date", "sum", "mean", "sem"],
                             parse_dates=["date"], 
                             date_format="%Y-%m-%d %H:%M:%S",  # Specify the date format
                             index_col="date")

activity_dataset = activity_dataset.drop('date')
activity_dataset = activity_dataset.loc[activity_dataset.index.dropna()]
activity_dataset.index = pd.to_datetime(activity_dataset.index)

# Create date range for reindexing
idx = pd.date_range('2017-01-31', '2017-12-31', freq='ME')  # 'ME' for month end
activity_dataset = activity_dataset.reindex(idx, fill_value=0)
activity_dataset.index = activity_dataset.index.strftime("%Y-%m")

# Read the data again - specify date format here too
activity_dataset = pd.read_csv(out_file, delimiter=';', decimal=",",
                             names=["date", "sum", "mean", "sem"],
                             parse_dates=["date"], 
                             date_format="%Y-%m-%d %H:%M:%S",  # Specify the date format
                             index_col="date")

# Read toilet hole data
toilet_hole = pd.read_csv(hole_csv, delimiter=';', decimal=",",
                        names=["date", "toilet_hole"],
                        parse_dates=["date"], index_col="date")

toilet_hole.index = toilet_hole.index.strftime("%Y-%m")

# Process activity dataset
activity_dataset = activity_dataset.drop('date')
activity_dataset = activity_dataset.loc[activity_dataset.index.dropna()]
activity_dataset.index = pd.to_datetime(activity_dataset.index)

idx = pd.date_range('2017-01-31', '2017-12-31', freq='ME')
activity_dataset = activity_dataset.reindex(idx, fill_value=0)
activity_dataset.index = activity_dataset.index.strftime("%Y-%m")

# Combine datasets
result = pd.concat([activity_dataset, toilet_hole], axis=1).reindex(activity_dataset.index)
result.index.rename('date', inplace=True)

# Export results to CSV
export_csv = result.to_csv(result_file, index=True, header=False, sep=';')

# Read final results
result_activity = pd.read_csv(result_file, delimiter=';',
                            names=["date", "sum", "mean", "sem", "width"])
result_activity = result_activity.fillna(0)
result_activity['width'] = (31 - result_activity['width']) / 35

# Create conditions for plotting
result_activity.loc[result_activity['width'] == 0, 'on'] = 0
result_activity.loc[result_activity['width'] > 0, 'on'] = -10

# Set up plot
plt.rcParams['figure.figsize'] = [9, 7]
x = np.arange(12)
plt.bar(x, result_activity['mean'], width=result_activity['width'], yerr=result_activity['sem'],
      color="#1f77b4")
plt.bar(x, result_activity['on'], width=result_activity['width'], color='green')
plt.gcf().autofmt_xdate()
plt.title(graph_title)
plt.xticks(x, result_activity.date)
plt.xlabel('Date')
plt.ylabel(label_y1, color="#1f77b4")
plt.tick_params(axis="y", labelcolor="#1f77b4")
plt.twinx()
plt.ylabel(label_y2, color="r")
plt.tick_params(axis="y", labelcolor="r", labelsize=8)
plt.plot(result_activity['sum'], "-r")
plt.xticks(x, result_activity.date)

# Set style
sns.set_style("dark")
print(result_activity)

# Uncomment to save the figure
# plt.savefig('/path/'+folder+'/'+figure)
# plt.savefig('/path/toilet.pdf')
# plt.show()
