import os
import pandas as pd
import numpy as np
data = pd.read_csv(r'datasets\ETT-data\TrainOneDay.csv')

# make X  = Y delay 1hr(15*n)
n = 4
data['Power_step'] = np.roll(data.Power.values,n) 
data = data.iloc[n:,:]
data = data.rename(columns={"DateTime": "date"}, errors="raise")

# data.columns: ['date', ...(other features), target feature]
target = 'Power'

col = data.columns[1:]
cols = list(data.columns)
cols.remove(target)
cols.remove('date')
data = data[data['Power']>0]
data = data[['date']+cols+[target]].dropna()
data.iloc[:,1:] = data.iloc[:,1:].astype(float)
data.to_csv(r'datasets\ETT-data\Etth1.csv',index=False)