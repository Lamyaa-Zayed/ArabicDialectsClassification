# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 21:15:04 2022

@author: lamya
"""

import pandas as pd
import numpy as np
import requests

data = pd.read_csv('dialect_dataset.csv')

data['id'] = data['id'].astype(str)
id_col = data['id']
id_col_arr = np.array(id_col) #make id vector to iterate request on it
id_col_arr = id_col_arr.tolist() #json need list to operate on it

# get data from post request
max=len(data['id'])//1000
start=0
df_new=pd.DataFrame(columns=['id', 'tweet'])
raw_data = pd.DataFrame()
for i in range(max):
  # print(str(i))
  end=start+1000
  results = requests.post('https://recruitment.aimtechnologies.co/ai-tasks', json=id_col_arr[start:end])

  r = pd.json_normalize(results.json()).T
  df_new['id'] = r.index
  df_new['tweet'] = r.values
  raw_data = raw_data.append(data.merge(df_new, on='id'))
  start=end
  
# the rest 197
results = requests.post('https://recruitment.aimtechnologies.co/ai-tasks', json=id_col_arr[start:])
r = pd.json_normalize(results.json()).T
df_new2=pd.DataFrame(columns=['id', 'tweet'])
df_new2['id'] = r.index
df_new2['tweet'] = r.values
raw_data = raw_data.append(data.merge(df_new2, on='id'))

# save data after merge
raw_data.to_csv('full_tweets.csv')

print("Done")
