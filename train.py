#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 13:21:36 2018

@author: sagar_paithankar
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from catboost import CatBoostRegressor as cbr
import logging
import time
import requests
import pickle
import json

# Defining Working Directory
#path1 = os.path.abspath(os.path.join(os.path.dirname('/Users/sagar_paithankar/Desktop/Load/myproject/Intra-Day/horizon/live/train.py'))) 
path1 = os.path.abspath(os.path.join(os.path.dirname('/root/myproject/intraday/forecast.py'))) 
path1 = r'G:\Anaconda_CC\spyder\myproject intraday\intraday'
os.chdir(path1)

# Import User Defined Class
from functions import *
from weather_functions import *

# Setting Logger
logger = logging.getLogger('train')
hdlr = logging.FileHandler(path1 + '/logs/train.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)
logger.info('Started')


# Setting Timezone
os.environ["TZ"] = "Asia/Kolkata"
#time.tzset() 
logger.info('Timezone Set: {}'.format(datetime.now()))

# Date
yesterday, today = get_date(-1), get_date(0)
tomorrow, dat = get_date(1), get_date(3) 
week_back = get_date(-7)
ydt = datetime.now().strftime("%Y-%m-%d") + " 00:00:00"
start_date = '2011-04-01'

# Get Load Data
logger.info('Getting Load Data')
df = pd.read_pickle('urd.pkl')
df_new = get_load(week_back,today)
df = pd.concat([df,df_new]).drop_duplicates().reset_index(drop=True)
df[df['datetime'] < today].to_pickle('urd.pkl')
del df_new

# Adding Features
df['date'] = df['datetime'].dt.date
df['tb'] = df['datetime'].apply(lambda x : ((x.hour*60 + x.minute)//15+1))
df['month'] = df['datetime'].dt.month
season = {1:1,2:1,3:1,4:2,5:2,6:2,7:2,8:2,9:2,10:2,11:1,12:1}
df['season'] = df['month'].map(season)
# Dealing with inconsistent values
df.loc[df['block_load'] < 420, 'block_load'] = np.nan
df.loc[(df['block_load'] < 1000) & (df['season'] == 2) & (df['month'] != 4) & (df['month'] != 10), 'block_load'] = np.nan ### Excluding the months april and october
df = df[['datetime','date','block_load','tb']]
df = df.sort_values('datetime')
df['holiday'] = holidays(df)
df = df[df['holiday'] == 0]
z = pd.DataFrame(pd.date_range(start=start_date, end=dat, freq='15min'), columns=['datetime'])
df = pd.merge(df, z, on='datetime', how='outer')
del z        
df['date'] = df['datetime'].dt.date
df['tb'] = df['datetime'].apply(lambda x : ((x.hour*60 + x.minute)//15+1))
df['hour'] = df['datetime'].dt.hour + 1
df['load_wm2h'] = df['block_load'].ewm(span=8).mean()
df['load_wm3h'] = df['block_load'].ewm(span=12).mean()
df['load_wm6h'] = df['block_load'].ewm(span=24).mean()
df['load_wm12h'] = df['block_load'].ewm(span=48).mean()
df['ramp1'] = df['block_load'] - df['block_load'].shift(1)
df['ramp2'] = df['block_load'].shift(1) - df['block_load'].shift(2)
df['ramp3'] = df['block_load'].shift(2) - df['block_load'].shift(3)
df['ramp4'] = df['block_load'].shift(3) - df['block_load'].shift(4)
hour_mean = df.groupby(['date','hour'])['block_load'].mean().reset_index()
hour_mean = hour_mean.rename(columns = {'block_load':'hour_mean'})
df = pd.merge(df, hour_mean, on=['date','hour'], how='left')

# Get Weather Data
logger.info('Getting Weather')
weather = pd.read_pickle('weather.pkl')
weather_new = get_weather(week_back,today,13)
weather = pd.concat([weather,weather_new]).drop_duplicates('datetime').reset_index(drop=True)   
weather[weather['datetime'] < today].to_pickle('weather.pkl')
del weather_new
# Merging Load and Weather
df = df.merge(weather, on='datetime', how='outer')
df['dow'] = df['datetime'].dt.dayofweek
df['doy'] = df['datetime'].dt.dayofyear
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['humidex'] = humidex(df)
df['RH'] = calculate_RH(df)
df['tb_aptemp'] = df['tb'] * df['apparent_temperature']
df['temp_mean_3h'] = df['temperature'].ewm(span=12).mean()
df['temp_mean_6h'] = df['temperature'].ewm(span=24).mean()
df['temp_mean_9h'] = df['temperature'].ewm(span=36).mean()
df['temp_mean_12h'] = df['temperature'].ewm(span=48).mean()
df['aptemp_mean_3h'] = df['apparent_temperature'].ewm(span=12).mean() 
df['aptemp_mean_5h'] = df['apparent_temperature'].ewm(span=20).mean() 
df['aptemp_mean_6h'] = df['apparent_temperature'].ewm(span=24).mean()
df['aptemp_mean_9h'] = df['apparent_temperature'].ewm(span=36).mean()
df['aptemp_mean_12h'] = df['apparent_temperature'].ewm(span=48).mean()
df['apptem_ahead_1h'] = df['apparent_temperature'].shift(-4)
df['apptem_mean_1h_ahead'] = df['apptem_ahead_1h'].ewm(span=4).mean()
df['humidity_mean_1h'] = df['humidity'].ewm(span=4).mean()
df['wind_speed_5h'] = df['wind_speed'].ewm(span=20).mean()
df.drop('apptem_ahead_1h',1,inplace=True)
df['hour_temp'] = df['hour'] * df['temperature']
logger.info('Data Ready for Training!')
df = df[df['datetime'] >= '2017-01-01 00:00:00']

### Training Starting
try:
    for i in range(4,161):
        print(i)
        df4 = pd.DataFrame()
        # For 1 hour to 2 hours ahead
        if i >=4 and i < 8:
            df['ramp'] = df['block_load'] - df['block_load'].shift(i)
            features = ['datetime','block_load','dow','year','month','tb','load_wm3h','load_wm2h',
            'ramp','ramp3','dew_point','temperature','humidity','apparent_temperature',
            'temp_mean_3h','aptemp_mean_6h']
            df4 = df[features].copy()
            df4['tb'] = df4['tb'].shift(-i)
            df4['temperature'] = df4['temperature'].shift(-i)
            df4['apparent_temperature'] = df4['apparent_temperature'].shift(-i)
            df4['dew_point'] = df4['dew_point'].shift(-i)
            df4['humidity'] = df4['humidity'].shift(-i)
            df4['aptemp_mean_6h'] = df4['aptemp_mean_6h'].shift(-i)
            df4['temp_mean_3h'] = df4['temp_mean_3h'].shift(-i)
            df4['dow'] = df4['dow'].shift(-i)
        # For 2 hours to 5 hours Ahead
        elif i >= 8 and i < 20:
            df['ramp'] = df['block_load'] - df['block_load'].shift(i)
            features = ['datetime','block_load','dow','month','year','tb','load_wm2h','load_wm3h',
            'ramp','ramp1','ramp3','dew_point','temperature','apparent_temperature', 'humidity',
            'wind_speed','temp_mean_3h','aptemp_mean_6h','humidity_mean_1h']
            df4 = df[features].copy()
            df4['tb'] = df4['tb'].shift(-i)
            df4['temperature'] = df4['temperature'].shift(-i)
            df4['apparent_temperature'] = df4['apparent_temperature'].shift(-i)
            df4['dew_point'] = df4['dew_point'].shift(-i)
            df4['aptemp_mean_6h'] = df4['aptemp_mean_6h'].shift(-i)
            df4['temp_mean_3h'] = df4['temp_mean_3h'].shift(-i)
            df4['humidity'] = df4['humidity'].shift(-i)
            df4['humidity_mean_1h'] = df4['humidity_mean_1h'].shift(-i)
            df4['dow'] = df4['dow'].shift(-i)
#            df4 = pd.get_dummies(df4,columns = ['year'])
            df4['wind_speed'] = df4['wind_speed'].shift(-i)
#            df4.drop(['year_2018'],1,inplace=True)
        # For 5 hours to 8 hours Ahead    
        elif i >= 161 and i < 320:
            df['ramp1'] = df['block_load'] - df['block_load'].shift(1)
            features = ['datetime','block_load','dow','tb','year','month','load_wm2h','ramp1',
            'dew_point','temperature','humidity','apparent_temperature','wind_speed',
            'temp_mean_3h','aptemp_mean_6h','RH', 'hour_temp']            
            df4 = df[features].copy()
            df4['temperature'] = df4['temperature'].shift(-i)
            df4['apparent_temperature'] = df4['apparent_temperature'].shift(-i)
            df4['dew_point'] = df4['dew_point'].shift(-i)
            df4['aptemp_mean_6h'] = df4['aptemp_mean_6h'].shift(-i)
            df4['temp_mean_3h'] = df4['temp_mean_3h'].shift(-i)
            df4['wind_speed'] = df4['wind_speed'].shift(-i)
            df4['humidity'] = df4['humidity'].shift(-i)
            df4['RH'] = df4['RH'].shift(-i)
            df4['hour_temp'] = df4['hour_temp'].shift(-i)
            df4['dow'] = df4['dow'].shift(-i)
            df4['lag1d'] = df4['block_load'].shift(96-i) 
            df4['lag1da'] = df4['block_load'].shift(97-i)
            df4['lag1db'] = df4['block_load'].shift(95-i)
            df4['load_stb'] = (df4['lag1d'] + df4['lag1da'] + df4['lag1db'])/3
            df4 = df4.drop(['lag1d', 'lag1da', 'lag1db'], axis=1)
        # For 8 hours to 10 hours Ahead
        elif i >= 20 and i < 40:
            df['ramp'] = df['block_load'] - df['block_load'].shift(i)
            features = ['datetime','block_load','hour_mean','load_wm3h','load_wm2h','dow','tb',
            'tb_aptemp','temperature','apparent_temperature','RH','dew_point','hour_temp','aptemp_mean_3h',
            'aptemp_mean_6h','temp_mean_3h','temp_mean_6h','humidex']
            df4 = df[features].copy()
            df4['temperature'] = df4.temperature.shift(-i)
            df4['apparent_temperature'] = df4.apparent_temperature.shift(-i)
            df4['dew_point'] = df4.dew_point.shift(-i)
            df4['RH'] = df4.RH.shift(-i)
            df4['temp_mean_6h']=df4.temp_mean_6h.shift(-i)
            df4['aptemp_mean_6h'] = df4.aptemp_mean_6h.shift(-i)
            df4['aptemp_mean_3h'] = df4.aptemp_mean_3h.shift(-i)
            df4['humidex'] = df4.humidex.shift(-i)
            df4['dow'] = df4.dow.shift(-i)
            df4['tb_aptemp'] = df4.tb_aptemp.shift(-i)
            df4['tb'] = df4.tb.shift(-i)
            df4['lag1d'] = df4['block_load'].shift(96-i) 
            df4['lag1da'] = df4['block_load'].shift(97-i)
            df4['lag1db'] = df4['block_load'].shift(95-i)
            df4['load_stb'] = (df4['lag1d'] + df4['lag1da'] + df4['lag1db'])/3
            df4 = df4.drop(['lag1d', 'lag1da', 'lag1db'], axis=1)
        # For 10 hours to 12 hours Ahead    
        elif i >= 40 and i < 56:
            df['ramp'] = df['block_load'] - df['block_load'].shift(i)
            features = ['datetime','block_load','dow','hour_mean','load_wm3h','load_wm2h','tb','tb_aptemp',
            'temperature','apparent_temperature','dew_point','aptemp_mean_3h','aptemp_mean_6h',
            'aptemp_mean_9h','temp_mean_3h','temp_mean_6h','RH','hour_temp', 'humidex']
            df4 = df[features].copy()
            df4['temperature'] = df4.temperature.shift(-i)
            df4['apparent_temperature'] = df4.apparent_temperature.shift(-i)
            df4['dew_point'] = df4.dew_point.shift(-i)
            df4['RH'] = df4.RH.shift(-i)
            df4['temp_mean_6h']=df4.temp_mean_6h.shift(-i)
            df4['temp_mean_3h'] = df4.temp_mean_3h.shift(-i)
            df4['aptemp_mean_6h'] = df4.aptemp_mean_6h.shift(-i)
            df4['aptemp_mean_9h'] = df4.aptemp_mean_9h.shift(-i)
            df4['aptemp_mean_3h'] = df4.aptemp_mean_3h.shift(-i)
            df4['humidex'] = df4.humidex.shift(-i)
            df4['dow'] = df4.dow.shift(-i)
            df4['tb_aptemp'] = df4.tb_aptemp.shift(-i)
            df4['tb'] = df4.tb.shift(-i)
            df4['lag1d'] = df4.block_load.shift(96-i) 
            df4['lag1da'] = df4.block_load.shift(97-i)
            df4['lag1db'] = df4.block_load.shift(95-i)
            df4['load_stb'] = (df4.lag1d + df4.lag1da + df4.lag1db)/3
            df4 = df4.drop(['lag1d', 'lag1da', 'lag1db'], axis=1)
        # For 10 hours to 18 hours Ahead    
        elif i >= 56 and i < 72:
            df['ramp'] = df['block_load'] - df['block_load'].shift(i)
            features = ['datetime','block_load','dow','hour_mean','load_wm3h','load_wm2h','year','month',
            'tb','ramp','ramp1','ramp2','temperature','apparent_temperature','dew_point','aptemp_mean_3h',
            'aptemp_mean_6h','aptemp_mean_9h','temp_mean_3h','temp_mean_6h','hour_temp','humidity']
            df4 = df[features].copy()
            df4['temperature'] = df4['temperature'].shift(-i)
            df4['apparent_temperature'] = df4['apparent_temperature'].shift(-i)
            df4['dew_point'] = df4['dew_point'].shift(-i)
            df4['temp_mean_6h']=df4['temp_mean_6h'].shift(-i)
            df4['temp_mean_3h'] = df4['temp_mean_3h'].shift(-i)
            df4['aptemp_mean_6h'] = df4['aptemp_mean_6h'].shift(-i)
            df4['aptemp_mean_9h'] = df4['aptemp_mean_9h'].shift(-i)
            df4['aptemp_mean_3h'] = df4['aptemp_mean_3h'].shift(-i)
            df4['humidity'] = df4['humidity'].shift(-i)
            df4['hour_temp'] = df4['hour_temp'].shift(-i)
            df4['dow'] = df4['dow'].shift(-i)
            df4['tb'] = df4['tb'].shift(-i)
            df4['lag1d'] = df4['block_load'].shift(96-i) 
            df4['lag1da'] = df4['block_load'].shift(97-i)
            df4['lag1db'] = df4['block_load'].shift(95-i)
            df4['load_stb'] = (df4['lag1d'] + df4['lag1da'] + df4['lag1db'])/3
            df4 = df4.drop(['lag1d', 'lag1da', 'lag1db'], axis=1)
        # For 18 hours to 30 hours Ahead    
        elif i >= 72 and i < 120:
            features = ['datetime','month','block_load','hour_mean','load_wm3h','load_wm2h','dow','tb',
            'tb_aptemp','temperature','apparent_temperature','dew_point','hour_temp','aptemp_mean_12h',
            'aptemp_mean_3h','aptemp_mean_6h','temp_mean_3h','humidex'].copy()
            df4 = df[features].copy()
            df4['temperature'] = df4.temperature.shift(-i)
            df4['apparent_temperature'] = df4.apparent_temperature.shift(-i)
            df4['dew_point'] = df4.dew_point.shift(-i)
            df4['aptemp_mean_6h'] = df4.aptemp_mean_6h.shift(-i)
            df4['aptemp_mean_12h'] = df4.aptemp_mean_12h.shift(-i)
            df4['aptemp_mean_3h'] = df4.aptemp_mean_3h.shift(-i)
            df4['humidex'] = df4.humidex.shift(-i)
            df4['dow'] = df4.dow.shift(-i)
            df4['tb_aptemp'] = df4.tb_aptemp.shift(-i)
            df4['tb'] = df4.tb.shift(-i)
            if i<=95:
                df4['lag1d'] = df4.block_load.shift(96-i) 
                df4['lag1da'] = df4.block_load.shift(97-i)
                df4['lag1db'] = df4.block_load.shift(95-i)
                df4['load_stb'] = (df4.lag1d + df4.lag1da + df4.lag1db)/3
                df4 = df4.drop(['lag1d', 'lag1da', 'lag1db'], axis=1)
        # For 40 hours to 40 hours Ahead    
        elif i >= 120 and i < 161:
            features = ['datetime','month','block_load','hour_mean','load_wm3h','load_wm2h','dow','tb',
            'tb_aptemp','temperature','apparent_temperature','dew_point','hour_temp','aptemp_mean_3h',
            'aptemp_mean_12h','aptemp_mean_6h','temp_mean_3h','humidex'].copy()
            df4 = df[features].copy()
            df4['temperature'] = df4.temperature.shift(-i)
            df4['apparent_temperature'] = df4.apparent_temperature.shift(-i)
            df4['dew_point'] = df4.dew_point.shift(-i)
            df4['aptemp_mean_6h'] = df4.aptemp_mean_6h.shift(-i)
            df4['aptemp_mean_12h'] = df4.aptemp_mean_12h.shift(-i)
            df4['aptemp_mean_3h'] = df4.aptemp_mean_3h.shift(-i)
            df4['humidex'] = df4.humidex.shift(-i)
            df4['dow'] = df4.dow.shift(-i)
            df4['tb_aptemp'] = df4.tb_aptemp.shift(-i)
            df4['tb'] = df4.tb.shift(-i)
        else:
            logger.info('We don"t forecast beyong 40 hours!')
            pass
            
        df4['target'] = df4['block_load'].shift(-i)
        df4.dropna(inplace=True)
        # Spliting into Input and Target Variable
        X = df4.drop('target',1).copy()
        y = df4[['datetime','target']].copy()       
        # Splitting the dataset into the Training set and Test set
        X_train = X[X['datetime'] < today+ ' 00:00:00'].iloc[:,1:]
        y_train = y[y['datetime'] < today+ ' 00:00:00'].iloc[:,1:]
        if i < 8:
            cat = cbr(loss_function='RMSE',learning_rate= 0.1,max_depth=8,reg_lambda=0.5) 
            lgbreg = cat.fit(X_train,y_train)
        elif i >= 8 and i < 20:
            cat = cbr(loss_function='RMSE',learning_rate=0.1,max_depth=7,iterations=1200) 
            lgbreg = cat.fit(X_train,y_train,silent=True)
        elif i >= 160 and i < 320:
            cat = cbr(loss_function='RMSE',iterations=800,learning_rate=0.1,max_depth=6,reg_lambda= 0.1) 
            lgbreg = cat.fit(X_train,y_train,silent=True)
        elif i >= 20 and i < 40:
            cat = cbr(iterations=1000, learning_rate=0.7, depth=5, loss_function='RMSE') 
            lgbreg = cat.fit(X_train,y_train,silent=True)
        elif i >=40 and i < 56:
            cat = cbr(iterations=1000,learning_rate=0.3,max_depth=10,loss_function='RMSE')
            lgbreg = cat.fit(X_train,y_train,silent=True)
        elif i >=56 and i < 72:
            cat = cbr(loss_function='RMSE',learning_rate=0.02,max_depth=8,iterations=700)
            lgbreg = cat.fit(X_train,y_train,silent=True)
        elif i >=72 and i < 120:
            cat = cbr(iterations=900,learning_rate=0.015,depth=10,bagging_temperature=0.8,loss_function='RMSE')
            lgbreg = cat.fit(X_train,y_train,silent=True)
        elif i >=120 and i < 161:
            cat = cbr(iterations=1300,loss_function='RMSE')
            lgbreg = cat.fit(X_train,y_train,silent=True)
        else:
            logger.info('We don"t forecast beyong 40 hours!')
            pass
        print('Model trained for horizon {}'.format(i))
        pickle.dump(lgbreg, open(path1+'/load_models/model_{}.pkl'.format(i), 'wb'))
    
    logger.info('Models trained!')
    #Email to report successful end of training models    
    data = {"to":["sagar.paithankar@dummy.com"],"from":"mailbot@dummy.com",
                  "subject":"","body":""}
    data['subject'] = "myproject intraday models trained."
    data['body'] = "An unbiased appreciation of uncertainity is a cornerstone to rationality (but it is not what people and organisations want)!"
    jsdata = json.dumps(data)
    requests.post("http://api.dummy.com/index.php?r=notifications/send-email",
            headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}, data= jsdata)
    
except:
    logger.info('Error while training for {} Horizon ahead'.format(i))
    #Email to report successful end of training models    
    data = {"to":["sagar.paithankar@dummy.com"],"from":"mailbot@dummy.com",
                  "subject":"","body":""}
    data['subject'] = "Error in myproject intraday models training."
    data['body'] = "An unbiased appreciation of uncertainity is a cornerstone to rationality (but it is not what people and organisations want)! Please fix Training for horizon {} ahead and onwards".format(i)
    jsdata = json.dumps(data)
    requests.post("http://api.dummy.com/index.php?r=notifications/send-email",
            headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}, data= jsdata)  
