#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 12:52:49 2018

@author: sagar_paithankar
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import pymysql
import pickle
import requests

# Defining Working Directory
path1 = '/root/myproject/intraday'
os.chdir(path1)

# Import User Defined Class
from functions import *
from weather_functions import *
from opt_switch import *

# Setting Logger
logger = logging.getLogger('forecast')
hdlr = logging.FileHandler(path1 + '/logs/forecast.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)
logger.info('Started')

# Setting Timezone
os.environ["TZ"] = "Asia/Kolkata"
time.tzset() 
logger.info('Timezone Set: {}'.format(datetime.now()))

# Date
dby, yesterday, today = get_date(-2), get_date(-1), get_date(0)
tomorrow, dat = get_date(1), get_date(3) 
week_back = get_date(-7)
start_date = (datetime.now() - timedelta(days=360)).strftime("%Y-%m-%d")
ydt = datetime.now().strftime("%Y-%m-%d") + " 00:00:00"
right_now = datetime.now() + timedelta(minutes=1)

# Get Load Data
logger.info('Getting Load Data')
df = pd.read_pickle('urd.pkl')
df_new = get_load(week_back,today)
df = pd.concat([df,df_new]).drop_duplicates().reset_index(drop=True)
df = df[df['datetime'] >= start_date]
sldc5 = sldc_load(yesterday, today, 5)
sldc1 = sldc_load(yesterday, today, 1)
df = df.append(sldc5).drop_duplicates('datetime',keep='first').reset_index(drop=True)
df = df.append(sldc1).drop_duplicates('datetime',keep='first').reset_index(drop=True)
del df_new,sldc5,sldc1
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
df = interpolating_live(df)


# Check if any new data is available. If yes then run, else, exit.
check = pd.read_csv(path1 + '/check_df.csv', parse_dates=['datetime'], infer_datetime_format=True)
if check.iloc[0]['datetime'] in df.datetime.tail(1).tolist():
 	logger.info("No new data since last run. What to do?!")
 	exit()

logger.info("New Data is available, so carrying on!")
df.tail(1).to_csv(path1 + '/check_df.csv')

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
weather =  weather[weather['datetime'] >= start_date]
weather1 = switch(dby,today,13) # Getting Optimised Weather
weather = pd.concat([weather,weather1]).drop_duplicates().reset_index(drop=True)
del weather1

# Merging Load and Weather
df = df.merge(weather, on='datetime', how='outer')
df['dow'] = df['datetime'].dt.dayofweek
df['doy'] = df['datetime'].dt.dayofyear
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['hour'] = df['datetime'].dt.hour + 1
df['tb'] = df['datetime'].apply(lambda x : ((x.hour*60 + x.minute)//15+1))
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
logger.info('Data Ready for Forecasting!')

conn = pymysql.connect(host="139.59.42.147",user="passt",password="passt123",db="energy_consumption")
mycursor = conn.cursor()

for i in range(4,161):
    print(i)
    logger.info('forecasting for horizon: {}'.format(i))
    # For 1 hours to 2 hours Ahead 
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
    # For 2 hours to 4 hours Ahead 
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
#        df4 = pd.get_dummies(df4,columns = ['year'])
        df4['wind_speed'] = df4['wind_speed'].shift(-i)
 #       df4.drop(['year_2018'],1,inplace=True)  
    # For 4 hours to 8 hours Ahead 
    elif i >=161 and i < 320:
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
        df4['dow'] = df4['dow'].shift(-i)
        df4['hour_temp'] = df4['hour_temp'].shift(-i)
        df4['RH'] = df4['RH'].shift(-i)
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
    # For 10 hours to 14 hours Ahead 
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
    # For 14 hours to 18 hours Ahead    
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
        df4['temperature'] = df4['temperature'].shift(-i)
        df4['apparent_temperature'] = df4['apparent_temperature'].shift(-i)
        df4['dew_point'] = df4['dew_point'].shift(-i)
        df4['aptemp_mean_6h'] = df4['aptemp_mean_6h'].shift(-i)
        df4['aptemp_mean_12h'] = df4['aptemp_mean_12h'].shift(-i)
        df4['aptemp_mean_3h'] = df4['aptemp_mean_3h'].shift(-i)
        df4['humidex'] = df4['humidex'].shift(-i)
        df4['dow'] = df4['dow'].shift(-i)
        df4['tb_aptemp'] = df4['tb_aptemp'].shift(-i)
        df4['tb'] = df4['tb'].shift(-i)
        if i<=95:
            df4['lag1d'] = df4['block_load'].shift(96-i) 
            df4['lag1da'] = df4['block_load'].shift(97-i)
            df4['lag1db'] = df4['block_load'].shift(95-i)
            df4['load_stb'] = (df4['lag1d'] + df4['lag1da'] + df4['lag1db'])/3
            df4 = df4.drop(['lag1d', 'lag1da', 'lag1db'], axis=1)
    # For 30 hours to 40 hours Ahead    
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
        logger.info('We don"t forecast beyong 15 hours!')
        pass
        
    df4.dropna(inplace=True)
    dtlist, forecast = [], []
    df4 = df4[(df4['datetime'] >= yesterday) & (df4['datetime'] <= tomorrow)].tail(1)
    dtlist.append(df4.datetime)
    df4 = df4.iloc[:,1:]
    model = pickle.load(open(path1+'/load_models/model_{}.pkl'.format(i), 'rb'))
    # Predicting
    y_pred = model.predict(df4)
    forecast.extend(y_pred)
    res = pd.DataFrame()
    res['datetime'] = dtlist[0]
    res['revision_no'] = res['datetime'].apply(lambda x : ((x.hour*60 + x.minute)//15+1))
    res['datetime'] = res.datetime + timedelta(minutes=i*15)
    res['date'] = res['datetime'].dt.date
    res['forecast'] = forecast
    res['start_time'] = res['datetime'].dt.time
    res['end_time'] = (res['datetime'] + timedelta(minutes=15)).dt.time
    res['horizon'] = i
    res['version'] = 'v9'
    res['created_at'] = right_now 
    res = res[['date','start_time','end_time','forecast','horizon','version','created_at','revision_no']]   
#    res['forecast'] = res['forecast']
    res1 = res[['date','start_time','end_time','forecast','horizon','version','created_at']].copy()
    res1['version'] = 'v1'
#    logger.info('forecast result: {}'.format(res))
    logger.info('forecast ready to go for horizon: {}'.format(i))    
    # Inserting in the DB ### ON DUPLICATE KEY UPDATE forecast='{}', horizon='{}', created_at='{}'"
    query = "INSERT INTO myproject_intraday_forecast_live(`date`,`start_time`,`end_time`,`forecast`,`horizon`,`version`,`created_at`,`revision_no`) VALUES ('{}','{}','{}','{}','{}','{}','{}','{}')"    
    query1 = "INSERT INTO intraday_forecast_live(`date`, `start_time`, `end_time`, `forecast`, `horizon`, `version`, `created_at`) VALUES ('{}','{}','{}','{}','{}','{}','{}') ON DUPLICATE KEY UPDATE forecast='{}', horizon='{}', created_at='{}'"
    for row in res.values:
        mycursor.execute(query.format(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]))
        
    for row in res1.values:
        mycursor.execute(query1.format(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[3], row[4], row[6]))
        
conn.commit()
mycursor.close()
conn.close()
res1.to_csv('/root/myproject/intraday/v1results.csv',index=False)
print('1')
def get_token():
    username = "tech@dummy.com"
    password = "dentintheuniverse"
    base_url = "https://apis.dummy.com/api"          
    try:
        headers = {}
        params = {'username': username, 'password': password}
        
        api_token = base_url + "/get-token"
        
        r = requests.post(url=api_token, headers=headers, params=params)
        return r.json()

    except Exception as e:
        print("Error in getting token : {a}".format(a=e))
        
def get_api_data(sdt, tdt, params, headers, api_get):
    
    headers['token']=get_token()['access_token']
    
    try:           
        r = requests.get(url=api_get, headers=headers, params=params)
        return r.json()
    
    except Exception as e:
            print("Error in fetching api data : {a}".format(a=e))

def store_forecast_api(df,version):
    try:
        print('12')
        token = get_token()
        print(df)
        print('Setting up final dataframe...')
        result = df[['date','start_time','end_time','forecast','horizon','version', 'type', 'created_at']]
        result['version']= version
        result['date'] = result['date'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
        result['source'] = "urd"
        result['revision'] = 0
        result = result.loc[:, ['date','start_time','end_time','source','forecast','version','horizon','revision']]
        ip=result.to_json(orient = 'records')  
        json_data = {'data': ip, 'type':'dayahead', "client_id":51,  "model" : "test" }
        url = "https://apis.dummy.com/api/load/setForecast"
        header={'token': get_token()['access_token'], 'content-type': 'application/x-www-form-urlencoded'}
        response = requests.post(url=url, data=json_data, headers=header)
        print(response.json())      
        print("Storing forecast in DB")               
    except:
        print("Error while stroing in DB")
        
print('2')

store_forecast_api(res1, 'v1')
logger.info("Snap like Thanos")
    
