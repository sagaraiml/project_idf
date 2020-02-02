#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 12:52:49 2018

@author: sagar_paithankar
"""


import pandas as pd
import numpy as np
import pymysql

def db_connect(host, user, password, db_name):
    wdb_connection = pymysql.connect(host=host,user=user,password=password,db=db_name)
    return wdb_connection

connection = db_connect('139.59.42.147', 'passt', 'passt123', 'energy_consumption') 

"""
Below function works for both 15 and 30 minutes resampling.
For 15 minutes - it fills one value forward and one value backwards and then linear interpolation
For 30 minutes - it simply fills the gap with the previous value
This has worked better compared to interpolating all values linearly.
Feel free to try something new yourself!
"""

def weather_processing(wdf, upsample):    
    wdf['datetime'] = pd.to_datetime(wdf['datetime'])
    wdf.iloc[:,1] = wdf.iloc[:,1].astype(float)
    wdf.loc[wdf['temperature'] < 2, 'temperature'] = np.nan
    wdf.loc[wdf.temperature.isnull(), 'dew_point'] = np.nan
    wdf.loc[wdf.temperature.isnull(), 'apparent_temperature'] = np.nan
    wdf.loc[wdf.temperature.isnull(), 'humidity'] = np.nan
    wdf.set_index('datetime', inplace=True)
    if upsample == '15min':
        wdf = wdf.interpolate(method='time')
        wdf = wdf.resample(upsample).asfreq()
        wdf = wdf.fillna(method='ffill', limit=1)
        wdf = wdf.fillna(method='bfill', limit=1)
        wdf = wdf.interpolate(method='time')
    else:
        wdf = wdf.interpolate(method='time')
        wdf = wdf.resample(upsample).asfreq()
        wdf = wdf.interpolate(method='time')
    wdf.reset_index(inplace=True)
    wdf = wdf.drop_duplicates('datetime')
    return wdf
    
def get_actual(sdt, tdt, plant_id):          
    # Get darksky for data till yesterday
    wSQL = "SELECT * FROM darksky_historical WHERE plant_id = {} AND datetime_local >= '{}'".format(plant_id,sdt)
    wdf = pd.read_sql(wSQL, con = connection)    
    wdf = wdf[[ 'datetime_local','apparent_temperature','dew_point','humidity','temperature','cloud_cover','wind_bearing', 'wind_speed']]
    wdf.insert(0, 'datetime', pd.to_datetime(wdf.datetime_local)) 
    wdf.drop('datetime_local', axis=1, inplace=True)
    wdf = weather_processing(wdf, '15min')
    return wdf

def get_weather_forecast(sdt, tdt, plant_id,table):
    # Wunderground Forecast
    w3SQL = "SELECT * FROM " + str(table) + " WHERE plant_id='" + str(plant_id) + "' AND datetime_local >='"+ tdt +"'"    
    wdf3 = pd.read_sql(w3SQL, con = connection)
    wdf3 = wdf3[['datetime_local','apparent_temperature','dew_point','humidity','temperature','cloud_cover','wind_bearing', 'wind_speed']]
    wdf3.insert(0, 'datetime', pd.to_datetime(wdf3['datetime_local'])) 
    wdf3.drop('datetime_local', axis=1, inplace=True)
    return wdf3

def optimized_forecast(sdt, tdt, plant_id):
    wSQL = "SELECT * FROM optimiser_forecast WHERE plant_id='" + str(plant_id) + "' AND datetime_local >= '" + sdt + "'"
    wdf = pd.read_sql(wSQL, con = connection)
    wdf['datetime'] = pd.to_datetime(wdf['datetime_local'])
    wdf = wdf[['datetime','apparent_temperature','dew_point','humidity','temperature','cloud_cover','wind_bearing', 'wind_speed']]
    wdf = wdf.sort_values('datetime').reset_index(drop=True)
    wdf = wdf.dropna()
    return wdf

"""
Get Actual Weather -> Then Optimized Weather -> Then Forecast -> Done! 
"""
def get_optimized_weather(sdt, tdt, plant_id):      
    wdf = get_actual(sdt, tdt, plant_id) # Actual Weather
    owdf = optimized_forecast(sdt, tdt, plant_id) # Optimized Weather
    wdf = wdf.merge(owdf, how='outer')
    wdf = wdf.sort_values('datetime')
    wdf = wdf.drop_duplicates('datetime')
    wdf3 = get_weather_forecast(sdt, tdt, plant_id, 'weather_selector') # Wunderground Forecast or Weather Combiner
    wdf = wdf.merge(wdf3, how='outer')
    wdf = wdf.sort_values('datetime')
    wdf = wdf.drop_duplicates('datetime')
    # Steps to convert to 15 minute format and the usual unnecessary precautious steps!
    wdf = weather_processing(wdf, '15min')
    return wdf

### Final Function
def switch(sdt, tdt, plant_id):
    weather = get_optimized_weather(sdt, tdt, plant_id)
    print('Optimization Baby!!')
    connection.close()
    return weather
