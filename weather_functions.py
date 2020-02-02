#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 14:06:53 2018

@author: sagar_paithankar
"""

import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import pymysql

def get_weather(start_date, end_date, plant_id):
    conn = pymysql.connect("139.59.42.147","passt","passt123","energy_consumption")
    sql = "SELECT * FROM darksky_historical WHERE plant_id = {} and DATE(datetime_local) >= '{}'".format(plant_id,start_date)
    df = pd.read_sql(sql, conn)
    df = df[['datetime_local','apparent_temperature','dew_point','humidity','temperature','cloud_cover',
             'wind_bearing', 'wind_speed']]
    df.insert(0, 'datetime', pd.to_datetime(df.datetime_local)) 
    df.drop('datetime_local', axis=1, inplace=True)
    df.sort_values('datetime',inplace=True)
    sql1 = "SELECT * FROM wunderground_forecast WHERE plant_id = {} and DATE(datetime_local) >= '{}'".format(plant_id,end_date)
    df1 = pd.read_sql(sql1, conn)
    df1 = df1[['datetime_local','apparent_temperature','dew_point','humidity','temperature','cloud_cover',
             'wind_bearing', 'wind_speed']]
    df1.insert(0, 'datetime', pd.to_datetime(df1.datetime_local)) 
    df1.drop('datetime_local', axis=1, inplace=True)
    df1.sort_values('datetime',inplace=True)
    conn.close()
    df = df.merge(df1, how='outer')
    df['datetime'] = pd.to_datetime(df.datetime)
    df.iloc[:,1:] = df.iloc[:,1:].astype(float)
    df.loc[df['temperature'] < 3, 'temperature'] = np.nan
    df.loc[df['temperature'].isnull(), 'dew_point'] = np.nan
    df.loc[df['temperature'].isnull(), 'apparent_temperature'] = np.nan
    df.loc[df['temperature'].isnull(), 'cloud_cover'] = np.nan
    df.loc[df['temperature'].isnull(), 'humidity'] = np.nan
    df.loc[df['temperature'].isnull(), 'wind_bearing'] = np.nan
    df.loc[df['temperature'].isnull(), 'wind_speed'] = np.nan
    df.set_index('datetime', inplace=True)
    df = df.interpolate(method='time')
    df = df.resample('15min').asfreq()
    df = df.fillna(method='ffill', limit=1)
    df = df.fillna(method='bfill', limit=1)
    df = df.interpolate(method='time')
    df.reset_index(inplace=True)
    df = df.drop_duplicates('datetime')  
    return df

def humidex(df):
    Td = df['dew_point'] + 273.15
    Tc = df['temperature']
    a = 1/273.15
    b = 1/Td
    c = Tc+0.555*((6.11*(np.exp(5417.75*(a-b))))-10)
    return c

def calculate_RH(df):
    Es = 6.11*10.0**((7.5*df['apparent_temperature']) / (237.7 + df['apparent_temperature']))
    E = 6.11*10.0**((7.5*df['dew_point'])/(237.7 + df['dew_point']))
    RH = (E/Es)*100
    return RH

