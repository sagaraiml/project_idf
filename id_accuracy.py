#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 07:19:25 2018

@author: sagar_paithankar
"""


import pandas as pd
import pymysql
from datetime import datetime, timedelta
import os
import time

os.environ['TZ'] = 'Asia/Calcutta'
time.tzset() 

today = datetime.now().strftime("%Y-%m-%d")
start = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

def get_data(table_name, sdt, tdt,version):
    db_connection = pymysql.connect("139.59.42.147","passt","passt123","energy_consumption")
    SQL = "SELECT * FROM " + table_name + " WHERE date BETWEEN '" + sdt + "' AND '" + tdt + "' AND version = '{}'".format(version)
    df = pd.read_sql(SQL, con=db_connection)
    return df
    

def get_sldc_data(table_name, sdt, tdt):
    db_connection = pymysql.connect("139.59.42.147","passt","passt123","energy_consumption")
    SQL = "SELECT * FROM " + table_name + " WHERE date BETWEEN '" + sdt + "' AND '" + tdt + "' AND discom_name='myproject'"
    df = pd.read_sql(SQL, con=db_connection)
    return df

a2 = get_sldc_data('tbl_actual_demand_met', start, today)
a2['time'] = pd.to_datetime(a2['timeslot']).dt.time
a2['datetime'] = pd.to_datetime(a2['date'].astype(str) + ' ' + a2['time'].astype(str))
a2 = a2[['datetime','actual_demand_met_mw']]
a2 = a2.rename(columns = {'actual_demand_met_mw':'block_load'})

actual = a2.copy()
actual = actual.set_index('datetime').resample('15min').mean().reset_index()
actual.drop_duplicates('datetime', inplace=True)

forecast = get_data('myproject_intraday_forecast_live', start, today,'v9')
forecast['datetime'] = pd.to_datetime(forecast.date.astype(str) + ' ' + (pd.to_datetime(forecast.start_time).dt.time).astype(str))
forecast = forecast[['datetime','forecast','horizon','revision_no','version']]
forecast = forecast.drop_duplicates(['datetime','horizon','revision_no'])
forecast = forecast.sort_values(['datetime','revision_no','horizon'])
forecast1 = forecast.copy()

conn = pymysql.connect("139.59.42.147","passt","passt123","energy_consumption")
mycursor = conn.cursor()

horizons = [4,8,24,32,48,60]
for i in horizons:
    forecast = forecast1[forecast1['horizon'] == i]
    df = pd.merge(actual, forecast, on='datetime', how='inner')
    df.insert(3, 'mae', abs(df['block_load'] - df['forecast']))
    df.insert(4, 'mae_4', df['mae'].rolling(window=4).mean())
    df.insert(5, 'mae_12', df['mae'].rolling(window=12).mean())
    df.insert(6, 'mae_96', df['mae'].rolling(window=96).mean())
    df.insert(7, 'avg_load_4', df['block_load'].rolling(window=4).mean())
    df.insert(8, 'avg_load_12', df['block_load'].rolling(window=12).mean())
    df.insert(9, 'avg_load_96', df['block_load'].rolling(window=96).mean())
    df.insert(0, 'date', df.datetime.dt.date)
    df.insert(1, 'time', df.datetime.dt.time)
    df.drop(['datetime','revision_no'], axis=1, inplace=True)
    df.fillna(0, inplace=True)
    df['datetime'] = list(map(lambda x,y: datetime.combine(x,y),df['date'],df['time']))
    df = df[df['datetime'] < (datetime.now() - timedelta(minutes=15))]
    df = df.iloc[:,:-1]
    df = df.tail(36)        
    query = "INSERT IGNORE INTO myproject_intraday_accuracy_live VALUES ('{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}')"
    for row in df.values:
        mycursor.execute(query.format(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12]))

conn.commit()
mycursor.close()
conn.close()
