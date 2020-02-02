# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:50:40 2019

@author: sagar_paithankar
"""
import pandas as pd
import numpy as np
import pymysql
from datetime import datetime, timedelta
import os
import time
import dateutil

os.environ['TZ'] = 'Asia/Calcutta'


def get_deviation():
    
    conn = pymysql.connect("139.59.42.147","passt","passt123","energy_consumption")
    mycursor = conn.cursor()
    
    sql = "select distinct created_at from myproject_intraday_forecast_live ORDER BY created_at DESC limit 2 "  
    df = pd.read_sql(sql, conn) 
    first= str(df.iloc[0,0]) 
    second= str(df.iloc[1,0])
    
    sql1= "select * from myproject_intraday_forecast_live where created_at in ( '"+first+"', '"+second+"')"
    #print(sql1)
    #exit()
    df_data = pd.read_sql(sql1, conn) 
        
    df_1 = df_data[df_data['created_at'] == first]
    df_2 = df_data[df_data['created_at'] == second]
    
    df_1 = df_1.reset_index(drop=True) 
    
    df_1['start_time']=pd.to_datetime(df_1['start_time'])
    df_1['start_time']= df_1['start_time'].apply(lambda x: x.time())
    
    df_1['end_time']=pd.to_datetime(df_1['end_time'])
    df_1['end_time']= df_1['end_time'].apply(lambda x: x.time())
    
    df_1=df_1[:17]
    
    df_2 = df_2.loc[1:,].reset_index(drop=True) 
    
    df_2['start_time']=pd.to_datetime(df_2['start_time'])
    df_2['start_time']= df_2['start_time'].apply(lambda x: x.time())
    
    df_2['end_time']=pd.to_datetime(df_2['end_time'])
    df_2['end_time']= df_2['end_time'].apply(lambda x: x.time())
    df_2=df_2.rename(columns={'forecast': 'forecast_2'})
    
    df_2=df_2.iloc[:,0:4]
    df_2=df_2.iloc[:17,]
    
    df_main=pd.merge(df_1, df_2, on =['date','start_time','end_time'],how='inner')
    
    df_main['deviation']=abs(df_main['forecast']-df_main['forecast_2'])
    
    df_main=df_main.drop(columns=["horizon","version"])
    
    df_main['boolean']=df_main['deviation'].apply(lambda x: x>30).astype(int)
    df_main=df_main.rename(columns={'forecast':'previous_revision_forecast', 'forecast_2':'current_revision_forecast','boolean':'deviation_value'})
    
    df_main=df_main[['date','start_time','end_time','previous_revision_forecast','current_revision_forecast','deviation','deviation_value','created_at','revision_no']]
    
    query = "INSERT INTO myproject_intraday_deviation_data(`date`,`start_time`,`end_time`,`previous_revision_forecast`,`current_revision_forecast`,`deviation`,`deviation_value`,`created_at`,`revision_no`) VALUES "
    
    for row in df_main.values:
        query += "('"+str(row[0])+"','"+str(row[1])+"','"+str(row[2])+"','"+str(row[3])+"','"+str(row[4])+"','"+str(row[5])+"','"+str(row[6])+"','"+str(row[7])+"','"+str(row[8])+"'),"
        
    query = query.rstrip(",")
    print(query)
    
    mycursor.execute(query)
    conn.commit()
    conn.close()
    
    return query
    

