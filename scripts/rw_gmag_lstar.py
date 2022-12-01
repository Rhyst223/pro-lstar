# -*- coding: utf-8 -*-
"""
Get the L* locations for ground-based
arrays over extended periods.

Uses pro_lstar to get ground-mag locations

Reads in yeary files of station geomagnetic
locations and determines L* as a function of time
(hourly time steps)

@author: krmurph1
"""

import pandas as pd
import os as os
import sys

sys.path.append('../')

from pro_lstar import *

stn_dir = 'D:\\GitHub\\SatRisk\\Stations'
out_dir = 'D:\\data\\L_Star\\pro_lstar\\stations'


def list_files():
    """
    Returns
    -------
    List of files for ground-based data

    """
    
    stn_files = [os.path.join(stn_dir,f) for f in os.listdir(stn_dir) if os.path.isfile(os.path.join(stn_dir,f))]
    
    return stn_files
    

# Main body to run stuff

stn_files = list_files()

y_st = 2006

# loop over station files
for sf in stn_files:
    df = pd.read_csv(sf, header=0, names=['date','stn','lat','lon','cgm_lat','cgm_lon'])
    
    if df['date'][0] < y_st:
        print(f'Skipping {df.iloc[0,0]}')
        continue
    
    sd = pd.to_datetime(df['date'][0],format='%Y')
    ed = pd.to_datetime(df['date'][0].astype(str)+'1231235959',format='%Y%m%d%H%M%S')
    
    print('Loading data for:')
    print(f'{sd} to {ed}')
    
    # load lstar data here
    ls = get_period(sd,ed,MFmodel='all')
    
    # get lstar location for 
    for stn, cgm_lat, cgm_lon in zip(df['stn'],df['cgm_lat'],df['cgm_lon']):
        if stn.lstrip() != 'ABK':
            continue
        
        print(f'{stn}, {cgm_lat}, {cgm_lon}')
        
        o_file = stn.lstrip()+'_'+sd.strftime('%Y')+'.csv.gz'
        o_file = os.path.join(out_dir,o_file)
        
        #skip if the file exists
        if os.path.isfile(o_file): 
            print('Skipping {o_file}... already exists')
            continue
        
        ldat = ls.get_location(cgm_lat,cgm_lon)
        
        
        
        ldat.data.to_csv(o_file,na_rep='Nan')
    



