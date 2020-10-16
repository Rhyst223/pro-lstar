import pandas as pd
import matplotlib.pyplot as plt
import requests
import urllib.request
import io
import numpy as np

grid_lats = np.array([50.77, 54.74, 57.69, 60.0, 61.87, 63.43, 64.76, 65.91, 66.91, 67.79, 68.58, 69.3, \
             69.94, 70.53, 71.07, 71.57])
grid_lons = np.arange(0,360,15)

def find_nearest(arr,m):
    '''
    Find nearest magnetic longitudes in Pro-L* grid for interpolation
    '''
    below = arr[np.searchsorted(arr,m,'left')-1]
    try:
        above = arr[np.searchsorted(arr,m,'right')]
    except IndexError:
        above = 0

    return [below,above]

class CustomData():
    """
    Class managing raw data for a given period/location/MF model. Finds and reads requested days
    from L* directory. Period must be continuous. As current L* files are yearly and take time to
    read in for a given year, must be able to recognise (for repeated custom requests) when year
    file is already in memory to speed up process.
    ### Input parameters:
    _start_: (string) Start of period. Resolution must be YYYYMMDDHH  
    _end_: (int) End of period. Assumed to inclusively close period (at midnight with no hourly resolution given)  
    _mlat_: (float) Magnetic latitude   
    _mlon_: (float) Magnetic longitude, in range (0-330 degrees)   
    _MFmodel_: (list) Magnetic field model to retrieve data for. Assumed all of them. If not one of\
                ['IGRF','OPQUIET','OSTA','T89','T96','T01QUIET','T01STORM','T05']
    _variable_: Variable to read in. Assumed only L*. Possible are [l,lm,b,x,y,z]. Denote l by blank string
    """
    
    def __init__(self, start, end, mlat, mlon, MFmodel = 'all', variable = ['']):
        
        self.start = start #string
        self.end = end #string
        self.mlat = mlat #float
        self.mlon = mlon % 360 #float
        self.variable = variable
        self.MFmodel = MFmodel #list
        
        # check that given period is within data period
        assert ((start.year >2005) and (end.year <2017)),\
        "Period must be within the yearly range 2006-2016"
        
        # check that given mlat is within range
        assert ((mlat >=50.77) and (mlat <=71.57)),\
        "Magnetic latitude must be between 50.77 and 71.57"
        
        #Check that MFmodel code is allowed if not all
        if MFmodel is not 'all':
            assert all(m in ['IGRF','OPQUIET','OSTA','T89','T96','T01QUIET','T01STORM','T05'] for m in MFmodel),\
            "MFmodels must be in ['IGRF','OPQUIET','OSTA','T89','T96','T01QUIET','T01STORM','T05'] or all"

        # now populate with data
        self.getData()
        
        
    def getData(self):
        """ 
        Read in the data for the given period, using pandas read_csv and slicing. Return data as
        a pandas dataframe.
        """
        
        def read_yearly_data(f,sdate,edate,mlon,mlat,cols):
            '''
            Function to read yearly L* files from data archive, and slice between start and
            end period
            '''
            raw_data = urllib.request.urlopen(f).read()
            data = gzip.decompress(raw_data)
            df = pd.read_csv(io.BytesIO(data),index_col=0,usecols=cols)  
            df = df.loc[str(sdate):str(edate)]
            df.mlat = df.mlat.apply(lambda x: round(x,2))
            #df = df[(df.mlat == mlat) & (df.mlon == mlon)]
            
            if (mlon not in grid_lons): 
                print('Off grid longitude - Interpolation required')
                mlon = find_nearest(grid_lons,mlon)
                interp='lon'
            else:
                mlon = [mlon]
                
            if (round(mlat,2) not in grid_lats): 
                print('Off grid latitude - Interpolation required')
                mlat = find_nearest(grid_lats,mlat)
                interp+='lat'
            else:
                mlat = [mlat]
    
            df = df[(df.mlat.isin(mlat)) & (df.mlon.isin(mlon))] 
        
            if interp in locals():
                interpolate_custom_data(df,interp)
            else: 
                return df
        
        def interpolate_custom_data(df,method):
            '''
            Function to interpolate custom period data if location is off Pro-L* grid
            Inputs:
                method  - lon: longitudinal interpolation only
                        - lat: latitudinal interpolation only
                        - lonlat: full-grid interpolation
            '''
            
            
        urls = ['https://researchdata.reading.ac.uk/222/9/2006.csv.gz',
                'https://researchdata.reading.ac.uk/222/11/2007.csv.gz',
                'https://researchdata.reading.ac.uk/222/12/2008.csv.gz',
                'https://researchdata.reading.ac.uk/222/13/2009.csv.gz',
                'https://researchdata.reading.ac.uk/222/14/2010.csv.gz',
                'https://researchdata.reading.ac.uk/222/15/2011.csv.gz',
                'https://researchdata.reading.ac.uk/222/16/2012.csv.gz',
                'https://researchdata.reading.ac.uk/222/17/2013.csv.gz',
                'https://researchdata.reading.ac.uk/222/18/2014.csv.gz',
                'https://researchdata.reading.ac.uk/222/19/2015.csv.gz',
                'https://researchdata.reading.ac.uk/222/20/2016.csv.gz']
        
        if self.MFmodel == 'all':
            self.MFmodel = ['IGRF','OPQUIET','OSTA','T89','T96','T01QUIET','T01STORM','T05']
            
        cols = ['0'] + [m + '_' + v if v!='' else m for v in self.variable for m in self.MFmodel] \
                    + ['mlat','mlon','mlt']
            
        
        if self.start.year != self.end.year:
            print('Period covers multiple years - expect longer read in time')
            url = [u for u in urls if u[-11:-7] in [str(self.start.year),str(self.end.year)]]
            
            #df = pd.concat([read_yearly_data(u,self.start,self.end,cols=cols) for u in url])
        else:
            url = [u for u in urls if u[-11:-7]==str(self.start.year)]
            
            df = read_yearly_data(url[0],self.start,self.end,self.mlon,self.mlat,cols=cols)
            
        self.data = df
        
