import pandas as pd
import matplotlib.pyplot as plt
import requests
import urllib.request
import io, gzip
import numpy as np
from itertools import groupby, count
from scipy.interpolate import griddata
import warnings
from copy import deepcopy

grid_lats = np.array([50.77, 54.74, 57.69, 60.0, 61.87, 63.43, 64.76, 65.91, 66.91, 67.79, 68.58, 69.3, \
             69.94, 70.53, 71.07, 71.57])
grid_lons = np.arange(0,360,15)

class MissingDataWarning(UserWarning):
    pass

class InterpolationWarning(UserWarning):
    pass

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

class get_period():
    """
    Class managing raw data for a given period/location/MF model. Finds and reads requested days
    from L* directory. Period must be continuous. As current L* files are yearly and take time to
    read in for a given year, must be able to recognise (for repeated custom requests) when year
    file is already in memory to speed up process.
    ### Input parameters:
    _start_: (string) Start of period. Resolution must be YYYYMMDDHH  
    _end_: (int) End of period. Assumed to inclusively close period (at midnight with no hourly resolution given)   
    _MFmodel_: (list) Magnetic field model to retrieve data for. Assumed all of them. If not one of\
                ['IGRF','OPQUIET','OSTA','T89','T96','T01QUIET','T01STORM','T05']
    _variable_: Variable to read in. Assumed only L*. Possible are [l,lm,b,x,y,z]. Denote l by blank string
    _model_threshold_: If probabilistic desired, require this to be the number of required models for L* output.
                        Default is None (provide model outputs only). If given must be <= len(MFmodel)
    """
    
    def __init__(self, start, end, MFmodel = 'all', variable = [''],model_threshold=None):
        
        self.start = start #string
        self.end = end #string
        self.variable = variable
        self.MFmodel = MFmodel #list
        self.model_threshold = model_threshold
        
        # check that given period is within data period
        assert ((start.year >2005) and (end.year <2017)),\
        "Period must be within the yearly range 2006-2016"
        
        if ((model_threshold is not None) and (MFmodel !='all')):
            assert (model_threshold <= len(MFmodel)),\
            "Model threshold must be less than or equal to the length of MFmodel"
        
        #Check that MFmodel code is allowed if not all
        if MFmodel is not 'all':
            assert all(m in ['IGRF','OPQUIET','OSTA','T89','T96','T01QUIET','T01STORM','T05'] for m in MFmodel),\
            "MFmodels must be in ['IGRF','OPQUIET','OSTA','T89','T96','T01QUIET','T01STORM','T05'] or all"
        
        #Check if given period contains non data days and provide warning if it does:
        if any(pd.to_datetime(d,infer_datetime_format=True) in pd.date_range(start,end) \
               for d in ['07/03/2006','19/02/2012','28/06/2013','29/06/2013','20/12/2015']):
            warnings.warn('Given period contains dates with missing data. Be wary of x-axis'\
                          +' in future plot_location_data() figures',MissingDataWarning)
        # now populate with data
        self.getData()
        
        
    def getData(self):
        """ 
        Read in the data for the given period, using pandas read_csv and slicing. Return data as
        a pandas dataframe.
        """
        
        def read_yearly_data(f,sdate,edate,cols):
            '''
            Function to read yearly L* files from data archive, and slice between start and
            end period
            '''
            raw_data = urllib.request.urlopen(f).read()
            data = gzip.decompress(raw_data)
            df = pd.read_csv(io.BytesIO(data),index_col=0,usecols=cols)  
            df = df.loc[str(sdate):str(edate)]
            df.mlat = df.mlat.apply(lambda x: round(x,2))
            
            return df
            
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
            url = [u for u in urls if u[-11:-7] in \
                   list(pd.date_range(self.start.year,self.end.year).year.unique())]
            
            df = pd.concat([read_yearly_data(u,self.start,self.end,cols) for u in url])
        else:
            url = [u for u in urls if u[-11:-7]==str(self.start.year)]
            
            df = read_yearly_data(url[0],self.start,self.end,cols)
            
        self.period_data = df
        
    def get_location(self,mlat, mlon):
        '''
        Get location data for the downloaded period, interpolating if off-grid in Pro-L*
        Inputs:
        _mlat_: (float) Magnetic latitude   
        _mlon_: (float) Magnetic longitude, in range (0-330 degrees)  
        '''
        
        def interpolate_custom_data(series,coords,mlat,mlon):
            '''
            Function to interpolate custom period variables if location is off Pro-L* grid
            '''
            if ((series.name=='mlt') and (series[:2].is_monotonic_increasing==False)):
                series[1]+=24
                series[-1]+=24
                    
            if pd.isnull(series).any()==True:
                #warnings.warn('Interpolation not possible for '+series.name+' at time '+str(series.index.unique()[0])+' due to undefined L*(s)'+\
                 #     ' at neighbouring points in Pro-L* domain',InterpolationWarning)
                return np.nan
            else:
                xnew = np.array([coords[:,0][0],mlon,coords[:,0][1]])
                if (xnew[-1] == 0):
                    xnew[-1]+=360
                ynew = np.array([min(coords[:,1]),mlat,max(coords[:,1])])
                znew = griddata(coords, series, (xnew[None,:], ynew[:,None]), method='linear')
                
                if series.name == 'mlt':
                    return znew[1][1] % 24
                else:
                    return znew[1][1]
                
        mlon = mlon % 360
        
        # check that given mlat is within range
        assert ((mlat >=50.77) and (mlat <=71.57)),\
        "Magnetic latitude must be between 50.77 and 71.57"
        
        interp=''
        if (mlon not in grid_lons): 
            print('Off grid longitude - Interpolation required')
            mlon_slice = find_nearest(grid_lons,mlon)
            interp+='lon'
        else:
            mlon_slice = [mlon]

        if (round(mlat,2) not in grid_lats): 
            print('Off grid latitude - Interpolation required')
            mlat_slice = find_nearest(grid_lats,mlat)
            interp+='lat'
        else:
            mlat_slice = [mlat]

        df = self.period_data[(self.period_data.mlat.isin(mlat_slice)) & \
                              (self.period_data.mlon.isin(mlon_slice))] 

        if interp !='':
            interp_df = pd.concat([tdf.apply(interpolate_custom_data,axis=0,\
                                             args=(np.array([[a,b] for a,b in zip(tdf.mlon,tdf.mlat)]),mlat,mlon)) \
                                   for time,tdf in df.groupby('0')],axis=1).T
            interp_df.index = df.index.unique()
            #self.data = interp_df
            locdf = deepcopy(self)
            locdf.period_data = None
            locdf.data = interp_df
            locdf.mlat = mlat
            locdf.mlon = mlon % 360
            return locdf
        else: 
            
            locdf = deepcopy(self)
            locdf.period_data = None
            locdf.data = df
            locdf.mlat = mlat
            locdf.mlon = mlon % 360
            #self.data = df
            return locdf
        
        
    def plot_location_data(self):
        '''
        Function to plot variables over given period, including shading for MLT
        '''
        
        def mlt_shade(ss,low,high,ubound=False,lbound=False):
            '''
            Function where to shade mlt area for GIVEN ss later shown.
            Not a well put togetehr function just a fudge to work right now
            '''
            hval = high
            lval = low
            if ubound == False:
                h = (np.ceil(ss['mlt'].iloc[high]) - ss['mlt'].iloc[high])/ \
                 (ss['mlt'].iloc[high+1]- ss['mlt'].iloc[high])

                hval = hval + h

            if lbound == False:
                l = (-np.floor(ss['mlt'].iloc[low]) + ss['mlt'].iloc[low])/ \
                 (-ss['mlt'].iloc[low-1]+ ss['mlt'].iloc[low])
                lval = lval - l

            return lval, hval
        
        #Check that user has actually filtered period data for a certain location
        assert hasattr(self, 'data'),\
        "plot_location_data() requires filtered location data following get_period. Please"\
            +" create location object with get_location() and apply this function to that"
        
        fig, ax = plt.subplots(len(self.variable),1,figsize=(12,8),sharex='all')
        
        styles = ['+','o','*','s','d','1','x','h'][:len(self.MFmodel)]
        plot_vars = [[m + '_' + v if v!='' else m for m in self.MFmodel] for v in self.variable]
        
        [self.data[p].plot(ax=a,style=styles,legend=False) for p,a in zip(plot_vars,ax.flatten())]
        
        if self.model_threshold is not None:
            [self.data[p].dropna(thresh=self.model_threshold).median(axis=1)\
             .plot(ax=a,color='black',legend=False,label='Probabilistic') for p,a in zip(plot_vars,ax.flatten())]
            
        
        #Handle MLT shading in each of the axis, taking care in case of only one variable
        night = [list(g) for k, g in groupby(np.where(~self.data['mlt'].between(6,18,inclusive=False))[0], \
                  key=lambda i,j=count(): i-next(j))]
        for x in night:
            if x[0] == 0:
                l,u = mlt_shade(self.data,x[0],x[len(x)-1],lbound=True)
                [this_ax.axvspan(l,u,facecolor="gray",alpha=0.2) \
                     for this_ax in ax]

            elif x[len(x)-1] == (len(self.data)-1):
                l,u = mlt_shade(self.data,x[0],x[len(x)-1],ubound=True)
                [this_ax.axvspan(l,u,facecolor="gray",alpha=0.2) \
                     for this_ax in ax]

            else:
                l,u = mlt_shade(self.data,x[0],x[len(x)-1])
                [this_ax.axvspan(l,u,facecolor="gray",alpha=0.2) \
             for this_ax in ax]
                
        [this_ax.set_ylabel(v) if v!='' else this_ax.set_ylabel('L*') for \
         this_ax,v in zip(ax,self.variable)]
        [this_ax.set_xlabel('') if v!='' else this_ax.set_ylabel('L*') for \
         this_ax,v in zip(ax,self.variable)]
        
        #fig.tight_layout()
        #ax[0].legend(bbox_to_anchor=(1, 1.1),ncol=len(self.MFmodel))
        if self.model_threshold is not None:
            ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
              ncol=len(self.MFmodel)+1, fancybox=True, shadow=True)
        else:
            ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
              ncol=len(self.MFmodel), fancybox=True, shadow=True)
        plt.xticks(rotation=90)
        plt.show()
        
 
        
 
        
