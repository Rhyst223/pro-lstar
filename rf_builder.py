from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
import joblib
import argparse
from sklearn.preprocessing import LabelEncoder
import spacepy.time as spt
import spacepy.omni as om

model_vars = {'T89': ['Kp'], 'T96': ['Dst','Pdyn', 'ByIMF', 'BzIMF'], 'OSTA': ['Dst','Pdyn','BzIMF', 'Kp'],\
              'T01QUIET': ['Dst','Pdyn','ByIMF','BzIMF','G1','G2'], 'T01STORM': ['Dst', 'Pdyn', 'ByIMF', 'BzIMF', 'G2', 'G3'],\
              'T05': ['Dst', 'Pdyn', 'ByIMF', 'BzIMF', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6']}

def check_params(args):
    "Function to check command line parameters are valid"
    assert args.mf in ['T89','T96','T01QUIET','T01STORM','T05','OSTA'], \
      "Magnetic field model must be in ['T89','T96','T01QUIET','T01STORM','T05','OSTA']"
    
    assert ((args.sample <=1) & (args.sample>0)), "Sampling fraction must be in (0,1]"
    
    assert ((args.test_size <1) & (args.test_size>0)), "Test data fraction must be in (0,1)"
    
    assert args.mtype in ['regressor','classifier'], "Random forest model type must be either of ['regressor','classifier']"
    
    assert args.ncpus >=1, "Must have positive number of ncpus"
    

def build_features_labels(mf,sample=1,mtype='regressor'):
    '''
    Function to build features and labels from Pro-L* database.
    Labels depend on model type given.
    
    Inputs:
      - mf: Magnetic field model key, must be of those available in Pro-L*
      - sample: Fraction of each year used in case computing power is limited. Must be leq 1. Default = 1
      - mtype: Type of random forest model we aim to build. Default is regressor. Other allowed argument in 'classifier'
    '''

    #Read in data
    years = np.arange(2006,2017).astype(str)
    features = pd.DataFrame()
    for year in years:
        f = pd.read_csv('PATH'+year+'.csv.gz',compression='gzip',\
                      usecols=['0',mf,'mlat','mlt'],index_col=0)

        times = sorted(pd.Series(f.index.unique()).sample(frac=sample, random_state=1))
        f = f.loc[times,:]
        features = pd.concat([features,f],axis=0)

    features.sort_index(inplace=True)
    print('Pro-L* data imported successfully')
    #Remove bad data, here we consider bad to be:
    # L* above 10
    if mtype=='regressor':
      features = features[features[mf] <=10].dropna(subset=[mf],how='any')
      
    elif mtype=='classifier':
      values = {mf: 0}
      features.fillna(value=values)
      features.loc[features.mf !=0, mf] = 1
     
    features, labels = features[['mlat','mlt']], features[mf]   
    times = features.index.unique()
    features['sinmlt'] = features.mlt.map(lambda x: np.sin(2*np.pi*x/24))
    features['cosmlt'] = features.mlt.map(lambda x: np.cos(2*np.pi*x/24))
    features.drop('mlt',axis=1,inplace=True)

    #Get omni parameters
    ticks = spt.Ticktock(times)
    d = om.get_omni(ticks)
    #Take care of W and G parameters
    for i in range(d['W'].shape[1]):
        d['W'+str(i+1)] = d['W'][:,i]
    for j in range(d['G'].shape[1]):
        d['G'+str(j+1)] = d['G'][:,j]
    d = pd.DataFrame.from_dict(d[model_vars[mf]])
    #Alter IMF variables for certain models
    if mf in ['T96','T01QUIET']:
        d['ByIMF'], d['BzIMF'] = abs(d['ByIMF']), abs(d['BzIMF'])

    #Merge features and OMNI on index
    d.index = times
    features = features.join(d)
    features = features.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    
    return features, labels
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mf', help='Magnetic field model for random forest. Must be one of [T89,T96,T01QUIET,T01STORM,T05,OSTA]')
    parser.add_argument('--ncpus', default=1, type=int, help='Number of cpus available')
    parser.add_argument('--sample', default=1, type=float, help='Fraction of years to subsample. Default is 1 = Use full Pro-L* dataset')
    parser.add_argument('--mtype', default='regressor', help='Type of random forest to build. Default is regressor')
    parser.add_argument('--test_size',default=0.2,type=float,help='Fraction of training data to use as test')
    options = parser.parse_args()
    
    #Check arguments are allowed
    check_params(options)
    
    features, labels = build_features_labels(args.mf,sample=args.sample,mtype=args.mtype)
    
    #Split into train/test and normalize
    features_train, features_test, labels_train, labels_test = train_test_split(features, 
                                                                                labels, 
                                                                                test_size=args.test_size, 
                                                                                random_state=0)
    del features, labels
    
    min_max_scaler = preprocessing.MinMaxScaler()
    #Train data normalize except circular MLT
    features_train[model_vars[options.mf]] = min_max_scaler.fit_transform(features_train[model_vars[options.mf]])
    #Test data normalize except circular MLT
    features_test[model_vars[options.mf]] = min_max_scaler.fit_transform(features_test[model_vars[options.mf]])
    
    import time
    start = time.time()
    #Fit the random forest
    print('Attempting to fit rf')
    if options.mtype=='regressor':
        rf = RandomForestRegressor(ADD PARAMS ARGUMENT OR TAKE DEFAULT)
    else:
        rf = RandomForestClassifier(ADD PARAMS ARGUMENT OR TAKE DEFAULT)
    rf.fit(features_train,labels_train)
    end = time.time()
    print('Time to fit using is ' +str(end - start))
    
    #Score model based on relevant metrics
    y_true=labels_test
    y_pred=rf.predict(features_test)
    
    from sklearn import metrics, classification_report
    
    if options.mtype == 'regressor:
        print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_true, y_pred))
        print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_true, y_pred))
        print('Root Mean Squared Error (RMSE):', metrics.mean_squared_error(y_true, y_pred, squared=False))
        print('Mean Absolute Percentage Error (MAPE):', metrics.mean_absolute_percentage_error(y_true, y_pred))
        print('Explained Variance Score:', metrics.explained_variance_score(y_true, y_pred))
        print('Max Error:', metrics.max_error(y_true, y_pred))
        print('Mean Squared Log Error:', metrics.mean_squared_log_error(y_true, y_pred))
        print('Median Absolute Error:', metrics.median_absolute_error(y_true, y_pred))
        print('R^2:', metrics.r2_score(y_true, y_pred))
        print('Mean Poisson Deviance:', metrics.mean_poisson_deviance(y_true, y_pred))
        print('Mean Gamma Deviance:', metrics.mean_gamma_deviance(y_true, y_pred))
    else:
        print(classification_report(y_true, y_pred))
        
    del features_train, features_test, labels_train, labels_test

    filename = 'FILEPATH'+options.mf+'.sav'
    with open(filename, 'wb') as f:
        joblib.dump(rf, f)

