# pro-lstar
Probabilistic L* mapping tool for ground and space observations

## Required Python packages:
  - Pandas
  - Numpy 
  - Matplotlib
  - Urllib
  - io
  - gzip
  - itertools
  - scipy
  - copy
  

## Several features of this package require the accompanying Pro-L* dataset:

Thompson, Rhys and Morley, Steven (2020): Pro-L*: probabilistic hourly L* values, with associated McIlwain Lm, magnetic field intensity B, and Cartesian coordinates for 7 global magnetic field models in the Northern Hemisphere in the period 2006-2016. University of Reading. Dataset. http://dx.doi.org/10.17864/1947.222


In its current iteration, Pro-L* allows users to:
  - Read in and plot Pro-L* variables for any given ground locations over a specific period (within ~2.5-10L in the Northern Hemisphere and the time frame given in the above dataset - refer for more details) for 7 magnetic field models, or simple probabilistic combinations of them. 
  
 The model is simple to implement, with example code as follows:

```python
import pandas as pd
from prolstar import *

# start, end - datetime type object
start = pd.to_datetime('2006-12-01')
end = pd.to_datetime('2006-12-02')

df = get_period(start, end, MFmodel = 'all', variable = ['','lm','b','x','y','z'],model_threshold=4)

# get the data for specific location
location  = df.get_location(66,4)

location.plot_location_data()
```

The above assumes that the data set has been downloaded and is available locally. If the dataset has not be downloaded set ```local=False``` in the function call ```get_period```. Note loading the data locally is signifcantly faster then loading it from the internet. 

To setup local loading of files create a configuration file ```pro_lstar_rc``` and add fill in the ```data_dir``` line with the local directory where the data set has been downloaded. See below on the details of the dataset.

## The Dataset

Full details of the Dataset can be found [here](https://researchdata.reading.ac.uk/222/21/ReadMe.docx). Data are stored in compressed csv files stored online. A brief summary is below.

The global magnetic field models in the dataset (also including the internal field from the International Geomagnetic Reference Field (IGRF)), in the order in which they appear for each variable, are coded as follows:

- IGRF - International Geomagnetic Reference Field
- T89 - Tsyganenko (1989)
- OPQUIET - Olson and Pfitzer (1974)
- T96 - Tsyganenko (1996)
- OSTA - Ostapenko and Maltsev (1997)
- T01STORM - Tsyganenko (2003)
- T05 - Tsyganenko and Sitnov (2005)
- T01QUIET - Tsyganenko et al. (2002)

The columns of the csv files are as follows: 

- Column 0 - Time (hour) given in YYYY-MM-DD HH:MM:SS
- Columns 1:8 - L* approximations, labelled MODEL
- Columns 9:16 - McIlwain L, labelled MODEL_lm
- Columns 17:24 - Magnetic field strength B at location, labelled MODEL_b
- Columns 25:32 - Mapped magnetic equator Cartesian x, labelled MODEL_x
- Columns 33:40 - Mapped magnetic equator Cartesian y, labelled MODEL_y
- Columns 41:48 - Mapped magnetic equator Cartesian z, labelled MODEL_z
- Column 49 - Magnetic latitude of grid point, labelled mlat
- Column 50 - Magnetic longitude of grid point, labelled mlon
- Column 51 - Magnetic local time of grid point, labelled mlt

## The next planned developments are:
  - Create fast probabilistic time series for L* ground locations based on magnetic field model/combined probabilistic distributions
  - Query L* statistics (mean, median, interquartile range, occurrence) for a given location over a specific time period
  - Variable parameterisations based on solar wind variables and geomagnetic indices
  
For all correspondence please contact r.l.thompson@pgr.reading.ac.uk (including any questions, feature or collaboration requests, and bugs).
