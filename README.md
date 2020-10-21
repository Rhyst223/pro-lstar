# pro-lstar
Probabilistic L* mapping tool for ground and space observations

Required Python packages:
  - Pandas
  - Numpy 
  - Matplotlib
  - Urllib
  - io
  - gzip
  - itertools
  - scipy
  - copy
  

Several features of this package require the accompanying Pro-L* dataset:

Thompson, Rhys and Morley, Steven (2020): Pro-L*: probabilistic hourly L* values, with associated McIlwain Lm, magnetic field intensity B, and Cartesian coordinates for 7 global magnetic field models in the Northern Hemisphere in the period 2006-2016. University of Reading. Dataset. http://dx.doi.org/10.17864/1947.222


In its current iteration, Pro-L* allows users to:
  - Read in and plot Pro-L* variables for any given ground locations over a specific period (within ~2.5-10L in the Northern Hemisphere and the time frame given in the above dataset - refer for more details) for 7 magnetic field models, or simple probabilistic combinations of them. 
  
The model is simple to implement, with example code as follows:

from pro-lstar import *

df = get_period(start, end, MFmodel = 'all', variable = ['','lm','b','x','y','z'],model_threshold=4)

location  = df.get_location(66,4)

location.plot_location_data()


The next planned developments are:
  - Create fast probabilistic time series for L* ground locations based on magnetic field model/combined probabilistic distributions
  - Query L* statistics (mean, median, interquartile range, occurrence) for a given location over a specific time period
  - Variable parameterisations based on solar wind variables and geomagnetic indices
  
For all correspondence please contact r.l.thompson@pgr.reading.ac.uk (including any feature requests or collaboration).
