# pro-lstar
Probabilistic L* mapping tool for ground and space observations

Several features of this package require the accompanying Pro-L* dataset:

Thompson, Rhys and Morley, Steven (2020): Pro-L*: probabilistic hourly L* values, with associated McIlwain Lm, magnetic field intensity B, and Cartesian coordinates for 7 global magnetic field models in the Northern Hemisphere in the period 2006-2016. University of Reading. Dataset. http://dx.doi.org/10.17864/1947.222

In its current iteration, Pro-L* allows users to:
  - Read in and plot L* for a given ground location (within ~2.5-10L in the Northern Hemisphere and the time frame given in the above dataset - refer for more details) for 7 magnetic field models or probabilisitc combinations of them
  - Create fast probabilistic time series for L* ground locations based on magnetic field model/combined probabilistic distributions
  - Query L* statistics (mean, median, interquartile range, occurrence) for a given location over a specific time period
  
Demo scripts for each of the above can be found in demo.py.
