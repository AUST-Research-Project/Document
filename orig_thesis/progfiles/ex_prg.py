# coding: utf-8
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt


# reading in the data using PANDAS
# import pandas as pd
# data = pd.read_csv("weightHeight.dat",header=None,sep='\s+')
# PANDAS is an EXCEL-like processing... the "sep=..." specifies
# a regular expression that accepts multiple spaces

# this is a second alternative using numpy
data = np.loadtxt("weightHeight.dat")

# getting the INPUTS - x0 -- the heights, and the outputs y0 the WEIGHTS
x0   = data[:,1].reshape( (-1,1) )
y0   = data[:,0].reshape( (-1,1) )
# visualising the data
plt.scatter(x0, y0)

# to enrich the regression function, we define
#
# a FEATURE GENERATOR -- example of a function
def PHI(X):
    nData = X.shape[0]
    return np.concatenate( 
         ( np.ones( (nData,1) ), X, X ** 2, X ** 3, X ** 4 ),
           axis = -1
         )

### PERFORMING REGRESSION
X  = PHI(x0)
Y = y0
# solving the linear system
theta = np.linalg.lstsq(X , Y, rcond=None)[0]


xMin, xMax = min(x0), max(x0)
allX = np.linspace(xMin-20,xMax+20,50).reshape( (-1,1) )
allY = PHI(allX) @ theta

# second visuaisation
plt.plot( allX, allY,'r--')

