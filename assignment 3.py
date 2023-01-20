# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 00:03:16 2023

@author: akhil
"""

#importing packages
import pandas as pd
import numpy as np
import wbgapi as wb
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
from sklearn.cluster import KMeans
from scipy.stats import norm
import seaborn as sns
from scipy.optimize import curve_fit
import itertools as iter

#indicator' contains the indicator ID
#'country_code' contains the code of a few countries
indicator = ["SH.STA.TRAF.P5","SP.POP.TOTL.FE.ZS"]
country_code = ["AUS","CAN","IND","GBR","USA"]
# function to read dataframe in world format and return dataframe
def read(indicator,country_code):
    """
    Reads in a file in Worldbank format and returns a tuple containing the original dataframe and its transpose.

    Parameters:
        dataframe: The path to the file to be read

    Returns:
        tuple: A tuple containing the original dataframe
    """
    df = wb.data.DataFrame(indicator, country_code, mrv=30)
    return df
# creating a dataframe
data  = read(indicator, country_code)
# removing 'YR' and assigning new index names to dat
data.columns = [i.replace('YR','') for i in data.columns]
data=data.stack().unstack(level=1)
data.index.names = ['Country', 'Year']
# resetting the index for dt1
dt1 = data.reset_index()
print(dt1)
# converting the data type of "Year" from object to int64
dt1["Year"] = pd.to_numeric(dt1["Year"])
# function to normalise the datas in the dataframe
def norm_df(dt1):
    y = dt1.iloc[:,2:]
    dt1.iloc[:,2:] = (y-y.min())/ (y.max() - y.min())
    return dt1
# normalised dataframe
dt_norm = norm_df(dt1)
dt_norm.dropna(inplace=True)
print(dt_norm)
# dataframe that is needed to do clustering
df_fit = dt_norm.drop('Country', axis = 1)
# need to create new dataframe with the 2 columns for clustering
# (or 3 or 4 columns if desired)
df_ex = dt_norm[["SH.STA.TRAF.P5", "SP.POP.TOTL.FE.ZS"]].copy()
# min and max operate column by column by default
max_val = df_ex.max()
min_val = df_ex.min()
# set up the clusterer for number of clusters
ncluster = 5
kmeans = cluster.KMeans(n_clusters=ncluster)
# Fit the data, results are stored in the kmeans object
kmeans.fit(df_ex) # fit done on x,y pairs
labels = kmeans.labels_
# print(labels) # labels is the number of the associated clusters of (x,y)points
# extract the estimated cluster centres
cen = kmeans.cluster_centers_
# calculate the silhoutte score
print(skmet.silhouette_score(df_ex, labels))
# plot using the labels to select colour
plt.figure(figsize=(15.0, 10.0))
col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
for l in range(ncluster): # loop over the different labels
    plt.plot(df_ex[labels==l]["SH.STA.TRAF.P5"], df_ex[labels==l]["SP.POP.TOTL.FE.ZS"],"o", markersize=5, color=col[l])
#
# show cluster centres
for ic in range(ncluster):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=15)
plt.title("5 Clusters",size=20)
plt.xlabel("Mortality caused by road traffic injury (per 100,000 population)",size=15)
plt.ylabel("Population, female (% of total population)",size=15)
plt.show()
df_cen = pd.DataFrame(cen, columns=["SP.POP.TOTL.FE.ZS", "SH.STA.TRAF.P5"])
print(df_cen)
df_cen = df_cen * (max_val - min_val) + max_val
df_ex = df_ex * (max_val - min_val) + max_val
print(df_ex.min(), df_ex.max())
print(df_cen)
print(cen)

"""
CURVE FIT

"""
# function to calculate the error range
def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.

    This routine can be used in assignment programs.
    """
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower

    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper
# dataframe containing the datas of the country Australia
dt2 = dt1[(dt1['Country'] == 'AUS')]
dt2.dropna(inplace=True)
#curve_fit function implementation for Australia

val = dt2.values
x, y = val[:, 1], val[:, 2]

def fct(x, a, b, c):
    return a*x**2+b*x+c
param, cov = opt.curve_fit(fct, x, y)

dt2["pop_log"] = fct(x, *param)
print("Parameters are: ", param)
print("Covariance is: ", cov)

plt.plot(x, dt2["pop_log"], label="Fit")
plt.plot(x, y, label="Data")
plt.grid(True)
plt.xlabel("Year")
plt.ylabel("Mortality caused by road traffic injury ")
plt.title("Mortality caused by road traffic injury in Australia")
plt.legend(loc='best', fancybox=True, shadow=True)
plt.show()
# extract the sigmas from the diagonal of the covariance matrix
sigma = np.sqrt(np.diag(cov))
print(sigma)
print()
print(f"N0 = {param[0]:5.1f} +/- {sigma[0]:3.1f}")
print(f"t_1/2 = {param[1]:4.2f} +/- {sigma[1]:4.2f}")
# Forcasting the emission rate in the coming 10 years
print("Mortality caused by road traffic injury")
low, up = err_ranges(2030, fct, param, sigma)
print("2030 between ", low, "and", up)

# dataframe containing the datas of the country India
dt3 = dt1[(dt1['Country'] == 'IND')]
dt3.dropna(inplace=True)
#curve_fit function implementation for India

val = dt3.values
x, y = val[:, 1], val[:, 2]

def fct(x, a, b, c):
    return a*x**2+b*x+c
param, cov = opt.curve_fit(fct, x, y)

dt3["pop_log"] = fct(x, *param)
print("Parameters are: ", param)
print("Covariance is: ", cov)

plt.plot(x, dt3["pop_log"], label="Fit")
plt.plot(x, y, label="Data")
plt.grid(True)
plt.xlabel("Year")
plt.ylabel("Mortality caused by road traffic injury ")
plt.title("Mortality caused by road traffic injury in India")
plt.legend(loc='best', fancybox=True, shadow=True)
plt.show()
# extract the sigmas from the diagonal of the covariance matrix
sigma = np.sqrt(np.diag(cov))
print(sigma)
print()
print(f"N0 = {param[0]:5.1f} +/- {sigma[0]:3.1f}")
print(f"t_1/2 = {param[1]:4.2f} +/- {sigma[1]:4.2f}")
# Forcasting the emission rate in the coming 10 years
print("Mortality caused by road traffic injury")
low, up = err_ranges(2030, fct, param, sigma)
print("2030 between ", low, "and", up)

# dataframe containing the datas of the country United Kingdom

dt3 = dt1[(dt1['Country'] == "GBR")]
dt3.dropna(inplace=True)
#curve_fit function implementation for United Kingdom

val = dt3.values
x, y = val[:, 1], val[:, 2]

def fct(x, a, b, c):
    return a*x**2+b*x+c
param, covar = opt.curve_fit(fct,x,y, absolute_sigma=False)
dt3["pop_log"] = fct(x, *param)
print("Parameters are: ", param)
print("Covariance is: ", cov)

plt.plot(x, dt3["pop_log"], label="Fit")
plt.plot(x, y, label="Data")
plt.grid(True)
plt.xlabel("Year")
plt.ylabel("Mortality caused by road traffic injury ")
plt.title("Mortality caused by road traffic injury in United Kingdom")
plt.legend(loc='best', fancybox=True, shadow=True)
plt.show()
# extract the sigmas from the diagonal of the covariance matrix
sigma = np.sqrt(np.diag(cov))
print(sigma)
print()
print(f"N0 = {param[0]:5.1f} +/- {sigma[0]:3.1f}")
print(f"t_1/2 = {param[1]:4.2f} +/- {sigma[1]:4.2f}")
# Forcasting the emission rate in the coming 10 years
print("Mortality caused by road traffic injury")
low, up = err_ranges(2030, fct, param, sigma)
print("2030 between ", low, "and", up)

