#!/usr/bin/env python

import numpy as np
import os
import glob
import pandas as pd
import pickle
import multiprocessing as mp

import scipy.optimize as spo
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib

from astropy.io import fits
import corner
import environment as galenv

import pystan

from tslearn.clustering import TimeSeriesKMeans

np.seed = 42

## Redshift bins from SLICS N-body sims
z_SLICS = np.array([0.042, 0.080, 0.130, 0.221, 0.317, 0.418, 0.525, 0.640,
                    0.764, 0.897, 1.041, 1.199, 1.372, 1.562, 1.772, 2.007,
                    2.269, 2.565, 2.899])

## Define a color map
colmap = 'Spectral_r'

def color_map(dat, map, vmin, vmax):
    """
        Placeholder color map
    """
    jet = plt.cm.get_cmap(map)
    cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    color=scalarMap.to_rgba(dat)
    return color

def GAMA_catalog_w_photometry(datadir_spec='./', datadir_phot='./'):
    """
    Create final dataframe
    # if you want to remake this, download the SpecObj.fits from
    previous notebook instructions.
    # You can find the photo_noinfs.csv on
    Emille's computer at /media/CRP6/Cosmology/
    """
    path_specobj = os.path.join(datadir_spec, 'photo_noinfs.csv')
    with fits.open('SpecObj.fits') as hdul:
        hdul.info()
        df_orig = pd.DataFrame(np.array(hdul[1].data).byteswap().newbyteorder())
        df_orig.index = df_orig['CATAID']

    path_phot = os.path.join(datadir_phot, 'photo_noinfs.csv')
    phot = pd.read_csv(path_phot)
    phot.index = phot['CATAID']

    new = df_orig.merge(phot, on='CATAID')
    df = new.drop(columns=new.columns[19])
    df.to_csv('SpecObjPhot.csv')

    return df


def redshift_bins(z, z_low=0.023, z_high=3.066):
    """
        Input: Array of lists of redshifts matching N-body data
        Output: Redshift bins for real data
        
        z_low: lower limit of lowest redshift bin. Default calculated for SLICS
        z_high: higher limit of highes redshift bin. Default calculated for SLICS
    """
    z_mids = (z[1:] + z[:-1]) / 2.
    z_bins = np.insert(z_mids, 0, z_low)
    z_bins = np.append(z_bins, z_high)
    return z_bins

def create_redshift_data(df, z, datadir='./', verbose=False, **kwargs):
    """
        Separates the galaxy data into redshift bins and saves the catalog
        into a pandas dataframe located in SpecObjPhot/
        
        Input
        df: Pandas data frame with catalog info ('SURVEY', 'Z') and photometry.
        z: redshift array
        datadir: location of output data
        verbose: Will bring the number of galaxies in each redshift bin
        kwargs: z_low, z_high for input into redshift_bins(z,z_low,z_high)
        
        Output:
        Returns nothing but creates directories contained the redshift bin files
        
    """
    z_low = kwargs.pop('z_low', 0.023) # default is SLICS
    z_high = kwargs.pop('z_high', 3.066) # default is SLICS
    
    endpoints = redshift_bins(z, z_low=z_low, z_high=z_high)[1:]
    
    survey_info = np.array([x.split("'")[1].strip() for x in df['SURVEY'].values])
    
    subsamples, lens = [], []

    # Create the subsample for the lowest bin
    subsamples.append(df.loc[(df['Z'] >= zbin[0])
                             & (df['Z'] < endpoints[0])
                             & (df['NQ'] > 2)
                             & ((df['lsstg'] > 0) |
                                (df['lsstr'] > 0) |
                                (df['lsstz'] > 0) |
                                (df['lssty'] > 0))
                             & ((survey_info == 'GAMA') |
                                (survey_info == 'SDSS') |
                                (survey_info == 'VVDS')
                                )])

    lens.append(len(subsamples[-1]))

    # create all other subsamples
    for i in np.arange(0, len(endpoints) -1 ):
        subsamples.append(df.loc[(df['Z'] >= endpoints[i])
                                 & (df['Z'] < endpoints[i+1])
                                 & (df['NQ'] > 2)
                                 & ((df['lsstg'] > 0) |
                                    (df['lsstr'] > 0) |
                                    (df['lsstz'] > 0) |
                                    (df['lssty'] > 0))
                                 & ((survey_info == 'GAMA') |
                                    (survey_info == 'SDSS') |
                                    (survey_info == 'VVDS')
                                   )])

        lens.append(len(subsamples[-1]))

    if verbose:
        print('My bins have this many galaxies: ' zbins, lens)

    # Save the subsamples per redshift to be called later
    os.makedirs(os.path.join(datadir, 'SpecObjPhot'), exist_ok=True)
    for i, arr in enumerate(subsamples):
        path = os.path.join(datadir, 'SpecObjPhot/SpecObjPhot_%5.3f.csv' % z[i])
        arr.to_csv(path)

    return



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    loc_on_emilles_comp = '/media/CRP6/Cosmology/recidivator/GAMA/'

    df = GAMA_catalog_w_photometry(datadir_spec=loc_on_emilles_comp,
                                   datadir_phot=loc_on_emilles_comp)

    create_redshift_data(df)





