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

def GAMA_catalog_w_photometry(datadir_spec='./', datadir_phot='./', outdir='./',
                              savefile=True):
    """
    Create final dataframe
    # if you want to remake this, download the SpecObj.fits from
    previous notebook instructions.
    # You can find the photo_noinfs.csv on
    Emille's computer at /media/CRP6/Cosmology/
    """
    path_specobj = os.path.join(datadir_spec, 'SpecObj.fits')
    with fits.open(path_specobj) as hdul:
        hdul.info()
        df_orig = pd.DataFrame(np.array(hdul[1].data).byteswap().newbyteorder())

    path_phot = os.path.join(datadir_phot, 'photo_noinfs.csv')
    phot = pd.read_csv(path_phot)

    new = df_orig.merge(phot, on='CATAID')
    df = new.drop(columns=new.columns[19])
    if savefile:
        df.to_csv(os.path.join(outdir, 'SpecObjPhot.csv'))

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

    survey_info = np.array([str(x).split()[0][2:] for x in df['SURVEY'].values])

    subsamples, lens = [], []

    # Create the subsample for the lowest bin
    subsamples.append(df.loc[(df['Z'] >= z_low)
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
        print('My bins have this many galaxies: ', z, lens)

    # Save the subsamples per redshift to be called later
    os.makedirs(os.path.join(datadir, 'SpecObjPhot'), exist_ok=True)
    for i, arr in enumerate(subsamples):
        path = os.path.join(datadir, 'SpecObjPhot/SpecObjPhot_%5.3f.csv' % z[i])
        arr.to_csv(path)

    return

def distance_bins(z, verbose=False, **kwargs):
    """
        I expect the bins to be a function of redshift in some way.
        This is a filler function so we have a way to plug and go if we change.
    """
    dmin= kwargs.pop('dmin', 0.2)
    dmax = kwargs.pop('dmax', 2.0)
    dnum = kwargs.pop('dnum', 10)

    try_distances = np.flip(np.geomspace(dmin, dmax, dnum), axis=0)

    if verbose:
       print(try_distances)

    return try_distances

def calculate_environment(df, z, savefile=True, outdir='./', verbose=False,
                          **kwargs):
    """
        Calculates the number of nearest neighbors per galaxy normalized by
        area/volume.
    """

    ## segment, sector, and area define the area to normalize the num neighbors
    def segment(r, d, theta=None):
        if theta == None:
            theta = 2. * np.arccos(d / r)
        return r**2 * (theta - np.sin(theta)) / 2.

    def sector(r, d, theta=None):
        if theta == None:
            theta = np.arcsin(d / r)
        return r**2 * theta / 2.

    # this throws an error at the points used to define minx, maxx, miny, maxy
    def area(r, x, y, minx, maxx, miny, maxy, vb=True):
        lx = x - minx
        ux = maxx - x
        ly = y - miny
        uy = maxy - y
        distances = np.array([lx, ux, ly, uy])
        condition = (distances >= r)
        ntrue = sum(condition)
        if ntrue == 4:
            return np.pi * r**2
        elif ntrue == 3:
            return np.pi * r**2 - segment(r, min(distances))
        elif ntrue == 2:
            if vb: print('radii should be chosen so that these cannot be parallel, \
                    but will at some point add in a check for this')
            distx = min(distances[:2])
            disty = min(distances[-2:])
            if np.sqrt(distx**2 + disty**2) < r:
                thetax = np.arcsin(distx / r)
                thetay = np.arcsin(disty / r)
                areax = distx * r * np.cos(thetax) / 2.
                areay = disty * r * np.cos(thetay) / 2.
                return sector(r, distx, theta=thetax) + sector(r, disty, theta=thetay) + \
                                sector(r, r, theta=np.pi / 2.) + distx * disty + areax + areay
            else:
                return np.pi * r**2 - segment(r, distx) - segment(r, disty)
        else:
            if vb: print('this case should not happen because we did not consider radii \
                    beyond half the shortest side of the footprint,\
                    but will at some point deal with this case')
            return None

    ## Calculates volume normalized environment
    def calc_env(ind):
        res = [subsamples[f][s]['CATAID'].values[ind]]
        friends = data
            for dist in try_distances:
                friends = galenv.nn_finder(friends, data[ind], dist)
                vol = area(dist, data[ind][0], data[ind][1], minx, maxx, miny, maxy, vb=False)
                res.append(float(len(friends)) / vol)
            return res

    z_bins = redshift_bins(z)

    RA_bin_ends = [0., 80., 160., 200., 360.]
    subsamples, lens = [], []
    field_bounds = []
    for i in range(len(RA_bin_ends)-1):
        one_field, one_len = [], []
        part_subsample = df.loc[(df['RA'] >= RA_bin_ends[i]) & (df['RA'] < RA_bin_ends[i+1])]
        (minx, maxx) = (np.floor(part_subsample['RA'].min()), np.ceil(part_subsample['RA'].max()))
        (miny, maxy) = (np.floor(part_subsample['DEC'].min()), np.ceil(part_subsample['DEC'].max()))
        for j in range(len(z_bins)-1):
            subsample = df.loc[(df['RA'] >= RA_bin_ends[i]) & (df['RA'] < RA_bin_ends[i+1])
                                 & (df['NQ'] > 2) & (df['Z'] >= z_bins[j]) & (df['Z'] < z_bins[j+1]),
                                 ['CATAID', 'RA', 'DEC', 'Z', 'NQ']]
            nn = len(subsample)
            if nn > 0:
                one_len.append(nn)
                one_field.append(subsample)
        subsamples.append(one_field)
        lens.append(one_len)
        field_bounds.append((minx, maxx, miny, maxy))

    ## Get the bins in angular distance
    try_distances = distance_bins(z, **kwargs)

    all_envs = []
    for f in range(len(subsamples)):
        (minx, maxx, miny, maxy) = field_bounds[f]
        assert(max(try_distances) <= min((maxx - minx), (maxy - miny)) / 2.)
        for s in range(len(subsamples[f])) :
            if verbose:
                print(lens[f][s])
            if lens[f][s] == 0:
                continue
            elif lens[f][s] == 1:
                envs_in_field = [[subsamples[f][s]['CATAID'].values[0]] + [1] * len(try_distances)]
            else:
                data = np.vstack(([subsamples[f][s]['RA'], subsamples[f][s]['DEC']])).T
                nps = mp.cpu_count()
                pool = mp.Pool(nps - 1)
                envs_in_field = pool.map(calc_env, range(len(data)))
                all_envs = all_envs + envs_in_field
                pool.close()

    envs_arr = np.array(all_envs)
    envs_df = pd.DataFrame(data=envs_arr, index = envs_arr[:, 0], columns = ['CATAID']+[str(i) for i in try_distances])

    df = pd.merge(envs_df, df, on='CATAID')

    if savefile=True:
        path = os.path.join(outdir, 'enviros.csv')
        df.to_csv(path)

    return df



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    loc_on_emilles_comp = '/media/CRP6/Cosmology/'

    df = GAMA_catalog_w_photometry(datadir_spec=loc_on_emilles_comp,
                                   datadir_phot=loc_on_emilles_comp)

    create_redshift_data(df, z_SLICS)

    env = calculate_environment(df, z_SLICS)





