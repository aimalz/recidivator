#!/usr/bin/env python

import numpy as np
import os
import glob
import pandas as pd
import pickle
import multiprocessing as mp

import scipy.optimize as spo
from scipy.stats import multivariate_normal
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib

from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import corner
import environment as galenv

import pystan

from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

np.seed = 42

## Redshift bins from SLICS N-body sims
z_SLICS = np.array([0.042, 0.080, 0.130, 0.221, 0.317, 0.418, 0.525])
#, 0.640, 0.764, 0.897, 1.041, 1.199, 1.372, 1.562, 1.772, 2.007,
#2.269, 2.565, 2.899])

# Define cosmology
cosmo = FlatLambdaCDM(H0=69.98, Om0=0.2905, Ob0=0.0473)

## Define a color map
colmap = 'Spectral_r'

def color_map(dat, vmin, vmax, map=colmap):
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
        df_orig = pd.DataFrame(np.array(hdul[1].data).byteswap().newbyteorder())

    path_phot = os.path.join(datadir_phot, 'photo_noinfs.csv')
    phot = pd.read_csv(path_phot)

    new = df_orig.merge(phot, on='CATAID')
    df = new.drop(columns=new.columns[19])
    if savefile:
        df.to_csv(os.path.join(outdir, 'SpecObjPhot.csv'))

    return df


def redshift_bins(z, z_low=0.023, z_high=3.066, nslice=None):
    """
        Input: Array of lists of redshifts matching N-body data
        Output: Redshift bins for real data

        z_low: lower limit of lowest redshift bin. Default calculated for SLICS
        z_high: higher limit of highes redshift bin. Default calculated for SLICS
    """
    if not nslice:
        z_mids = (z[2:] + z[1:-1]) / 2.
        z_bins = np.insert(z_mids, 0, 0.05949)
        z_bins = np.insert(z_bins, 0, z_low)
        z_bins = np.append(z_bins, z_high)

    elif 'one_slice' in nslice:
        z_bins = np.array([0.031915757912297296, 0.05213079888495292,
                           0.0697375795614684, 0.09031243711212583,
                           0.11948849618564256, 0.1405662608081968,
                           0.20999304440408048, 0.2320708833705863,
                           0.30541295361458104, 0.3286614228895124,
                           0.4057415280014885, 0.43034469547157433,
                           0.5119651965968968, 0.5381345176677429])

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
    nslice = kwargs.pop('nslice', None) # default is SLICS

    survey_info = np.array([str(x).split()[0][2:] for x in df['SURVEY'].values])

    subsamples, lens = [], []

    if not nslice:
        endpoints = redshift_bins(z, z_low=z_low, z_high=z_high, nslice=nslice)[1:]

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
        for i in np.arange(0, len(endpoints)-1):
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

    elif 'one_slice' in nslice:
        endpoints = redshift_bins(z, z_low=z_low, z_high=z_high, nslice=nslice)

        for i in np.arange(0, len(endpoints), 2):
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

def redshift_df(str_z, zenvdf, datadir='./'):
    path = os.path.join(datadir, 'SpecObjPhot/SpecObjPhot*.csv')
    r_files = glob.glob(path)
    f = [s for s in r_files if str_z in s]
    phodf = pd.read_csv(f[0])
    phodf = phodf.drop(columns=['GAMA_NAME', 'IC_FLAG',
                                'N_SPEC', 'N_GAMA_SPEC', 'DIST',
                                'SPECID', 'SURVEY', 'SURVEY_CODE',
                                'RA', 'DEC', 'WMIN', 'WMAX', 'Z', 'NQ',
                                'PROB', 'FILENAME', 'URL', 'URL_IMG',
                                'lsstg', 'lsstg_err', 'lsstr', 'lsstr_err',
                                'lssti', 'lssti_err', 'lsstz', 'lsstz_err',
                                'lssty', 'lssty_err', 'sdssr', 'sdssr_err',
                                'sdssi', 'sdssi_err', 'sdssz', 'sdssz_err'])
    df = pd.merge(phodf, zenvdf, on=['CATAID'])
    return df

def distance_bins(z, btype, n=10, verbose=False, **kwargs):
    """
        Returns list of angular distances.

        Input:
        z : array of redshift bins
        btype : kind of bins to use.
        options: 'angular'       - use same angular distance over all z, n bins
        'physical'  - use same physical coordinates over all z, n bins total.
                     low z does not have all n but a fraction
                     and high-z has all n.
        'changez'   - for each redshift bin, find the angular distance
                      corresponding to 1 Mpc and the one that either corresponds
                      to 100 Mpc or 2.5 deg (whichever is smaller) and create
                      n number of bins per redshift.
        n : number of distances
        verbose : prints the list of distances returned.
        kwargs: for the no redshift (angular) option, dmin and dmax specifies the
        upper and lower limit of the distances.

        Output:
        list of angular distances
        """

    phys_anchors = [1., 10., 100.]
    dc = cosmo.comoving_distance(z)
    da = dc / (1. + z)
    ang_anchor = phys_anchors / da * 180. / np.pi

    if 'angular' in btype:
        dmin= kwargs.pop('dmin', 0.2)
        dmax = kwargs.pop('dmax', 2.5)
        try_distances = np.flip(np.geomspace(dmin, dmax, n), axis=0)

    if 'physical' in btype:
        phys_spacing = np.flip(np.geomspace(1., 55., n), axis=0)
        ang_spacing = phys_spacing / da.value * 180. / np.pi
        if 'physical' in kwargs.keys():
            return phys_spacing
        try_distances = ang_spacing[np.where(ang_spacing < 2.5)[0]]

    if 'changez' in btype:
        try_distances = np.flip(np.geomspace(min(ang_anchor.value),
                                             min(2.5, max(ang_anchor.value)),
                                             n), axis=0)

    if verbose:
        print(try_distances)

    return try_distances

### This group of functions is to calculate environment
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
    """
        Runs galenv to calculate galaxy environment.
        This is set up to run in the multiprocessing so a lot of inputs are
        not set when you call the function, but are supposed to be defined
        when running this code.

        Output: nearest neighbors at a given angular separation.
    """
    if opts.run_environment:
        # Generates environments for GAMA RA/Dec data
        res = [subsamples[f][s]['CATAID'].values[ind]]
    if opts.run_particle_environment:
        # Generates environments for particle RA/Dec data
        res = [ind]

    friends = data
    for dist in try_distances:
        friends = galenv.nn_finder(friends, data[ind], dist)
        vol = area(dist, data[ind][0], data[ind][1], minx, maxx, miny, maxy, vb=False)
        res.append(float(len(friends)) / vol)
    return res
###

### Clustering function
def deprecated_run_clustering(str_z, zenvdf, n_clusters=10, metric="euclidean",
                   max_iter=5, random_state=0, savefiles=True, outdir='./',
                   **kwargs):
    """
        info - may change waiting
    """
    btype = kwargs.pop('btype', 'angular')
    n = kwargs.pop('n', 10)

    try_distances = distance_bins(float(str_z), btype=btype, n=n)
    str_tryd = [str(i) for i in np.arange(1, len(try_distances)+1)]

    df = redshift_df(str_z, zenvdf)
    if float(str_z) < 0.3:
        df = df.loc[(df['lsstr'] > 0) & (df['lssti'] > 0) & (df['lsstz'] > 0)]
    else:
        df = df.loc[(df['lssti'] > 0) & (df['lsstz'] > 0) & (df['lssty'] > 0)]

    # the three available filters
    X = df[str_tryd].values

    km = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric,
                          max_iter=max_iter, random_state=random_state).fit(X)

    df['label'] = km.labels_

    if float(str_z) < 0.3:
        df['mag3'] = df['lsstr']
    else:
        df['mag3'] = df['lssty']

    if savefiles:
        path_dir = os.path.join(outdir, 'ts_kmeans')
        os.makedirs(path_dir, exist_ok=True)
        filename = 'ts_kmeans/tskmeans_%s.pkl' % str_z
        path_filename = os.path.join(outdir, filename)
        pickle.dump(km, open(path_filename, 'wb'))

        path = os.path.join(outdir, 'Redshift_df_label_%s.csv' % str_z)
        df.to_csv(path)

    return df

### Make populations
def create_fit_summaries(df, lbl, str_z, patch, gnum,
                         iter=1000, chains=4, warmup=500,
                         savefiles=True, outdir='./'):
    """
        Takes in a pandas data frame with 2 colors and a
        magnitude and returns the means and covariance matrix
        of the corresponding 3D Gaussian distribution.

        Input
        df :
        lbl : label
        str_z :
        patch :
        gnum :

        iter :
        chains :
        warmup :
        savefiles :
        outdir :

        Output
        data frame containing means and covariance matrix

        TBD: generalize per redshift bin
    """
    label_name = 'class_%sgroups' % gnum

    df_l = df.loc[(df[label_name]==int(lbl)) & (df['patch']==int(patch))]

    N = len(df_l)

    vals = df_l[['lssti', 'lsstz', 'mag3']].values

    N_re = 1

    dat = {
        'N': N,
        'D': 3,
        'N_resamples': N_re,
        'tavy': vals,
    }

    ## You only need this ONE model to fix all of the data no matter what label or redshift you have!
    try:
        sm = pickle.load(open('3D_Gaussian_model.pkl', 'rb'))
    except:
        sm = pystan.StanModel(file='some_dimensional_gaussian.stan')
        # Save the model so that you don't have to keep regenerating it.
        with open('3D_Gaussian_model.pkl', 'wb') as f:
            pickle.dump(sm, f)

    fit = sm.sampling(data=dat, iter=iter, chains=chains, warmup=warmup)

    summary = pd.DataFrame(fit.summary(pars=['mu', 'Sigma'])['summary'],
                           columns=fit.summary(pars=['mu', 'Sigma'])['summary_colnames'])

    summary['Names'] = fit.summary(pars=['mu', 'Sigma'])['summary_rownames']

    if savefiles:
        path_dir = os.path.join(outdir, 'fit_summaries')

        path_files = os.path.join(path_dir,
                                  'summary_%s_label_%s_patch_%s_groups_%s.csv'
                                  % (str_z, lbl, patch, gnum))

        os.makedirs(path_dir, exist_ok=True)
        summary.to_csv(path_files)

    return summary


## Given environment curves for particle data return galaxy properties
def deprecated_get_label(n_r, str_redshift, verbose=False, indir='./'):
    """
        n_r: array of number of neighbors at same radii as TS Kmeans was trained
        redshift: redshift bin to get model

        returns: label in an array
    """

    filename = 'ts_kmeans/tskmeans_%s.pkl'% str_redshift
    path = os.path.join(indir, filename)
    km = pickle.load(open(path, 'rb'))
    if len(n_r.shape) == 1:
        predicted = km.predict(n_r.reshape(1, -1))
    else:
        predicted = km.predict(n_r)

    if verbose:
        print("Predicted label is: ", predicted)

    return predicted

def assign_cluster(envcurve, patch, gnum, redshift, modeldir, btype):
    """Assign one new environmental curve to one of the clusters.

        Parameters
        ----------
        envcurve: list or np.array
            Number of neighbors in each distance bin.
            Dimensionality will be different for different redshifts.
        patch: int
            Patch in the sky. Possibilities are [0,1,2,3].
        redshift: str or float
            Redshift bin.
        modeldir: str
            Directory where trained models are stored.
        gnum: int
            Number of clusters
        Returns
        -------
        group: int
            Group to which the new environmental curve belongs.
    """

    # calculate distance bins given redshift
    dbins = list(distance_bins(float(redshift), btype=btype))
    dbins.reverse()

    # uniform xaxis
    dz = (dbins[-1] - dbins[0])/(len(dbins)+1)
    x2 = np.arange(dbins[0], dbins[-1]+dz+0.0001, 0.0001)

    filename = modeldir + 'classifiers/' + str(redshift) + \
               '/' + 'model_z_' + str(redshift) + \
               '_patch' + str(patch) + '_' + str(gnum) + 'groups.pkl'

    ts = TimeSeriesKMeans()
    loaded_model = ts.from_pickle(filename)

    all_group = []
    for env in envcurve:
        # interpolate each environment curve
        y = list(env)
        y.reverse()

        tck = UnivariateSpline(dbins, y,s=0)
        intp_curv = tck(x2)

        # add zero point to data
        deriv = [env[0]]

        # calculate derivative
        for k in range(len(intp_curv) - 1):
            deriv.append((intp_curv[k + 1] - intp_curv[k])/(x2[k + 1] - x2[k]))

        group = loaded_model.predict(to_time_series_dataset(deriv).reshape(1,1,len(deriv[0])))
        all_group.append(group[0])

    return all_group


def get_random_sample(label, str_redshift, patch, group_num, indir='./'):
    """
        label: Cluster label
        str_redshift: redshift bin to get model as a string
        patch: patch number
        group_num: number of clusters
        indir: Where to find the fit summaries

        returns: random draws in an array
    """

    from scipy.stats import multivariate_normal

    rando = []
    for l in label:
        path = os.path.join(indir,
                            'fit_summaries/summary_%s_label_%s_patch_%s_groups_%s.csv'
                            % (str_redshift, label[0], patch, group_num))
        summary = pd.read_csv(path)
        mus = summary.iloc[:3]['mean'].values
        cov = summary.iloc[3:]['mean'].values.reshape(3, 3)

        rando.append(multivariate_normal.rvs(mus, cov))
    return rando


def get_properties(n_r, str_redshift, patch, gnum,
                   btype='physical',
                   ra=None, dec=None,
                   verbose=False,
                   modeldir='./', indir='./',
                   savefiles=True, outdir='./', **kwargs):
    """
        Combines chosing label and generating a random sample to
           generate the mock catalog

        n_r: array of number of neighbors at same radii as TS Kmeans was trained
        str_redshift: redshift bin to get model as a string
        patch: patch that the density was modeled from
        btype:
        ra : RA of each particle if wanted in output
        dec : Dec of each particle if wanted in output
        verbose: print label
        indir: Where to find fit_summaries
        modeldir: Where to find cluster models
        savefiles: Save the mock catalog
        outdir: location to save the mock catalog

        returns: Mock catalog
    """

    #l = get_label(n_r, str_redshift, verbose=verbose, indir=indir)

    l = assign_cluster(n_r, patch=patch, gnum=gnum,
                       redshift=str_redshift,
                       modeldir=modeldir,
                       btype=btype)

    samp = get_random_sample(l, str_redshift, patch, gnum,
                            indir=indir)

    if ra is not None:
        radec = np.vstack((ra, dec)).T
        samp1 = np.concatenate((radec, samp), axis=1)
        samp_pd = pd.DataFrame(samp1, columns=['ra', 'dec', 'lssti', 'lsstz', 'mag3'])
    else:
        samp_pd = pd.DataFrame(samp, columns=['lssti', 'lsstz', 'mag3'])

    if float(str_redshift) < 0.3:
        samp_pd.rename(columns={'mag3': 'mag_r_lsst',
                                'lssti': 'mag_i_lsst',
                                'lsstz': 'mag_z_lsst'}, inplace=True)
    else:
        samp_pd.rename(columns={'lssti': 'mag_i_lsst',
                                'lsstz': 'mag_z_lsst',
                                'mag3': 'mag_y_lsst'}, inplace=True)

    samp_pd['group_label'] = l

    if savefiles:
        path = os.path.join(outdir, 'results_%s_patch_%s_groups_%s.csv' % (str_redshift, patch, gnum))
        samp_pd.to_csv(path, index=False)

    return samp_pd


### Plotting routines
def make_orchestra(z, zenvdf, btype='angular', savefig=True, verbose=False):
    """
        Generates the orchestra plot for a given bin type

        z: List of redshifts to include
        zenvdf: environment file containing the nearest neighbors (run_env)
        btype: Type of binning used to probe the  nearest neighbors
        savefig: save the figure
        verbose: print warnings

        returns: Plot
    """

    color = color_map(z, vmin=z[0], vmax=z[-1])
    cNorm  = colors.Normalize(vmin=z[0], vmax=z[-1])

    fig, ax = plt.subplots(figsize=(15,10))

    for n, z in enumerate(z):
        try_distances = distance_bins(z, btype=btype)
        orig_distances = np.flip(try_distances, axis=0)

        inx_td = [str(i) for i in np.arange(1, len(try_distances)+1)]
        orig_inx_td = np.flip(inx_td, axis=0)

        df = redshift_df(str(z), zenvdf)
        if len(df) > 0:
            for i in range(len(orig_distances)):
                if 'changez' in btype:
                    parts = ax.violinplot(df[orig_inx_td[i]], positions=[orig_distances[i]])
                else:
                    parts = ax.violinplot(df[orig_inx_td[i]], positions=[i])

                c = color[n]
                for pc in parts['bodies']:
                    pc.set_facecolor(c)
                    pc.set_edgecolor(c)
                    pc.set_alpha(0.5)

                parts['cbars'].set_color(c)
                parts['cbars'].set_alpha(0.4)

                parts['cmaxes'].set_color(c)
                parts['cmaxes'].set_alpha(0.4)

                parts['cmins'].set_color(c)
                parts['cmins'].set_alpha(0.4)
        else:
            if verbose:
                print("I have nothing for you at n=%s, z=%s"%(n,z))

    if 'physical' in btype:
        plt.xticks(range(len(orig_distances)), np.around(np.flip(distance_bins(z, btype='physical', physical=True)), 3))
        ax.set_xlabel('physical distance [Mpc]', size=15)
    elif 'changez' in btype:
        ax.set_xlabel('radial distance [deg]', size=15)
    else:
        plt.xticks(range(len(orig_distances)), np.around(orig_distances, 3))
        ax.set_xlabel('radial distance [deg]', size=15)

    ax.semilogy()
    ax.set_ylabel('Normalized number of neighbors', size=15)

    cax, _ = matplotlib.colorbar.make_axes(ax, pad=0.01)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=plt.cm.Spectral_r, norm=cNorm)
    cbar.ax.set_ylabel('redshift', size=12)

    if savefig:
        plt.savefig('orchestra_neighbor_v_distance.pdf')
    return


def make_ang_phys_plot(z_SLICS, n=10, savefig=True):
    """
        Generates the plot illustrating different bin typs

        z_SLICS: List of redshifts to include
        n: Number of radial bins to try to include
        savefig: save the figure

        returns: Plot
    """

    fig, ax = plt.subplots()
    ang_dist_grid = []

    color = color_map(z_SLICS, vmin=z_SLICS[0], vmax=z_SLICS[-1])
    cNorm  = colors.Normalize(vmin=z_SLICS[0], vmax=z_SLICS[-1])

    phys_anchors = [1., 10., 100.]
    ang_anchors = []
    for z in z_SLICS:
        dc = cosmo.comoving_distance(z)
        da = dc / (1 + z)
        ang_anchor = phys_anchors / da * 180. / np.pi
        ang_anchors.append(ang_anchor)

    for i in range(len(z_SLICS)):
        try_distances = np.flip(np.geomspace(min(ang_anchors[i].value),
                                             min(2.5, max(ang_anchors[i].value)), 100), axis=0)

        dc = cosmo.comoving_distance(z_SLICS[i])
        da = dc / (1 + z_SLICS[i])
        ax.plot(np.flip(try_distances, axis=0) * da * np.pi / 180.,
                np.flip(try_distances, axis=0), color=color[i], lw=3)

    ax.semilogx()

    for i in np.flip(distance_bins(z_SLICS[0], n=n,
                                   btype='physical', physical=True), axis=0):
        ax.axvline(i, c='grey', alpha=0.7)

    for j in np.flip(distance_bins(z_SLICS[0], n=n,
                     btype='angular'), axis=0):
           ax.axhline(j, ls='--', c='grey', alpha=0.4)

    ax.set_ylabel('Angular distances', fontsize=13)
    ax.set_xlabel('Physical scales [Mpc]', fontsize=13)

    plt.setp(ax, xticks=([1, 2, 5, 10, 25, 60]), xticklabels=([1, 2, 5, 10, 25, 60]))

    cax, _ = matplotlib.colorbar.make_axes(ax, pad=0.01)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=plt.cm.Spectral_r, norm=cNorm)
    cbar.ax.set_ylabel('redshift', size=12)
    if savefig:
        plt.savefig('physical_scale_v_angular_distance.pdf')
    return

def compare_environ_curves(str_red, zenvdf, btype, n=10, num=4000, indir='./', savefig=True):
    """
        Generates the plot that compares environment curves of simulations
        to environment curves from GAMA at a given redshift.

        str_red: Single redshift as a string
        zenvdf: environment file containing the nearest neighbors from particle
                data (run_particle)
        btype: Type of binning used to probe the  nearest neighbors
        n: number of bins
        num: Number of random samples
        indir: Location of particle data
        savefig: save the figure

        returns: Plot
    """

    df = redshift_df(str_red, zenvdf)

    path_partenviros = os.path.join(indir,
                                    'particle_enviros_%s.csv' % str_red)
    envs_df = pd.read_csv(path_partenviros)
    envs_in_field = envs_df.to_numpy()

    try_distances = distance_bins(float(str_red),
                                  btype=btype,
                                  n=n)
    orig_distances = np.flip(try_distances, axis=0)

    fig = plt.figure(figsize=(11,10))
    for i in range(int(len(envs_in_field[:,1:])/3.)):
        plt.loglog(try_distances, envs_in_field[i,1:], alpha=0.1, color='r')

    # repeat of the last one to get the legend
    plt.loglog(try_distances, envs_in_field[i,1:],
               alpha=0.1, color='r', label='Simulation')

    # Randomly choose 4000 to match the number chosen from the particle data
    ## but only plot the same number as the number plotted for the sims
    rand_gals = np.random.uniform(low=0, high=len(df), size=num)

    for index, row in df.iloc[rand_gals].iterrows():
        if index < int(len(envs_in_field[:,1:])/3.):
            dist = [str(i) for i in np.arange(len(orig_distances))]
            plt.loglog(orig_distances, row[dist], 'o-k', alpha=0.2)

    # repeat of the last one to get the legend
    plt.loglog(orig_distances, row[dist],
               'o-k', alpha=0.2, label='Real Galaxies')

    plt.xlabel('radial distance', size=15)
    plt.ylabel('# neighbors', size=15)
    plt.legend()

    if savefig:
        plt.savefig('environ_curve_sim_v_real_%s.pdf' % str_red)
    return


def make_corner(gama, mock, str_red, savefig=True):
    """
        * this is a first draft and untested *
        Generates the corner plot that compares GAMA magnitudes to the
        mock magnitdues.

        str_red: Single redshift as a string
        gama: pandas dataframe of GAMA data
        mock: array of mock data
        savefig: save the figure

        returns: Plot
    """

    jet = plt.cm.Spectral
    cNorm  = colors.Normalize(vmin=0, vmax=20)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    color=scalarMap.to_rgba(np.arange(0, 20))

    if float(str_red) < 0.3:
        mag3 = 'lsstr'
    else:
        mag3 = 'lssty'
    fig = corner.corner(np.array([gama.loc[(gama['lsstz'] > 0)
                                              & (gama[mag3] > 0)
                                              & (gama['lssti'] > 0)][mag3],
                                  gama.loc[(gama['lsstz'] > 0)
                                              & (gama[mag3] > 0)
                                              & (gama['lssti'] > 0)]['lssti'],
                                  gama.loc[(gama['lsstz'] > 0)
                                              & (gama[mag3] > 0)
                                              & (gama['lssti'] > 0)]['lsstz'],
                                  ]).T,
                        #labels=['r-', 'i', 'z'], show_titles=True,
                        range = [(-1.6,1.6), (-1.6,1.6),(15,21)],
                        color=color[1],
                        plot_density=False,
                        plot_contours=False,
                        quantiles=[0.5],
                        hist_kwargs={'density': True})

    corner.corner(mock,
                  labels=[mag3, 'lssti', 'lsstz'],
                  show_titles=True,
                  color=color[-3], fig=fig,
                  plot_contours=False,
                  plot_density=False,
                  quantiles=[0.5],
                  hist_kwargs={'density': True})

    if savefig:
        plt.savefig('corner_plot_%s.pdf'  % str_red)
    return

#### End Plotting routines

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--no_files', dest='savefile',
                        default=True, action='store_false')
    parser.add_argument('--outdir', default='./')
    parser.add_argument('--modeldir',
                        default='/media/CRP6/Cosmology/environmet_clustering/')
    parser.add_argument('--radii', dest='radial_binning',
                        default='angular')
    parser.add_argument('--slices', dest='nslice',
                        default=None)
    parser.add_argument('--bins', dest='n',
                        default=10)
    parser.add_argument('--create_red', dest='create_redshift',
                        default=False, action='store_true')
    parser.add_argument('--run_env', dest='run_environment',
                        default=False, action='store_true')
    parser.add_argument('--gen_summaries', dest='generate_fit_summaries',
                        default=False, action='store_true')
    parser.add_argument('--run_particle', dest='run_particle_environment',
                        default=False, action='store_true')
    parser.add_argument('--particle_file', dest='particle_file',
                        default='ang2deg.csv')
    parser.add_argument('--patch', dest='patch',
                        default=1)
    parser.add_argument('--groups', dest='gnum',
                        default=2)

    opts = parser.parse_args()

    loc_on_emilles_comp = '/media/CRP6/Cosmology/'

    df = GAMA_catalog_w_photometry(datadir_spec=loc_on_emilles_comp,
                                   datadir_phot=loc_on_emilles_comp,
                                   savefile=opts.savefile)

    if opts.create_redshift:
        create_redshift_data(df, z_SLICS, nslice=opts.nslice)

    if opts.run_environment:
        z_bins = redshift_bins(z_SLICS, nslice=opts.nslice)

        RA_bin_ends = [0., 80., 160., 200., 360.]
        subsamples, lens = [], []
        field_bounds = []
        for i in range(len(RA_bin_ends)-1):
            one_field, one_len = [], []
            part_subsample = df.loc[(df['RA'] >= RA_bin_ends[i]) & (df['RA'] < RA_bin_ends[i+1])]
            (minx, maxx) = (np.floor(part_subsample['RA'].min()), np.ceil(part_subsample['RA'].max()))
            (miny, maxy) = (np.floor(part_subsample['DEC'].min()), np.ceil(part_subsample['DEC'].max()))
            if not opts.nslice:
                for j in range(len(z_bins)-1):
                    subsample = df.loc[(df['RA'] >= RA_bin_ends[i]) & (df['RA'] < RA_bin_ends[i+1])
                                         & (df['NQ'] > 2) & (df['Z'] >= z_bins[j]) & (df['Z'] < z_bins[j+1]),
                                         ['CATAID', 'RA', 'DEC', 'Z', 'NQ']]
                    nn = len(subsample)
                    one_len.append(nn)
                    one_field.append(subsample)

            elif 'one_slice' in opts.nslice:
                for j in np.arange(0, len(z_bins), 2):
                    subsample = df.loc[(df['RA'] >= RA_bin_ends[i]) & (df['RA'] < RA_bin_ends[i+1])
                                       & (df['NQ'] > 2) & (df['Z'] >= z_bins[j]) & (df['Z'] < z_bins[j+1]),
                                       ['CATAID', 'RA', 'DEC', 'Z', 'NQ']]
                    nn = len(subsample)
                    one_len.append(nn)
                    one_field.append(subsample)

            subsamples.append(one_field)
            lens.append(one_len)
            field_bounds.append((minx, maxx, miny, maxy))

        all_envs = []
        for f in range(len(subsamples)):
            (minx, maxx, miny, maxy) = field_bounds[f]
            for s in range(len(subsamples[f])) :
                try_distances = distance_bins(z_SLICS[s],
                                              btype=opts.radial_binning,
                                              n=opts.n)
                assert(max(try_distances) <= min((maxx - minx), (maxy - miny)) / 2.)
                if opts.verbose:
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
                    if len(try_distances) < opts.n:
                        new_envs_in_field = []
                        for envs in envs_in_field:
                            envs += [''] * (opts.n - len(try_distances))
                            new_envs_in_field.append(envs)
                        envs_in_field = new_envs_in_field
                    all_envs = all_envs + envs_in_field
                    pool.close()

        envs_arr = np.array(all_envs)
        envs_df = pd.DataFrame(data=envs_arr, index = envs_arr[:, 0],
                               columns = ['CATAID']+[str(i) for i in np.arange(1, opts.n+1)])

        envs_df['CATAID']=envs_df['CATAID'].astype(int)
        df['CATAID']=df['CATAID'].astype(int)

        zenvdf = pd.merge(envs_df, df, on='CATAID')

        if opts.savefile:
            path = os.path.join(opts.outdir, 'enviros.csv')
            zenvdf.to_csv(path)
    else:
        try:
            path_enviros = os.path.join(opts.outdir, 'enviros.csv')
            zenvdf = pd.read_csv(path_enviros)
        except FileNotFoundError:
            print('No enviros.csv file. Must run with --run_env flag to generate.')

    for z_string in ['0.042', '0.080', '0.130', '0.221', '0.317', '0.418', '0.525']:
        if opts.generate_fit_summaries:

            dfr = redshift_df(z_string, zenvdf)
            if float(z_string) < 0.3:
                dfr = dfr.loc[(dfr['lsstr'] > 0) & (dfr['lssti'] > 0) & (dfr['lsstz'] > 0)]
                dfr['mag3'] = dfr['lsstr']
            else:
                dfr = dfr.loc[(df['lssti'] > 0) & (dfr['lsstz'] > 0) & (dfr['lssty'] > 0)]
                dfr['mag3'] = dfr['lssty']

            class_path = os.path.join(opts.modeldir, 'classes/')
            df_classified = pd.read_csv(class_path + 'z_%s_manygroups.csv' % z_string)

            df_w_label = pd.merge(dfr, df_classified, on='CATAID')


            for l in range(0, int(opts.gnum)):
                create_fit_summaries(df_w_label, l, z_string,
                                     patch=opts.patch, gnum=opts.gnum,
                                     chains=20, outdir=opts.outdir)

        if opts.run_particle_environment:
            try_distances = distance_bins(float(z_string),
                                          btype=opts.radial_binning,
                                          n=opts.n)

            try:
                data = pd.read_csv(opts.particle_file, header=None, delimiter=' ').values
            except FileNotFoundError:
                print('Mock RA/Dec file %s not found' % opts.particle_file)

            (minx, maxx) = (np.floor(data[:, 0].min()), np.ceil(data[:, 0].max()))
            (miny, maxy) = (np.floor(data[:, 1].min()), np.ceil(data[:, 1].max()))

            nps = mp.cpu_count()
            pool = mp.Pool(nps - 1)
            envs_in_field = pool.map(calc_env, range(len(data)))
            pool.close()

            # List of environments in the particle data
            envs_in_field = np.array(envs_in_field)

            envs_in_field = np.concatenate((envs_in_field, data), axis=1)

            envs_df = pd.DataFrame(data=envs_in_field,
                                   index=envs_in_field[:, 0],
                                   columns=['CATAID']+[str(i) for i in np.arange(1, opts.n+1)]+['RA', 'Dec']
                                   )

            if opts.savefile:
                path = os.path.join(opts.outdir, 'particle_enviros_%s.csv' % z_string)
                envs_df.to_csv(path, index=False)

        else:
            try:
                path_partenviros = os.path.join(opts.outdir,
                                            'particle_enviros_%s.csv' % z_string)
                envs_df = pd.read_csv(path_partenviros)
            except FileNotFoundError:
                print('No particle_enviros_%s.csv file.' % z_string, \
                      'Must run with --run_particle flag to generate.')

        # The results are a mock catalog
        results = get_properties(envs_df[:].values[:,1:-2], z_string,
                                 patch=opts.patch,
                                 gnum=opts.gnum,
                                 btype=opts.radial_binning,
                                 ra=envs_df['RA'],
                                 dec=envs_df['Dec'],
                                 modeldir=opts.modeldir,
                                 indir=opts.outdir,
                                 savefiles=opts.savefile,
                                 outdir=opts.outdir)
