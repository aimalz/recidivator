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
from astropy.cosmology import FlatLambdaCDM
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


def redshift_bins(z, z_low=0.023, z_high=3.066):
    """
        Input: Array of lists of redshifts matching N-body data
        Output: Redshift bins for real data

        z_low: lower limit of lowest redshift bin. Default calculated for SLICS
        z_high: higher limit of highes redshift bin. Default calculated for SLICS
    """

    z_mids = (z[2:] + z[1:-1]) / 2.
    z_bins = np.insert(z_mids, 0, 0.05949)
    z_bins = np.insert(z_bins, 0, z_low)
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

def distance_bins(z, n=10, noz=False, verbose=False, **kwargs): ### NOTE: I REWROTE THIS
    """
        Returns list of angular distances.

        Input:
        z : array of redshift bins
        n : number of distances
        noz : uses the redshift invariant distances
        verbose : prints the list of distances returned.
        kwargs: for the no redshift (noz) option, dmin and dmax specifies the
                upper and lower limit of the distances.

        Output:
        list of angular distances
    """

    if noz:
        dmin= kwargs.pop('dmin', 0.2)
        dmax = kwargs.pop('dmax', 2.0)
        try_distances = np.flip(np.geomspace(dmin, dmax, n), axis=0)

    else:
        phys_anchors = [1., 10., 100.]
        cosmo = FlatLambdaCDM(H0=69.98, Om0=0.2905, Ob0=0.0473)
        dc = cosmo.comoving_distance(z)
        da = dc / (1. + z)
        ang_anchor = phys_anchors / da * 180. / np.pi

        try_distances = np.flip(np.geomspace(min(ang_anchor.value),
                                             min(2.0, max(ang_anchor.value)),
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
def run_clustering(str_z, zenvdf, n_clusters=10, metric="euclidean",
                   max_iter=5, random_state=0, savefiles=True, outdir='./'):
    """
        info - may change waiting
    """
    try_distances = distance_bins(float(str_z))
    str_tryd = [str(i) for i in np.arange(1, 11)]

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
        df['color1'] = df['lsstr'] - df['lssti']
        df['color2'] = df['lssti'] - df['lsstz']
    else:
        df['color1'] = df['lssti'] - df['lsstz']
        df['color2'] = df['lsstz'] - df['lssty']

    if savefiles:
        os.makedirs('ts_kmeans', exist_ok=True)
        filename = 'ts_kmeans/tskmeans_%s.pkl' % str_z
        pickle.dump(km, open(filename, 'wb'))

        path = os.path.join(outdir, 'Redshift_df_label_%s.csv' % str_z)
        df.to_csv(path)

    return df

### Make populations
def create_fit_summaries(df, lbl, str_z, iter=1000, chains=4, warmup=500,
                         savefiles=True, outdir='./'):
    """
        Takes in a pandas data frame with 2 colors and a magnitude and returns
        the means and covariance matrix of the corresponding 3D Gaussian distribution.

        Input
        df :
        lbl : label
        str_z :

        iter :
        chains :
        warmup :
        savefiles :
        outdir :

        Output
        data frame containing means and covariance matrix

        TBD: generalize per redshift bin
    """
    df_l = df.loc[df['label']==lbl]

    N = len(df_l)

    vals = df_l[['color1', 'color2', 'lssti']].values

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
        path_files = os.path.join(outdir,
                                  'fit_summaries/summary_%s_label_%s.csv'
                                  % (str_z, lbl))
        os.makedirs(path_dir, exist_ok=True)
        summary.to_csv(path_files)

    return summary


## Given environment curves for particle data return galaxy properties
def get_label(n_r, str_redshift, verbose=False):
    """
        n_r: array of number of neighbors at same radii as TS Kmeans was trained
        redshift: redshift bin to get model

        returns: label in an array
        """

    filename = 'ts_kmeans/tskmeans_%s.pkl'% str_redshift
    km = pickle.load(open(filename, 'rb'))
    if len(n_r.shape) == 1:
        predicted = km.predict(n_r.reshape(1, -1))
    else:
        predicted = km.predict(n_r)

    if verbose:
        print("Predicted label is: ", predicted)

    return predicted

def get_random_sample(label, str_redshift):
    from scipy.stats import multivariate_normal
    rando = []
    for l in label:
        summary = pd.read_csv('fit_summaries/summary_%s_label_%s.csv' %(str_redshift, label[0]))
        mus = summary.iloc[:3]['mean'].values
        cov = summary.iloc[3:]['mean'].values.reshape(3, 3)

        rando.append(multivariate_normal.rvs(mus, cov))
    return rando

def get_properties(n_r, str_redshift, verbose=False):
    l = get_label(n_r, str_redshift, verbose=verbose)
    samp = get_random_sample(l, str_redshift)
    return samp

### Plotting routines
def make_orchestra(z, zenvdf, savefig=True, verbose=False):
    # Still some color map issues and need to generalize for redshifts.
    try_distances = distance_bins(z[0])
    orig_distances = np.flip(try_distances, axis=0)

    color = color_map(z, vmin=z[0], vmax=z[-1])
    cNorm  = colors.Normalize(vmin=z[0], vmax=z[-1])

    fig, ax = plt.subplots(figsize=(15,10))
    for n, z in enumerate(z):
        df = redshift_df(str(z), zenvdf)
        if len(df) > 0:
            for i in range(len(orig_distances)):
                parts = ax.violinplot(df[str(orig_distances[i])], positions=[i])
                np.where(df[str(orig_distances[i])] < 1)
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

    plt.xticks(range(len(orig_distances)), np.around(orig_distances, 3))
    ax.semilogy()
    ax.set_xlabel('radial distance [deg]', size=15)
    ax.set_ylabel('Normalized number of neighbors', size=15)

    cax, _ = matplotlib.colorbar.make_axes(ax, pad=0.01)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=plt.cm.Spectral_r, norm=cNorm)
    cbar.ax.set_ylabel('redshift', size=12)

    plt.tight_layout()

    if savefig:
        plt.savefig('orchestra_neighbor_v_distance.pdf')
    return


def make_ang_phys_plot(savefig=True):
    fig, ax = plt.subplots()
    ang_dist_grid = []
    for i in range(len(z_SLICS[0:8])):
        try_distances = distance_bins(z, n=10)

        dc = cosmo.comoving_distance(z_SLICS[i])
        da = dc / (1 + z_SLICS[i])
        ax.plot(np.flip(try_distances, axis=0) * da * np.pi / 180.,
                np.flip(try_distances, axis=0), color=color[i], lw=3)

    ax.semilogx()

    # need to update this once we figure out which radii to use
    ax.axvline(1, c='grey', alpha=0.4)
    ax.axvline(2, c='grey', alpha=0.4)
    ax.axvline(3, c='grey', alpha=0.4)
    ax.axvline(5, c='grey', alpha=0.4)
    ax.axvline(8, c='grey', alpha=0.4)
    ax.axvline(20, c='grey', alpha=0.4)
    ax.axvline(40, c='grey', alpha=0.4)
    ax.axvline(60, c='grey', alpha=0.4)

    ax.set_ylabel('Angular distances', fontsize=13)
    ax.set_xlabel('Physical scales [Mpc]', fontsize=13)

    plt.setp(ax, xticks=([1, 5, 10, 50, 70]), xticklabels=([1, 5, 10, 50, 70]))

    cax, _ = matplotlib.colorbar.make_axes(ax, pad=0.01)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=jet, norm=cNorm)
    cbar.ax.set_ylabel('redshift', size=12)
    if savefig:
        plt.savefig('physical_scale_v_angular_distance.pdf')
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
    parser.add_argument('--run_env', dest='run_environment',
                        default=False, action='store_true')
    parser.add_argument('--gen_summaries', dest='generate_fit_summaries',
                        default=False, action='store_true')
    parser.add_argument('--run_particle', dest='run_particle_environment',
                        default=False, action='store_true')
    opts = parser.parse_args()

    loc_on_emilles_comp = '/media/CRP6/Cosmology/'

    df = GAMA_catalog_w_photometry(datadir_spec=loc_on_emilles_comp,
                                   datadir_phot=loc_on_emilles_comp)

    create_redshift_data(df, z_SLICS)

    if opts.run_environment:
        n = 10
        z_bins = redshift_bins(z_SLICS)

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
                one_len.append(nn)
                one_field.append(subsample)
            subsamples.append(one_field)
            lens.append(one_len)
            field_bounds.append((minx, maxx, miny, maxy))

        all_envs = []
        for f in range(len(subsamples)):
            (minx, maxx, miny, maxy) = field_bounds[f]
            for s in range(len(subsamples[f])) :
                try_distances = distance_bins(z_SLICS[s], n=n)
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
                    if len(try_distances) < n:
                        new_envs_in_field = []
                        for envs in envs_in_field:
                            envs += [''] * (n - len(try_distances))
                            new_envs_in_field.append(envs)
                        envs_in_field = new_envs_in_field
                    all_envs = all_envs + envs_in_field
                    pool.close()

        envs_arr = np.array(all_envs)
        envs_df = pd.DataFrame(data=envs_arr, index = envs_arr[:, 0], columns = ['CATAID']+[str(i) for i in np.arange(1, n+1)])

        zenvdf = pd.merge(envs_df, df, on='CATAID')

        if opts.savefile:
            path = os.path.join(opts.outdir, 'enviros.csv')
            zenvdf.to_csv(path)
    else:
        try:
            zenvdf = pd.read_csv('enviros.csv') # note: need specify location
        except FileNotFoundError:
            print('No enviros.csv file. Must run with --run_env flag to generate.')

    if opts.run_particle_environment:
        particles = pd.read_csv('ang2deg.csv')

    for z_string in ['0.042', '0.080', '0.130', '0.221', '0.317', '0.418', '0.525', '0.640']:
        if opts.generate_fit_summaries:
            df_w_label = run_clustering(z_string, zenvdf)

            # create the fit summaries for every label
            n_clusters = 10
            label = np.arange(0, n_clusters, 1)

            for l in label:
                create_fit_summaries(df_w_label, l, z_string)

        if opts.run_particle_environment:
            try_distances = distance_bins(float(z_string))

            nrand = 3000 ## will be updated to function by Malz
            rand_indicies = np.random.uniform(low=0, high=len(particles), size=nrand)
            data = particles.iloc[rand_indicies].values

            (minx, maxx) = (np.floor(data[:, 0].min()), np.ceil(data[:, 0].max()))
            (miny, maxy) = (np.floor(data[:, 1].min()), np.ceil(data[:, 1].max()))

            nps = mp.cpu_count()
            pool = mp.Pool(nps - 1)
            envs_in_field = pool.map(calc_env, range(len(data)))
            pool.close()

            # List of environments in the particle data
            envs_in_field = np.array(envs_in_field)
            #envs_in_field = np.delete(envs_in_field, 0, axis=1)

            envs_df = pd.DataFrame(data=envs_in_field,
                                   index = envs_in_field[:, 0],
                                   columns = ['CATAID']+[str(i) for i in try_distances])

            if opts.savefile:
                path = os.path.join(opts.outdir, 'particle_enviros_%s.csv' % z_string)
                envs_df.to_csv(path, index=False)

        else:
            try:
                envs_df = pd.read_csv('particle_enviros_%s.csv' % z_string)
            except FileNotFoundError:
                print('No particle_enviros_%s.csv file.' % z_string, \
                      'Must run with --run_particle flag to generate.')

        # The results are a mock catalog
        results = get_properties(envs_df[:].values[:,1:], z_string)
