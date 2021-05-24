#!/usr/bin/env python

import numpy as np
import pandas as pd

import os
import glob
import pickle
import corner

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib

import bisect
import sklearn

from astropy import units
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction.projection import FPCA
from sklearn.cluster import KMeans

import pystan

def clean_env_curves(fname_env: str, ftype: str):
    """Remove problematic elements from synthetic environmental curves.
    
    Parameters
    ----------
    fname_env: str
        Path to environmental curves.
    ftype: str
        Real GAMA data or synthetic data
        Options -- "real" or "synth"
        
    Returns
    -------
    data_clean: np.array
        Clean set of environmental curves.
    redshift: np.array
        Corresponding redshifts.
    """
    # read environmental curves
    if len(fname_env) > 1:
        li = []
        for filename in fname_env:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)
        data_env = pd.concat(li, axis=0, ignore_index=True)
    else:
        if 'pkl' in fname_env:
            data_env = pd.read_pickle(fname_env)
        else:
            data_env = pd.read_csv(fname_env)


    if 'real' in ftype:
        col_keys = list(data_env.columns[36:])
    elif 'synth' in ftype:
        col_keys = list(data_env.columns[4:])
    else:
        raise NameError('ftype: Options are "real" for real GAMA data or "synth" for synthetic GAMA data')
    
    c = 0
    indx = []

    # remove problematic elements
    # find the nans
    for i in range(data_env.shape[0]):
        if np.any(np.isnan(list(data_env.iloc[i][col_keys]))):
            indx.append(i)

    data_env.drop(indx)
    
    return data_env

def calc_dist_points(ndist: int, max_dist=2.5):
    """Calculates distance evaluation points.
    
    Parameters
    ----------
    ndist: int
        Number of points in distance grid.
    max_dist: float (optional)
        Maximum distance. Default is 2.5.
        
    Returns
    -------
    distance_evaluation_points: np.array
        Values of distances in grid.
    """

    maxang = units.Quantity(max_dist, 'deg')
    distance_evaluation_points = np.linspace(0., max_dist, ndist+2)[1:-1]
    pos = bisect.bisect(distance_evaluation_points, maxang.value)
    distance_evaluation_points = distance_evaluation_points[:pos]
    
    return distance_evaluation_points


def do_functional_PCA(data: np.array, ndist: int,
                      npcs: int, ftype: str,
                      max_dist=2.5, norm=False):
    """Calculate functional principal components.
    
    Parameters
    ----------
    data: np.array
        Environmental curves.
    ndist: int
        Number of points in distance grid.
    npcs: int
        Number of principal functions to be calculated.
    max_dist: float (optional)
        Maximum distance to which calculate distance bins.
        Default is 2.5.
    norm: bool (optional)
        If True, normalize functions by its own maximum and add
        collumn with normalization value. Default is False.
        
    Returns
    -------
    res: dict
        Dictionary of results. Keys are:
        proj: projections
        comp_orig: original functional PCs
        comp_inv: inverted PC matrix (for reconstruction)
        mean: mean function
    """
    # Convert input
    if 'real' in ftype:
        col_keys = list(data_env.columns[36:])
    elif 'synth' in ftype:
        col_keys = list(data_env.columns[4:])
    else:
        raise NameError('ftype: Options are "real" for real GAMA data or "synth" for synthetic GAMA data')
    
    data = data[col_keys].to_numpy()

    # dictionary to store results
    res = {}
    
    # prepare the data
    grid_points = calc_dist_points(ndist=ndist, max_dist=max_dist)

    if norm:
        data_use = np.array([[line[i]/max(line) for i in range(data.shape[1])]
                            for line in data])
        max_env = [np.max(line) for line in data]
    else:
        data_use = data

    
    data_fda = FDataGrid(data_use, grid_points)

    # calculate principal functions
    n = data_fda.data_matrix.shape[1] # in order to be invertable number PCs
                                      # should be equal to the number of
                                      #parameters
    fpca = FPCA(n_components=n)

    # projections
    res['proj'] = fpca.fit_transform(data_fda)[:,:npcs]
    
    # original functional pcs
    comp_orig = fpca.components_.data_matrix.reshape(n, n)
    res['comp_orig'] = comp_orig[:npcs]
    
    # inverted functional PCs for reconstruction
    res['comp_inv'] = np.linalg.inv(comp_orig).transpose()[:npcs]

    # mean function
    res['mean'] = fpca.mean_.data_matrix.flatten()

    if norm:
        # maximum value of environmental curves
        res['max_env'] = max_env

    return res


def build_feature_matrix(redshift: np.array, proj: np.array,
                         max_env=None):
    """Build normalized feature matrix (incl. redshift).
    
    Paramters
    ---------
    redshift: np.array
        Array of redshifts.
    proj: np.array
        Projections. Dimensions is n_objs x n_pcs.
    max_env: np.array (bool)
        Array with maximum values for environmental curves.
        Should only be used if curves were normalized before fPCA.
        Default is None.
        
    Returns
    -------
    cent_matrix: np.array
        Standardize feature matrix. First column is redshift,
        others correspond to projections in different functional PCs.
    """
    
    # add redshift to projections and output to file
    proj_z = np.array([np.insert(proj[i], 0, redshift[i], axis=0)
                       for i in range(len(redshift))])
    
    if max_env != None:
        proj_z0 = np.array([np.insert(proj[i], 0, redshift[i], axis=0)
                           for i in range(len(redshift))])
        proj_z = np.array([np.insert(proj_z0[i], 0, max_env[i], axis=0)
                           for i in range(len(max_env))])

    mean_proj_z = np.array([np.mean(proj_z[:,i])
                            for i in range(proj_z.shape[1])])
    std_proj_z = np.array([np.std(proj_z[:,i])
                           for i in range(proj_z.shape[1])])

    cent_matrix = np.array([[(proj_z[j][i] - mean_proj_z[i])/std_proj_z[i]
                             for i in range(proj_z.shape[1])]
                            for j in range(proj_z.shape[0])])

    return cent_matrix


def build_cluster_models(matrix: np.array, ngroups: list, screen=True,
                         save=False, output_dir=None):
    """Perform cluster and save models and classifications.
    
    Parameters
    ----------
    matrix: np.array
       Feature matrix.
    ngroups: list
       Number of groups for which to build clusters.
    output_dir: str (optional)
        Output directory to save models and classificaitons.
        Only used if save == True. Default is None.
    save: bool (optional)
        If True, save models and classification to file.
        Default is False.
    screen: bool (optional)
        If True print evolution steps. Default is True.
    
    Returns
    -------
    classes: pd.DataFrame
        Estimated classes. Keys are the number of groups.
    models: pd.DataFrame
        Trained models. Keys are the number of groups.
    """
    
    # place holder to store predicted classes and models
    classes = {}
    models = {}

    for i in ngroups:
        
        # fit k-means
        km = KMeans(n_clusters=i, random_state=42, n_jobs=20).fit(matrix)
        models[i] = km
        
        # estimate which cluster each curve belongs to
        classes[i] = km.labels_
        
        if save:
            pickle.dump(models[i], open(output_dir + '/model_' + str(i).zfill(2) + '_groups.pkl', 'wb'))
            np.savetxt(output_dir + '/classes_' + str(i).zfill(2) + '_classes.csv', classes[i],
                     delimiter=',')
        if screen:
            print('Finished ' + str(i) + ' groups!')
    
    return pd.DataFrame.from_dict(classes), pd.DataFrame(models, index=[0])

def assign_groups(data: np.array, comp_orig: np.array, redshift: np.array,
                  model: sklearn.cluster._kmeans.KMeans, npcs: int,
                  norm=False):
    """
    Project synthetic data into principal functions.
    
    Parameters
    ----------
    data: np.array
        Environmental curves: [n_objs, n_distances].
    comp_orig: np.array
        Principal functions.
    model: sklearn.cluster._kmeans.KMeans
        Trained KMeans clustering model.
    npcs: int
        Number of principal functions to be calculated
    redshift: np.array
        Redshifts for each object in data.
    norm: bool (optional)
        If True, normalize environmental curves and add column
        with max value. Default is False.
        
    Returns
    -------
    labels: np.array
        Groups assigned to each environmental curve in data.
    """
    if norm:
        data_temp = data.div(data.max(axis=0), axis=0)
        max_env = data.max(axis=0)
    else:
        data_temp = data

    # calculate projections
    proj = np.array([[np.dot(data_temp.iloc[j], comp_orig[i]) for i in range(npcs)]
                      for j in range(data_temp.shape[0])])

    # add redshift (and max env) to the projections
    if norm:
        proj_z = np.array([[max_proj[i], redshift[i]] + list(proj[i])
                           for i in range(proj.shape[0])])
    else:
        proj_z = np.array([[redshift[i]] + list(proj[i]) for i in range(proj.shape[0])])

    mean_proj_z = np.array([np.mean(proj_z[:,i])
                            for i in range(proj_z.shape[1])])
    std_proj_z = np.array([np.std(proj_z[:,i])
                           for i in range(proj_z.shape[1])])

    cent_matrix = np.array([[(proj_z[j][i] - mean_proj_z[i])/std_proj_z[i]
                             for i in range(proj_z.shape[1])]
                            for j in range(proj_z.shape[0])])

    # assign groups
    labels = model.predict(cent_matrix)

    return labels

def create_fit_summaries(df, lbl, str_z, patch, gnum,
                         iter=1000, chains=4, warmup=500,
                         savefiles=True, outdir='./'):
    """
        Takes in a pandas data frame with 3 magnitudes and
        returns the means and covariance matrix
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
    """
    label_name = 'label'

    df_l = df.loc[df[label_name]==int(lbl)]
    
    df_low = df_l.loc[df_l['Z'] < 0.3]
    df_low = df_low.loc[(df_low['lsstr'] > 0) & (df_low['lssti'] > 0) & (df_low['lsstz'] > 0)]
    
    df_high = df_l.loc[df_l['Z'] > 0.3]
    df_high = df_high.loc[(df_high['lssty'] > 0) & (df_high['lssti'] > 0) & (df_high['lsstz'] > 0)]

    vals_low = df_low[['lsstr', 'lssti', 'lsstz']].values
    vals_high = df_high[['lssti', 'lsstz', 'lssty']].values

    N_re = 1

    N_low = len(df_low)
    N_high = len(df_high)
    
    ## You only need this ONE model to fix all of the data no matter what label you have!
    try:
        sm = pickle.load(open('3D_Gaussian_model.pkl', 'rb'))
    except:
        sm = pystan.StanModel(file='some_dimensional_gaussian.stan')
        # Save the model so that you don't have to keep regenerating it.
        with open('3D_Gaussian_model.pkl', 'wb') as f:
            pickle.dump(sm, f)
    
    do_low = True
    do_high = True
    if N_low < 12 and N_high < 12:
        return "There are not enough objects to run Stan. N_low = %s, N_high = %s"%(N_low, N_high)
    elif N_low < 12:
        do_low = False
    elif N_high < 12:
        do_high = False

    if do_low:
        dat_low = {
            'N': N_low,
            'D': 3,
            'N_resamples': N_re,
            'tavy': vals_low,
        }

        fit_low = sm.sampling(data=dat_low, iter=iter, chains=chains, warmup=warmup)
        
        summary_low = pd.DataFrame(fit_low.summary(pars=['mu', 'Sigma'])['summary'],
                       columns=fit_low.summary(pars=['mu', 'Sigma'])['summary_colnames'])

        summary_low['Names'] = fit_low.summary(pars=['mu', 'Sigma'])['summary_rownames']

        if savefiles:
            path_dir = os.path.join(outdir, 'fit_summaries')
            os.makedirs(path_dir, exist_ok=True)

            path_files_low = os.path.join(path_dir,
                                      'summary_low_label_%s_patch_%s_groups_%s.csv'
                                      % (lbl, patch, gnum))
            summary_low.to_csv(path_files_low)

    if do_high:
        dat_high = {
            'N': N_high,
            'D': 3,
            'N_resamples': N_re,
            'tavy': vals_high,
        }

        fit_high = sm.sampling(data=dat_high, iter=iter, chains=chains, warmup=warmup)

        summary_high = pd.DataFrame(fit_high.summary(pars=['mu', 'Sigma'])['summary'],
                               columns=fit_high.summary(pars=['mu', 'Sigma'])['summary_colnames'])

        summary_high['Names'] = fit_high.summary(pars=['mu', 'Sigma'])['summary_rownames']

        if savefiles:
            path_dir = os.path.join(outdir, 'fit_summaries')
            os.makedirs(path_dir, exist_ok=True)

            path_files_high = os.path.join(path_dir,
                              'summary_high_label_%s_patch_%s_groups_%s.csv'
                              % (lbl, patch, gnum))
            summary_high.to_csv(path_files_high)

    if do_low and not do_high:
        return summary_low, []
    if do_high and not do_low:
        return [], summary_high
    if do_high and do_low:
        return summary_low, summary_high

def get_random_sample(label, redshift_bin, patch, group_num, indir='./'):
    """
        label: Cluster label
        redshift_bin: "low" or "high" depending on z<0.3 or z> 0.3
        patch: patch number
        group_num: number of clusters
        indir: Where to find the fit summaries

        returns: random draws in an array
    """

    from scipy.stats import multivariate_normal

    rando = []
    notmissing = []
    info = []
    for i, (l, z) in enumerate(zip(label, redshift_bin)):
        try:
            path = os.path.join(indir,
                                'fit_summaries/summary_%s_label_%s_patch_%s_groups_%s.csv'
                                % (z, label[0], patch, group_num))
            summary = pd.read_csv(path)
            mus = summary.iloc[:3]['mean'].values
            cov = summary.iloc[3:]['mean'].values.reshape(3, 3)
            madd = multivariate_normal.rvs(mus, cov)

            if 'low' in z :
                madd = np.append(madd, -999.0)
            if 'high' in z:
                madd = np.insert(madd, 0, -999.0)


            rando.append(madd)
            notmissing.append(i)

        except:
            print('files does not exist: ', z, label[0], patch, group_num)
            info.append([z, label[0], patch, group_num])
            continue
    return rando, notmissing

def make_catalog_per_fname(f1, comp_orig, model, npcs):
    data = pd.read_csv(f1)
    header = list(data.keys())[4:]
    
    str_redshift = []

    for z in data['Z'].values:
        if z < 0.3:
            str_redshift.append('low')
        else:
            str_redshift.append('high')

    l = assign_groups(data=pd.DataFrame(data, columns=header), comp_orig=comp_orig, redshift=data['Z'].values,
                       model=model, npcs=npcs)
    samp, nm = get_random_sample(l, str_redshift, patch=patch, group_num=ngroups,
                                indir='./')

    ra = data['RA'].values[nm]
    dec = data['DEC'].values[nm]
    z = data['Z'].values[nm]
    l = l[nm]

    radec = np.vstack((ra, dec, z, l)).T
    samp1 = np.concatenate((radec, samp), axis=1)
    samp_pd = pd.DataFrame(samp1, columns=['ra', 'dec', 'z', 'label', 'lsstr', 'lssti', 'lsstz', 'lssty'])
    
    samp_pd.rename(columns={'lsstr': 'mag_r_lsst',
                            'lssti': 'mag_i_lsst',
                            'lsstz': 'mag_z_lsst',
                            'lssty': 'mag_y_lsst'}, inplace=True)
    return samp_pd


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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--no_files', dest='savefile',
                        default=True, action='store_false')
    parser.add_argument('--outdir', default='./')
    parser.add_argument('--modeldir',
                        default='/media/CRP6/Cosmology/workdir/emille/fda/results_norm/',
                        help='Location of ML model for GAMA environment curve clutersting')
    parser.add_argument('--chunkdir',
                        default='/media/CRP6/Cosmology/recidivator/GAMA/flexible_envirocurves/',
                        help='Location of GAMA and Simulation Chunks')
    parser.add_argument('--gen_summaries', dest='generate_fit_summaries',
                        default=False, action='store_true',
                        help='Generate 3D Gaussian Stan fits to GAMA curves')
    parser.add_argument('--do_class', dest='gdo_classification',
                        default=False, action='store_true',
                        help='Rerun the Principle Components + Kmeans Classification')
    parser.add_argument('--patch', dest='patch',
                        default='G15', dtype=str)
    parser.add_argument('--groups', dest='gnum',
                        default=2)
    parser.add_argument('--ndist', dest='ndist',
                        default=100)
    parser.add_argument('--npcs', dest='npcs',
                        default=15)
    parser.add_argument('--max_dist', dest='max_dist',
                        default=2.5)
    parset.add_argument('--nonorm', dest='norm',
                        default=True, action='store_false')
    parser.add_argument('--los', dest='LOS',
                        default='42')
    parser.add_argument('--iter', dest='iter',
                        default=2000, help='# of iterations for pySTAN')
    opts = parser.parse_args()

    if opts.do_classification:
        env_path = os.path.join(opts.chunkdir,
                                opts.patch + 'chunk*dists' + str(opts.ndist) + '.csv')
        fnames_env = glob.glob(env_path)
        data_env = clean_env_curves(fname_env=fnames_env, ftype='real')

        redshift = list(data_env['Z'].values)

        fPCA = do_functional_PCA(data=data_env, ndist=opts.ndist, npcs=opts.npcs,
                                 max_dist=opts.max_dist, norm=opts.norm, ftype='real')
        matrix = build_feature_matrix(redshift=redshift, proj=fPCA['proj'])

        classes, models = build_cluster_models(matrix=matrix, ngroups=opts.gnum,
                                               output_dir=os.path.join(opts.modeldir, str(opts.npcs) + 'PCS/'),
                                               save=opts.savefile)


    if opts.generate_fit_summaries:
        for l in range(ngroups):
            create_fit_summaries(data_env, lbl=l, str_z='all', patch=opts.patch, gnum=opts.gnum,
                                 iter=opts.iter, chains=4, warmup=int(opts.iter/4),
                                 savefiles=opts.savefile, outdir=opts.outdir)


    model_path = os.path.join(opts.modeldir,
                              str(opts.npcs) + 'PCs/' + opts.patch + '/model_' + str(opts.gum).zfill(2) + '_groups.pkl')
    model = pd.read_pickle(model_path)

    los_chunks_path = os.path.join(opts.chunkdir,
                                  'LOS' + str(opts.LOS) + 'chunk*dists' + str(opts.ndist) + '.csv'
    los_chunks = glob.glob(los_chunks_path)
    los_chunks.sort()

    comp_orig_path = os.path.join(opts.modeldir,
                                  str(npcs) +'PCs/' + opts.patch + '/comp_orig_' + str(opts.npcs) + 'fPC.csv'
                                  
    comp_orig = pd.read_csv(comp_orig_path, header=None).values

    for i, f1 in enumerate(los_chunks):
        print(f1)
        if i == 0:
            samp = make_catalog_per_fname(f1, comp_orig, model, opts.npcs)
        else:
            samp1 = make_catalog_per_fname(f1, comp_orig, model, opts.npcs)
            samp = pd.concat([samp, samp1])

    if opts.savefile:
        save_path = os.path.join(opts.outdir,
                                 'results_LOS42_' + opts.patch + '_' + str(opts.gum).zfill(2) + 'groups_' + str(npcs) +'PCs.csv')
        samp.to_csv(save_path, index=False)

