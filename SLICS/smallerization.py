
# coding: utf-8

# # Exploring the __SLICS-HR__ particle data
# notebook by _Alex Malz (GCCL@RUB)_, (add your name here)

# In[ ]:


import astropy as ap
from astropy.cosmology import FlatLambdaCDM
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd

basepath = 'particle_data'
datapath = os.path.join(basepath, 'cuillin.roe.ac.uk/~jharno/SLICS/SLICS_HR/LOS1/')

# Read in from binary float(4) format.

# In[ ]:


dt_each = 'f' + str(4)
dt = np.dtype([('x', dt_each), ('y', dt_each), ('z', dt_each), ('vx', dt_each), ('vy', dt_each), ('vz', dt_each)])

# ## Convert to physical units
#
# The particle data starts out in simulation units relative to the per-node subvolume and needs to be converted to physical units in the space of all subvolumes before the whole volume can be considered.

# In[ ]:


# number of MPI tasks per dimension
nodes_dim = 4

# volume size
rnc = 3072.

# subvolume size
ncc = rnc / nodes_dim

# physical scale in Mpc/h
phys_scale = 505.


# ## Read in data
#
# Download one of the 64 nodes $\times$ 20 redshifts files at each redshift from Joachim Harnois-Deraps to start.
# I chose file 21 at $z=0.042$ for this example.

# In[ ]:

# We're only considering the first 8 SLICS snapshots because the GAMA data doesn't have good enough coverage beyond that.

# In[ ]:


z_SLICS = np.array([0.042, 0.080, 0.130, 0.221, 0.317, 0.418, 0.525, 0.640])
#, 0.764, 0.897, 1.041, 1.199, 1.372, 1.562, 1.772, 2.007, 2.269, 2.565, 2.899])
z_mids = (z_SLICS[1:] + z_SLICS[:-1]) / 2.
z_bins = np.insert(z_mids, 0, 0.023)
z_bins = np.append(z_bins, 3.066)

# ## How much data do we need?
#
# Obtain necessary depth from ~~[Ned Wright's cosmology calculator](http://www.astro.ucla.edu/~wright/CosmoCalc.html)~~ `astopy`.
# The SLICS cosmology has $\Omega_{m} = 0.2905$, $\Omega_{\Lambda} = 0.7095$, $\Omega_{b} = 0.0473$, $h = 0.6898$, $\sigma_{8} = 0.826$, and $n_{s} = 0.969$.
# Let's assume the naive Cartesian-to-angular coordinates and flatten along the `z` direction.
# We need to flatten a depth corresponding to the bounds of each redshift bin.

# In[ ]:


h = 0.6898
cosmo = FlatLambdaCDM(H0=100.*h, Om0=0.2905, Ob0=0.0473)
d_comov = []
for z in z_bins:
    dc = cosmo.comoving_distance(float(z))
    d_comov.append(dc.value / h)
d_comov = np.array(d_comov)
depths = d_comov[1:] - d_comov[:-1]

avg_d_comov = []
for z in z_SLICS:
    dc = cosmo.comoving_distance(float(z))
    avg_d_comov.append(dc.value / h)

# Obtain angular diameter distance $d_{a}$ in units $\theta = x / d_{a}$ with $d_{a} = d_{c} / (1 + z)$, where $d_{c}$ is the comoving diameter distance and $x$ is the distance in the SLICS data.
# Compare with the GAMA footprint of $286^{\circ^{2}} * (\pi / 180^{\circ})^{2} \approx 0.087 sr$.

# In[ ]:


d_ang = avg_d_comov / (1 + z_SLICS)
theta_box = phys_scale / d_ang * 180. / np.pi
footprint = theta_box**2
# print(footprint)


# The scaling behavior is as expected;
# `phys_scale` subtends a larger angle at low redshifts and a smaller angle at high redshifts.
# One file's worth of SLICS data subtends an angular area larger than the GAMA footprint in the first five GAMA redshift bins, but the next three GAMA redshift bins would definitely require more than one file's worth of data.
# We need to pick an angular area for our mock galaxy catalog.
# Let's go with twice that for now.
# _Do we think twice the GAMA area is sufficiently compelling?_

# In[ ]:


theta_gama = 286.
GAMA_phys_scale = np.sqrt(theta_gama) * (np.pi / 180.) * d_ang
# print(2. * GAMA_phys_scale)

# We'd change this for the area of our mock survey when we decide on it.
# _There is an edge effect going on right now.
# I need to switch to one of the internal files to avoid roll-over that's breaking min/max checks._

# In[ ]:


lim_theta = np.sqrt(2. * theta_gama)

def proc_data(which_z):

    z_str = '{:<05}'.format(str(z_SLICS[which_z]))

    print('starting on ' + z_str)

    if not which_z in (0, 1, 2):
        print('not equipped for combining files yet!')
        return

    fn_base = 'xv'

# TODO: don't hardcode the file number, set it in response to above calculations of necessary size

    fn_index = 21
    fn_ext = '.dat'
    fn = z_str + fn_base + str(fn_index) + fn_ext


# In[ ]:


    with open(os.path.join(datapath, fn), 'rb') as f1:
        raw_data = np.fromfile(f1, dtype=dt)


# Throw out first 12 entries as unwanted header information.

# In[ ]:


    loc_data = pd.DataFrame(data=raw_data[2:], columns=['x', 'y', 'z', 'vx', 'vy', 'vz'])



# Note that the conversion below makes sense for `x`, `y`, and `z` but not for `vx`, `vy`, and `vz`.
# Because of how the data is distributed across the files, I think 21, 22, 25, 26, 37, 38, 41, 42 are "adjacent" and free of edge effects.

# In[ ]:


    all_nodes_coords = np.empty((nodes_dim, nodes_dim, nodes_dim))
    for k1 in range(1, nodes_dim+1):
        for j1 in range(1, nodes_dim+1):
            for i1 in range(1, nodes_dim+1):
                current_ind = (i1 - 1) + (j1 - 1) * nodes_dim + (k1 - 1) * nodes_dim ** 2
                node_coords = {'x': i1 - 1, 'y': j1 - 1, 'z': k1 - 1}
                if fn_index == current_ind:
                    # print('found index '+str(fn_index)+' at '+str((i1, j1, k1)))
                    true_node_coords = node_coords
                all_nodes_coords[node_coords['x'], node_coords['y'], node_coords['z']] = current_ind

# print(all_nodes_coords)


# To get coherent coordinates across all files, we need to shift them accordingly.
# The next cell is unexpectely slow.

# In[ ]:


# shift data
    glob_data = loc_data
    for col in ['x', 'y', 'z']:
        glob_data[col] = np.remainder(loc_data[col] + true_node_coords[col] * ncc, rnc)
        assert(max(glob_data[col] <= rnc))


# In[ ]:


# convert to Mpc/h
    phys_data = glob_data * phys_scale / rnc


# In[ ]:


# for dim in ['x', 'y', 'z']:
#     plt.hist(phys_data[dim], density=True, alpha=0.5)
# plt.xlabel('distance (Mpc/h)')


# print(depths)
# print(avg_d_comov)


# Sadly, `depths` < `phys_scale` $Mpc/h$ only in the first three redshift bins, meaning the depths of the next five GAMA redshift bin may require opening two files.
# I think the way they're arranged means that (21, 37), (22, 38), (25, 41), and (26, 42) are pairs adjacent in `z`.
#
# _This is as good a time as any to note that our mock catalog will have a bit of a degeneracy if we use the same file numbers for all redshifts because each file corresponds to the same physical volume across cosmic time, whereas in a real survey, our redshift bins contain different volumes/galaxies.
# We have a choice to make about discontinuities or non-physical repetitition._

# ## Convert to angular units
#



# If we go with twice the GAMA footprint, then the first three redshift bins need only one file, the next three need two, and the last two need three.
# I think (21, 22), (25, 26), (37, 38), and (41, 42) are adjacent in `x`/`RA` and (21, 25), (22, 26), (37, 41), and (38, 42) are adjacent in `y`/`DEC`.

# In[ ]:


# for i in range(4):
#     j = i+1
#     subset = phys_data[(phys_data['x'] <= 10.*j) & (phys_data['y'] <= 10.*j) & (phys_data['z'] <= 10.*j)]
#     subset.to_csv('spat'+str(j)+'0Mpc.csv', header=False, index=False, sep=' ', columns=['x', 'y', 'z'])
#     angular = subset / 313.5 * 69.6 / 100. * float(j) * 180 / np.pi
#     print((min(angular['x']), max(angular['x'])))
#     print((min(angular['y']), max(angular['y'])))
#     angular.to_csv('ang'+str(j)+'deg.csv', header=False, index=False, sep=',', columns=['x', 'y'])


# Right now, I'm not going to deal with combining adjacent files, just chopping up ones that are too big.
# This is slow!

# In[ ]:


    ang_data = phys_data[np.mod(phys_data['z'] - min(phys_data['z']), phys_scale) < depths[which_z]]

    # TODO: add a check here for the different cases of combining data files

    ang_data['RA'] = ang_data['x'] / d_ang[which_z] * 180. / np.pi
    ang_data['DEC'] = ang_data['y'] / d_ang[which_z] * 180. / np.pi


    cut_data = ang_data[(ang_data['RA'] < lim_theta + min(ang_data['RA']))
                    & (ang_data['DEC'] < lim_theta + min(ang_data['DEC']))]

# plt.hist(cut_data['RA'])
# plt.hist(cut_data['DEC'])
    print(str(len(cut_data))+' particles in mock footprint at '+z_str)

    cut_data.to_csv(os.path.join(basepath, z_str+'cut.csv'), header=True, index=False, sep=',', columns=['RA', 'DEC'])


# In[ ]:


# plt.hist2d(cut_data['RA'], cut_data['DEC'], bins=100, norm=mpl.colors.LogNorm(), cmap='Spectral_r')
# plt.xlabel('RA (deg)')
# plt.ylabel('DEC (deg)')
    return cut_data

# TODO need to deal with redshifts requiring multiple files next
which_snapshots = [0, 1, 2]

nps = mp.cpu_count()
pool = mp.Pool(nps - 1)
cutting = pool.map(proc_data, which_snapshots)
pool.close()
