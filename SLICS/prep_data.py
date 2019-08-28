import numpy as np
import pandas as pd
import treecorr

z_str = '0.042'
fn_base = 'xv'
fn_index = 0
fn_ext = '.dat'
fn = z_str + fn_base + str(fn_index) + fn_ext

dt_each = 'f' + str(4)
dt = np.dtype([('x', dt_each), ('y', dt_each), ('z', dt_each), ('vx', dt_each), ('vy', dt_each), ('vz', dt_each)])

with open(fn, 'rb') as f1:
    raw_data = np.fromfile(f1, dtype=dt)

loc_data = pd.DataFrame(data=raw_data[2:], columns=['x', 'y', 'z', 'vx', 'vy', 'vz'])

# number of MPI tasks per dimension
nodes_dim = 4

# subvolume size
ncc = 768

# volume size
rnc = 3072.

for k1 in range(1, nodes_dim+1):
    for j1 in range(1, nodes_dim+1):
        for i1 in range(1, nodes_dim+1):
            if fn_index == (i1 - 1) + (j1 - 1) * nodes_dim + (k1 - 1) * nodes_dim ** 2:
                print('found index '+str(fn_index)+' at '+str((i1, j1, k1)))
                node_coords = {'x': i1 - 1, 'y': j1 - 1, 'z': k1 - 1}
                print(node_coords)

# shift data
glob_data = loc_data
for col in ['x', 'y', 'z']:
    glob_data[col] = np.remainder(loc_data[col] + node_coords[col] * ncc, rnc)
    assert(max(glob_data[col] <= rnc))

# convert to Mpc/h
phys_data = glob_data * 505. / 3072.

phys_data.to_csv('for_treecorr.csv', header=False, index=False, sep=' ', columns=['x', 'y'])
