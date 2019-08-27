import numpy as np
import pandas as pd
import treecorr

z_str = '0.042'
fn_base = 'xv'
fn_index = 27
fn_ext = '.dat'
fn = z_str + fn_base + str(fn_index) + fn_ext

dt_each = 'f' + str(4)
dt = np.dtype([('x', dt_each), ('y', dt_each), ('z', dt_each), ('vx', dt_each), ('vy', dt_each), ('vz', dt_each)])

with open(fn, 'rb') as f1:
    raw_data = np.fromfile(f1, dtype=dt)

loc_data = pd.DataFrame(data=raw_data[2:], columns=['x', 'y', 'z', 'vx', 'vy', 'vz'])

loc_data.to_csv('for_treecorr.csv', columns=['x', 'y'])
