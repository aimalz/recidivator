import numpy as np

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['savefig.dpi'] = 250
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['savefig.bbox'] = 'tight'
import matplotlib.pyplot as plt
import matplotlib.colors

our_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('our_cmap',
            ['#1B3B52', '#45C0CE', '#F7DB3F','#F26811', '#821308'])
