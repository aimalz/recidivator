# How to `use classify_to_colors.py` pipeline

Here is a quick tutorial on how to use the current pipeline and what is currently going wrong with it!

## Funcitions Included:
- `clean_env_curves(fname_env, ftype)`: Remove problematic elements from synthetic environmental curves.
    - Code source: Emille
- `calc_dist_points(ndist: int, max_dist=2.5)`: Calculates distance evaluation points.
    - Code source: Emille
- `do_functional_PCA(data: np.array, ndist: int, npcs: int, ftype: str, max_dist=2.5, norm=False)`: Calculate functional principal components.
    - Code source: Emille
-  `build_feature_matrix(redshift: np.array, proj: np.array, max_env=None)`: Build normalized feature matrix (incl. redshift).
    - Code source: Emille
- `build_cluster_models(matrix: np.array, ngroups: list, screen=True, save=False, output_dir=None)`: Perform cluster and save models and classifications.
- `assign_groups(data: np.array, comp_orig: np.array, redshift: np.array, model: sklearn.cluster._kmeans.KMeans, npcs: int, norm=False)`:   Project synthetic data into principal functions. You _must_ ensure that the sklearn you are using is consistent between running this code and generating the cluster assignments.
    - Code source: Emille
-  `create_fit_summaries(df, lbl, str_z, patch, gnum, iter=1000, chains=4, warmup=500, savefiles=True, outdir='./')`:  Takes in a pandas data frame with 3 magnitudes and returns the means and covariance matrix of the corresponding 3D Gaussian distribution. Splits into high and low redshift as long as there are enough objects in the bin.
    - Code source: Kara
- `get_random_sample(label, redshift_bin, patch, group_num, indir='./')`: Uses scipy.stats.multivariate_normal to draw a random sample from the 3D multivariate normal fit in create_fit_summaries. This could also be redone by drawing the samples directly from STAN.
    - Code source: Kara
- `make_catalog_per_fname(f1, comp_orig, model, npcs, patch, ngroups)`: Creates a catalog given a filename. This is a helper function since the synthetic data is in chunks in different files.
    - Code source: Kara
- `make_corner(gama, mock, str_red, savefig=True)`: This should make a corner plot but it was created for the previous pipeline version and I have not tested it with the updated pipelines.
    - Code source: Kara
        
        
## Command line Keywords:
- `--debug`:  action=store_true, default=False
- `--verbose`, action=store_true, default=False
- `--no_files`,  default=True, action=store_false: If used, does not save any results
- `--outdir`, default="./": Locations of all outputs
- `--modeldir`, default="/media/CRP6/Cosmology/workdir/emille/results_norm/": Location of ML model for GAMA environment curve clutersting
- `--chunkdir_SLICS`, default="/media/CRP6/Cosmology/recidivator/SLICS/flexible_envirocurves/":  Location of Simulation Chunks
- `--chunkdir_GAMA`, default="/media/CRP6/Cosmology/recidivator/GAMA/flexible_envirocurves/": Location of GAMA Chunks
- `--gen_summaries`, default=False, action=store_true: Generate 3D Gaussian Stan fits to GAMA curves
- `--do_class`, default=False, action=store_true: Rerun the Principle Components + Kmeans Classification
- `--patch`, default='G15': Denote which GAMA patch you're using
- `--groups`, default=2: Number of groups to sort GAMA galaxies into.
- `--ndist`, default=100: Number of environment curve samples
- `--npcs`, default=15: Number of principle components
- `--max_dist`, default=2.5 degrees: Maximum distance from galaxy when measuring environment curve.
- `--nonorm`, default=True, action=store_false: Input for `do_functional_PCA`
- `--los`, dest='LOS', default='42': Line of Sight file number
- `--iter`, default=2000: Number of iterations for pySTAN

## Different ways to run the code.
- Run everything beginning to end:

`python classify_to_colors.py --outdir ./ --modeldir ./ --do_class --gen_summaries --patch G09 --groups 10`

- Only generate summaries (if GAMA classifcation already done):

`python classify_to_colors.py --outdir ./ --gen_summaries --patch G09 --groups 10`

- Create new catalogs because GAMA classification and STAN models already run:

`python classify_to_colors.py --outdir ./ --patch G09 --groups 10`

Output is a file name `results_LOS42_G15_10groups_15PCs.csv`.

## Current errors:
- Labels are missing from file being passed to `create_fit_summaries` which is the output of `assign_groups`. Code needs to be added to ensure a column named "label" is passed to `create_fit_summaries`.
