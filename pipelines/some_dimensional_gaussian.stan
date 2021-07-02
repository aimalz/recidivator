data {
  int N; //Number of rows/vectors
  int D; //Number of dimensions
  int N_resamples; //Number of samples to generate per iteration

  vector[D] tavy[N]; //The data. This should be N D=3 dimensional vectors.
}
transformed data {
  vector[D] mu_lower;
  vector[D] mu_width;

  mu_lower = [14, 14, 14]';
  mu_width = [11, 11, 11]';
}
parameters {
  vector[D] mu_trans; //Transformed version of the mean
  vector[D] sigma; //Transformation of variance
  cholesky_factor_corr[D] L_Omega; //Correlation matrix
}
transformed parameters {
  vector[D] mu; //Mean
  cholesky_factor_cov[D] L_Sigma; //Covariance matrux (cholesky factor)

  mu = mu_lower + mu_width .* inv_logit(mu_trans);
  L_Sigma = diag_pre_multiply(sigma, L_Omega);
}
model {
  //Priors
  mu_trans ~ logistic(0,1);
  sigma ~ cauchy(0,5);
  L_Omega ~ lkj_corr_cholesky(1);

  //Cycles through rows/observations and includes their contribution to the likelihood
  for (n in 1:N) {
    tavy[n] ~ multi_normal_cholesky(mu, L_Sigma);
  }
}
generated quantities {
  corr_matrix[D] Omega; //Correlation matrix
  cov_matrix[D] Sigma; //Covariance matrix
  vector[D] tavy_resampled[N_resamples]; //Resampled version of tavy

  //Generate samples of the actual matreices, rather than their cholesky factors
  Omega = multiply_lower_tri_self_transpose(L_Omega);
  Sigma = quad_form_diag(Omega, sigma);

  //Draws N_resamples realisations of tavy from the multivariate normal corresponding to this HMC sample of the mean and covariance
  for (n in 1:N_resamples) {
    tavy_resampled[n] = multi_normal_rng(mu, Sigma);
  }
}
