data {
  int N; //Number of rows/vectors
  int D; //Number of dimensions

  vector[D] tavy[N]; //The data. This should be N D dimensional vectors.
}
transformed data {
  vector[D] mu_lower;
  vector[D] mu_width;

  mu_lower = [-5, -5, 14]';
  mu_width = [15, 15, 11]';
}
parameters {
  vector[D] mu_trans; //Transformed version of the mean
  vector[D] sigma; //Transformation of variance
  cholesky_factor_corr[D] L_Omega; //Correlation matrix
}
transformed parameters {
  vector[D] mu; //Mean
  cholesky_factor_cov[D] Sigma; //Covariance matrux

  mu = mu_lower + mu_width .* inv_logit(mu_trans);
  L_Sigma = diag_pre_multiply(sigma, L_Omega);
}
model {
  //Priors
  mu_trans ~ logistic(0,1);
  sigma[1:D-1] ~ cauchy(0,2);
  sigma[D] ~ cauchy(0,5);
  omega ~ lkj_corr(1);

  //Cycles through rows/observations and includes their contribution to the likelihood
  for (n in 1:N) {
    tavy[n] ~ multi_normal_cholesky(mu, L_Sigma);
  }
}
generated quantities {
  corr_matrix[D] Omega;
  cov_matrix[D] Sigma;

  //Generate samples of the actual matreices, rather than their cholesky factors
  Omega = multiply_lower_tri_self_transpose(L_Omega);
  Sigma = quad_form_diag(Omega, sigma);
}
