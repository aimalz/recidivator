data {
  int N; //Number of rows/vectors
  int D; //Number of dimensions

  vector[D] tavy[N]; //The data. This should be N D dimensional vectors.
}
parameters {
  vector[D] mu; //Mean of the
  vector<lower=0>[D] sigma; //Something like a variance, I guess
  cholesky_factor_corr[D] L_Omega; //Correlation matrix
}
transformed parameters {
  cholesky_factor_cov[D] Sigma; //Covariance matrux

  L_Sigma = diag_pre_multiply(sigma, L_Omega);
}
model {
  //Priors
  mu[1:D-1] ~ uniform(-5,10);
  mu[D] ~ uniform(14,25);
  sigma[1:D-1] ~ uniform(0.1,2);
  sigma[D] ~ cauchy(0,5);
  omega ~ lkj_corr(1);

  //Cycles through rows/observations and includes their contribution to the likelihood
  for (n in 1:N) {
    tavy[n] ~ multi_normal_cholesky(mu, L_Sigma);
  }

generated quantities {
  corr_matrix[D] Omega;
  cov_matrix[D] Sigma;

  //Generate samples of the actual matreices, rather than their cholesky factors
  Omega = multiply_lower_tri_self_transpose(L_Omega);
  Sigma = quad_form_diag(Omega, sigma);
}
