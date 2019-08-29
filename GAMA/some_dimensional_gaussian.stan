data {
  int N; //Number of rows/vectors
  int D; //Number of dimensions

  vector[D] data[N]; //The data. This should be N D dimensional vectors.
}
parameters {
  vector[D] mu; //Mean of the
  vector<lower=0>[D] sigma; //Something like a variance, I guess
  corr_matrix[D] omega; //Correlation matrix
}
transformed parameters {
  cov_matrix[D] Sigma; //Covariance matrux

  Sigma = quad_form_diag(omega, sigma);
}
model {
  mu[1:D-1] ~ uniform(-5,10);
  mu[D] ~ uniform(14,25);
  sigma[1:D-1] ~ uniform(0.1,2);
  sigma[D] ~ uniform(0.1,5);
  omega ~ lkj_corr(1);
  //Cycles through rows/observations and includes their contribution to the likelihood
  for (n in 1:N) {
    data[n] ~ multi_normal(mu, Sigma);
  }
}
