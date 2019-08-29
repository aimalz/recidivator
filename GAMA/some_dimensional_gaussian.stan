data {
  int N; //Number of rows/vectors
  int D; //Number of dimensions

  vector[D] data[N]; //The data. This should be N D dimensional vectors.
}
parameters {
  vector[D] mu; //Mean of the
  matrix[D,D] sigma; //Coariance matrix
}
model {
  for (n in 1:N) {
    data[n] ~ multi_normal(mu, sigma);
  }
}
