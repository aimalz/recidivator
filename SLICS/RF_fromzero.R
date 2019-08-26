require(reshape)
require(dplyr)
require(rayshader)
require(ggplot2)

N = 200 # Grid size
x <- seq(0,20,length.out = N) # 20 is arbitrary, box size
y <- seq(0,20,length.out = N)

###  Different kernels
se_kernel <- function(x, y, sigma = 1, length = 1) {
  sigma^2 * exp(- (x - y)^2 / (2 * length^2))
}

###
ExpCosKernel <- function(x, y, P = 0.45){
  cos((2*pi*(x-y)^2/P))
} 
###

### 
matern_kernel <- function(x, y, nu = 0.5, sigma = 10, l = 1) {
  if (!(nu %in% c(0.5, 1.5, 2.5))) {
    stop("p must be equal to 0.5, 1.5 or 2.5")
  }
  p <- nu - 0.5
  d <- abs(x - y)
  if (p == 0) {
    sigma^2 * exp(- d / l)
  } else if (p == 1) {
    sigma^2 * (1 + sqrt(3)*d/l) * exp(- sqrt(3)*d/l)
  } else {
    sigma^2 * (1 + sqrt(5)*d/l + 5*d^2 / (3*l^2)) * exp(-sqrt(5)*d/l)
  }
}

### Estimate covariance matrix for a given kernel choice
cov_matrix <- function(x,y, kernel_fn, ...) {
  outer(x, y, function(a, b) kernel_fn(a, b, ...))
}

### Evaluate kernel for a vector of x,y coordinates
K <- cov_matrix(x,y, kernel_fn = se_kernel)


### Sample a Gaussian Field with a kernel of choice
Y <- matrix(mvrnorm(N*N,mu = rep(0,N), Sigma = K), nrow = N, ncol = N)

### Just format data to be ggplot2 friendly
gY <- melt(Y)

### ggplot >>> matplotlib and seaborn
gg <- ggplot(data=gY,aes(x=X1,y=X2)) +
  geom_raster(aes(fill = value)) +
  scale_fill_viridis_c() + 
  theme_bw() 

### Just a fancy 3D plot, not usefull for anything
plot_gg(gg,multicore=TRUE,width=5,height=5,scale=250,windowsize=c(1200,800),
        zoom = 0.55, phi = 30)  # parameters of the 3D plot  



