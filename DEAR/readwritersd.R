# Read and convert .rds file
require(reshape)
require(purrr)
require(colorspace)
require(lattice)
library(readr)
require(ggplot2)

datadir = "/media2/CRP6/Cosmology/RDEAR_output/0042/AKDE/"
plotdir =  "/media2/CRP6/Cosmology/RDEAR_output/0042/AKDE_png/"
outdir = "/media2/CRP6/Cosmology/RDEAR_output/0042/AKDE_csv/"

fnames = list.files(datadir)

for(i in 1:length(fnames)){

Ohdear <- readRDS(paste(datadir,fnames[i], sep=""))
number <- parse_number(fnames[i])

# If just visualization the above suffice
intensity <- Ohdear$kde2d$v # read intensity values, as a Matrix 128 x 128

# Visualize matrix via lattice
#png(filename = paste(plotdir,"DEAR_AKDE_",number,".png",sep=""))
#print(levelplot(intensity,panel = panel.levelplot.raster,
#          col.regions = rev(gray(0:100/100))))
#dev.off()

# If reshaping is needed, then go ahead
x <- Ohdear$kde2d$xcol  # read x values
y <- Ohdear$kde2d$yrow   # read y values
xy <- expand.grid(x=x,y=y) # create a grid 


# Here it creates a data.frame and changes column names using purrr
myDEAR <- cbind(xy,melt(intensity)[,3]) %>% set_names(c("x","y","value"))

# Visualize via ggplot2
#ggplot(myDEAR,aes(x=x,y=y,z=value)) + 
#  geom_tile(aes(fill=value)) +
#  scale_fill_viridis_c()

# If you want to prit the data

write.csv(myDEAR,paste(outdir,"DEAR_AKDE_",number,".csv", sep=""),row.names = F)
}
