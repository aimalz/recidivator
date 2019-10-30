# 
# R DEAR :: Density Estimation And Resampling
# 
# COIN 2019
# 

require(spatstat)
require(sparr)
require(data.table)

# This is a function to use in small datasets. Not yet subdividing the space. 
DEARsmall <- function(filename="./Data/pos10000.csv", nPointsNew=1000, createNewDensity=FALSE, viewPlots=FALSE) {
	# Read data
	myGals <- fread(filename)
	
	# Reshape into [0;1]
	xx <- myGals$V1
	yy <- myGals$V2
	xx <- xx - min(xx)
	xx <- xx/max(xx)
	yy <- yy - min(yy)
	yy <- yy/max(yy)
	
	if(viewPlots) {
		# Create a scatter plot of the original positions... we are visual beasts...
		plot(xx, yy, pch=19, cex=0.1, asp=1, main="Original points")
	}
	
	# Perform the density estimate
	myDenEst <- doDensEst(xx, yy, FALSE)
	
	if(viewPlots) {
		# Create an image of the density estimate
		image((myDenEst), asp=1, main="Original density")
	}
		
	# Sample a new point set from the density estimate
	myNewData <- doRejSamp(myDenEst, nPoints=nPointsNew)
	
	if(viewPlots) {
		# Create a scatter plot of the new positions
		plot(myNewData$x, myNewData$y, pch=19, cex=0.1, asp=1, main="New points")
	}
	
	if(createNewDensity) {
		# Perform a new density estimate using the new positions (just to verify)
		myDenEstTest <- doDensEst(myNewData$x/max(myNewData$x), myNewData$y/max(myNewData$y), FALSE)
	
		# Create an image of the density estimate of the new positions (just to verify)
		image((myDenEstTest), asp=1, main="New density")
	}

	return(myNewData)	
}

# Density estimation using sparr::bivatiate.density. 
# Added as an option to use the multiscale version found by Rafa.
doDensEst <- function(x, y, createPlot=FALSE, multiscale=FALSE, h0=0.01, resolution=128) {	
	myPointData <- ppp(x, y, xrange=range(x), yrange=range(y))
	if(multiscale) {
		ddest <- multiscale.density(myPointData, h0=h0)
	} else {
		ddest <- bivariate.density(myPointData, h0=h0, adapt=TRUE, resolution=resolution)
	}
	if(createPlot) {
		plot(ddest)
	}
	return(ddest$z)
}

# Simple rejection sampling 
doRejSamp <- function(densEst, nPoints=1000, depuration=FALSE, serial=TRUE, batch = ceiling(nPoints/2)) {
	
	densEst <- densEst/max(densEst, na.rm=TRUE)
	xdim <- dim(densEst)[1]
	ydim <- dim(densEst)[2]
	if(depuration) {
		print(xdim)
		print(ydim)
	}
		
	myNewXX <- vector(mode="numeric", length=nPoints)
	myNewYY <- vector(mode="numeric", length=nPoints)
	selPoints <- 1
	
	if(serial) {
		# Very simple and lazy serial rejection
		while(selPoints <= nPoints) {
			xx <- (runif(1) * xdim)
			yy <- (runif(1) * ydim)
			zz <- runif(1)
			# Check the validity of the new sampled point
			if(densEst[ceiling(yy), ceiling(xx)] > zz) { 
				myNewXX[selPoints] <- xx
				myNewYY[selPoints] <- yy
				selPoints <- selPoints + 1	
			} 
		}
	} else {
		# Very simple rejection in batches, a bit less lazy; 
		# it should be at least 3x faster than the serial version due to vectorization. 
		# But not working that fast yet as of 30th oct.
		while(selPoints <= nPoints) {
			myDf <- matrix(runif(batch*3), ncol=3)
			myDf[,1] <- myDf[,1] * xdim
			myDf[,2] <- myDf[,2] * ydim
			validI <- apply(myDf, 1, function(x, densEst){ densEst[ceiling(x[2]), ceiling(x[1])] > x[3] }, densEst)
			if(depuration) {
				print(length(validI))
			}
			# How many new points to get?			
			nToGet <- length(which(validI))
			if(depuration) {
				cat(paste(batch, " -- ", nToGet," -- "))
			}
			if( (selPoints+nToGet) > nPoints ) {
				nToGet <- nToGet - ((selPoints+nToGet) - nPoints) + 1
			} 
			if(depuration) {
				cat(paste(nToGet," -- ",selPoints, " -- ", nPoints, "\n"))
			}
			if(nToGet >= 1) {
				myNewXX[selPoints:(selPoints + nToGet -1)] <- myDf[ which(validI)[1:nToGet], 1 ]
				myNewYY[selPoints:(selPoints + nToGet -1)] <- myDf[ which(validI)[1:nToGet], 2 ]
				selPoints <- selPoints + nToGet
			}	
		}
	}
	
	return(data.frame(x=myNewXX, y=myNewYY))
}

