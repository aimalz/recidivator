# 
# R DEAR :: Density Estimation And Resampling
# 
# COIN 2019
# 

require(spatstat)
require(sparr)
require(data.table)
require(foreach)
require(doParallel)
require(scales)


# This function just partition the data and compute the kernel density estimates, saving them to files.
# This is the function to use in middle-sized datasets. Subdividing the space using a quad-tree and multicore processing for the moment.
DEARmiddleAKDE <- function(filename="./Data/pos10000.csv", maxPointsPerPartition=100000, nCores=1, parallelization=c("multicore", "MPI")[1], outFileDir="./Data/") {
	# Register the parallel backend if necessary
	if(nCores==1) {
		registerDoSEQ()
	} else {
		if(parallelization=="multicore") {
			cl <- parallel::makeForkCluster(nCores)
			doParallel::registerDoParallel(cl)
		} else if(parallelization=="MPI") {
			stop("MPI functionality is not yet implemented. May the patience be with you...")
		}
	}
	
	# Read data
	myGals <- fread(filename)
	
	# Partition the data in different regions with approximately the same number of objects
	pointsPartitionIdx <- getNBalancedSpatialPartitionIds(x=myGals$V1, y=myGals$V2, maxN=maxPointsPerPartition)
	
	# Dispatch the Adaptive KDE estimates in different nodes or cores,
	# as registered by the paralelization environment, and write the different files.
	uniquePartitions <- unique(pointsPartitionIdx$id)
	foreach(i = 1:length(uniquePartitions)) %dopar% {
		idxPts <- which(pointsPartitionIdx$id==uniquePartitions[i])
  		DEARsmallToOutFile(x = pointsPartitionIdx$x[ idxPts ], 
  						   y = pointsPartitionIdx$y[ idxPts ] , 
  						   outfilename=paste(outFileDir,"DEAR_AKDE_",uniquePartitions[i],".rds",sep=""))
	}

	# Stop the parallel environment, if necessary
	if(nCores>1) {
		if(parallelization=="multicore") {
			parallel::stopCluster(cl)
		} else if(parallelization=="MPI") {
			stop("MPI functionality is not yet implemented. May the patience be with you...")
		}
	}
}


########################################################################################
# THIS FUNCTION IS STILL NOT WORKING.
# This function just resample new points from previously estimated density distributions. 
# This is a function to use in middle-sized datasets. Only multicore is implemented for the moment.
DEARmiddleRejSampler <- function(inFileDir="./Data/", nPointsNew=10000, nCores=1, parallelization=c("multicore", "MPI")[1], outFileDir="./Data/") {
	# Register the parallel backend if necessary
	if(nCores==1) {
		registerDoSEQ()
	} else {
		if(parallelization=="multicore") {
			cl <- parallel::makeForkCluster(nCores)
			doParallel::registerDoParallel(cl)
		} else if(parallelization=="MPI") {
			stop("MPI functionality is not yet implemented. May the patience be with you...")
		}
	}
	
	# Grab the list of regions where the density was estimated beforehand
	regionsToEstimateDensity <- list.files(inFileDir, pattern=".rds", full.names=TRUE)
											
	# Dispatch the request to perform the sampling in multiple cores,
	# as registered by the paralelization environment, and write the different files, 
	# one per region.
	pointsPerRegion <- ceiling(nPointsNew/length(regionsToEstimateDensity))
	foreach(i = 1:length(regionsToEstimateDensity)) %dopar% {
		DEARsampleFromDensFile(nPoints=pointsPerRegion, infilename=regionsToEstimateDensity[i], 
							   outfilename=paste(outFileDir,"DEAR_NEWPOSITIONS_",i,".csv",sep="")) 
	}

	# Stop the parallel environment, if necessary
	if(nCores>1) {
		if(parallelization=="multicore") {
			parallel::stopCluster(cl)
		} else if(parallelization=="MPI") {
			stop("MPI functionality is not yet implemented. May the patience be with you...")
		}
	}
}

# This is a function to use in small datasets. Not yet subdividing the space.
DEARsmallToOutFile <- function(x, y, outfilename="./Data/adapkde.rds") {
	# Boundaries
	mins <- c(min(x), min(y))
	maxs <- c(max(x), max(y))

	# Reshape into [0;1]
	xx <- x
	yy <- y
	xx <- xx - min(xx)
	xx <- xx/max(xx)
	yy <- yy - min(yy)
	yy <- yy/max(yy)
		
	# Perform the density estimate
	myDenEst <- doDensEst(xx, yy, FALSE)
	
	# Write the file
	saveRDS(object=list(kde2d=myDenEst, mins=mins, maxs=maxs), file=outfilename)
}

# Perform the sampling by reading the density from a file
DEARsampleFromDensFile <- function(nPoints, infilename="./Data/adapkde.rds", outfilename="./Data/DEAR_REJSAMP.csv") {
	# read density file
	myDensData <- readRDS(infilename)
	
	# Sample the points
	newPoints <- doRejSamp(myDensData$kde2d, nPoints=nPoints, serial=FALSE)
	
	# Rescale the intervals
	newPoints$x <- rescale(newPoints$x, to = c(myDensData$mins[1], myDensData$maxs[1]))
	newPoints$y <- rescale(newPoints$y, to = c(myDensData$mins[2], myDensData$maxs[2]))
	
	# Write the file
	write.csv(newPoints, file=outfilename)
}

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


#########################################################################################
# 2D space partioning in similar number of points
# Lisbon, 2019. For the YSO and Cosmology-1 CRP6 projects.
#########################################################################################

# Breaks a 1D data set in two, each one containing a similar number of points
getTwoBalancedSpatialPartitionIds <- function(x) {
  n1 <- ceiling(length(x)/2)
  n2 <- floor(length(x)/2)
  partA <- x[order(x)][1:n1]
#  partB <- x[rev(order(x))][1:n2]
  retIds <- rep(1, length(x))
  retIds[which(x %in% partA)] <- 2
  return(retIds)
}

# Breaks the 2D data set in four regions with a similar number of points
getFourBalancedSpatialPartitionIds <- function(x, y, id=rep(0, length(x))) {

  retIds <- id*10#rep(NA, length(x))

  # First slpit in two in x
  partAsplit <- getTwoBalancedSpatialPartitionIds(x)

  # then split each partition in two in y
  a1Ids <- which(partAsplit==1)
  a2Ids <- which(partAsplit==2)
  partB1 <- getTwoBalancedSpatialPartitionIds(y[a1Ids])
  partB2 <- getTwoBalancedSpatialPartitionIds(y[a2Ids])

  # Finally, return the ids
  retIds <- retIds + partAsplit*10
  retIds[a1Ids] <- retIds[a1Ids] + partB1
  retIds[a2Ids] <- retIds[a2Ids] + partB2

  return(data.frame(x=x, y=y, id=retIds))
}

# Breaks recursively the data set in four until the largest partition contains at most maxN points
getNBalancedSpatialPartitionIds <- function(x, y, id=rep(0, length(x)), maxN=40) {

  # Perform first partition
  partitions <- getFourBalancedSpatialPartitionIds(x, y, id)
  partionIdInfo <- getNmembersPerId(partitions$id)

  # while there are partitions with more than the maximum number of members, divide!
  while( max(partionIdInfo$nMembers) >= maxN ) {

    partIdsToDivide <-partionIdInfo$id[which(partionIdInfo$nMembers >= maxN)]
    for(i in 1:length(partIdsToDivide)) {
      presentId <- partIdsToDivide[i]
      arrIds <- which(partitions$id == presentId)
      tempPartitions <- getFourBalancedSpatialPartitionIds(x[arrIds], y[arrIds], partitions$id[arrIds])
      partitions$id[arrIds] <- tempPartitions$id
    }

    partionIdInfo <- getNmembersPerId(partitions$id)

  }

  return(data.frame(x=x, y=y, id=partitions$id))

}

# Return a data frame containing the id of the region and the number of points inside the region
getNmembersPerId <- function(id) {
  uId <- unique(id)
  retDf <- data.frame(id=rep(NA,length(uId)), nMembers=rep(NA,length(uId)) )
  for(i in 1:length(uId)) {
    retDf$id[i] <- uId[i]
    retDf$nMembers[i] <- length( which(id == uId[i]) )
  }
  return(retDf)
}

runTestgetFourBalancedSpatialPartitionIds <- function() {
  xx <- c(runif(100), runif(100, 0, 0.3))
  yy <- c(runif(100), runif(100, 0, 0.3))
  ppCol <- getFourBalancedSpatialPartitionIds(xx, yy)
  plot(xx,yy, pch=19, col=as.factor(ppCol))
}

runTestgetNBalancedSpatialPartitionIds <- function() {
  xx <- c(runif(100), runif(100, 0, 0.3))
  yy <- c(runif(100), runif(100, 0, 0.3))
  ppCol <- getNBalancedSpatialPartitionIds(xx, yy, maxN=40)
  plot(xx,yy, pch=19, col=as.factor(ppCol$id))
}
