import numpy as np

def nn_finder(data, point, radius):
    """
    Extracts coordinates that fall within a certain radius around a point.
    
    Inputs:
    -------
    data : array-like
        (N, 2) array, with each row being a point coordinate. Declination
        values are in the first column, right ascension values in the second.
        T his is the dataset of coordinates among which we want to find the
        ones that fall within a certain radius using the great-circle distance.
           
    point : array-like
        Cooridnate tuple with the declination value as the first entry and
        the right ascension value as the second entry. This is the point
        around which we are looking for coordinateś within a radius.
            
    radius : float or int
        The radius around the provided point in which extracted coordinates
        have to fall in order to be returned as valid coordinates to return.
        
    Returns:
    --------
    radius_data : array-like
        The cut-down dataset, meaning only the coordinates of points in the
        provided dataset that fall within the given radius.
        
    Note:
    ----
    The whole things runs in around 0.7 seconds per point, with a dataset of
    100,000 coordinates. For a speed-up, it is embarrassingly parallelizable.
    It should also be tested on the real dataset to make sure that everything
    works as intended. Batteries not included.
             
           
    """
    # Specify the cutoff values for the pre-cutting
    x_cut_lower = point[0] - radius
    x_cut_upper = point[0] + radius
    y_cut_lower = point[1] - radius
    y_cut_upper = point[1] + radius
    # Find the indices of eligible points across dimensions
    eligible_points = np.where((data[:, 0] >= x_cut_lower) 
                               & (data[:, 0] <= x_cut_upper)
                               & (data[:, 1] >= y_cut_lower)
                               & (data[:, 1] <= y_cut_upper))[0]
    # Cut based on the radius boundaries in dimension 1
    cut_data = data[eligible_points, :]
    # Extract the points within the specified radius
    distances = np.asarray([haversine(point, cut_data[i]) 
                            for i in range(0, len(cut_data))])
    radius_data = cut_data[np.where(distances <= radius)[0]]
    # Return the extracted points
    

def haversine(point_1, 
              point_2):

    """
    Calculates the great-circle distance based on the haversine function.

    Inputs:
    ------
    point_1 : array_like
        Cooridnate tuple with the declination value as the first entry and
        the right ascension value as the second entry.

    point_2 : array_like
        Cooridnate tuple with the declination value as the first entry and
        the right ascension value as the second entry.
        
    Returns:
    --------
    haversine_distance : float
        The great-circle distance between the two provided points using the
        haversine formula. Better than the Euclidean distance in this case.
        
    Note:
    -----
    The value 'circle_radius' has to be set, in kilometres. The one used
    below is just a random value for testing and should not be used.
    """
    # Specify the radius of the sphere in kilometetres
    circle_radius = 10
    # Extract Dec and RA values from the provided points
    dec_1 = point_1[0]
    dec_2 = point_2[0]
    ra_1 = point_1[1]
    ra_2 = point_2[1]
    # Convert the Dec and RA values to radians
    dec_1, ra_1 = np.radians((dec_1, ra_1))
    dec_2, ra_2 = np.radians((dec_2, ra_2))
    # Calculate the differences between Dec values in radians
    dec_difference = dec_2 - dec_1
    # Calculate the differences between RA values in radians
    ra_difference = ra_2 - ra_1
    # Calculate the haversine distance between the coordinates
    step_1 = np.square(np.sin(np.multiply(dec_difference, 0.5)))
    step_2 = np.square(np.sin(np.multiply(ra_difference, 0.5)))
    step_3 = np.multiply(np.cos(dec_1), np.cos(dec_2))
    step_4 = np.arcsin(np.sqrt(step_1 + np.multiply(step_2, step_3)))
    haversine_distance = np.multiply(np.multiply(2, circle_radius), step_4)
    # Return the computed haversine distance for the coordinates
    return haversine_distance
    return radius_data
    #return distances
