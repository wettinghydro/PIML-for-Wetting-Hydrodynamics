import numpy as np
from similaritymeasures import frechet_dist
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean

def arc_length_parametrization(curve):
    """
    Parametrize a curve by its arc length.
    
    Parameters:
    curve (numpy.ndarray): Array of shape (n, d) representing n points in d dimensions
    
    Returns:
    tuple: (arc_lengths, normalized_arc_lengths) where arc_lengths are the cumulative
           distances along the curve and normalized_arc_lengths are in [0, 1]
    """
    # Calculate distances between consecutive points
    diffs = np.diff(curve, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    
    # Calculate cumulative distance (arc length) along the curve
    arc_lengths = np.zeros(len(curve))
    arc_lengths[1:] = np.cumsum(segment_lengths)
    
    # Normalize arc lengths to [0, 1]
    total_length = arc_lengths[-1]
    normalized_arc_lengths = arc_lengths / total_length if total_length > 0 else arc_lengths
    
    return arc_lengths, normalized_arc_lengths, total_length

def resample_curve(curve, num_points=100):
    """
    Resample a curve to have a fixed number of points, evenly spaced by arc length.
    
    Parameters:
    curve (numpy.ndarray): Array of shape (n, d) representing n points in d dimensions
    num_points (int): Number of points in the resampled curve
    
    Returns:
    numpy.ndarray: Resampled curve with num_points points
    """
    # Get arc length parametrization
    arc_lengths, normalized_arc_lengths, _ = arc_length_parametrization(curve)
    
    # Handle special case of a single point or all points being the same
    if len(curve) == 1 or np.allclose(arc_lengths[-1], 0):
        return np.tile(curve[0], (num_points, 1))
    
    # Create interpolation function for each dimension
    dimensions = curve.shape[1]
    resampled_curve = np.zeros((num_points, dimensions))
    
    # New parameter values for evenly spaced points
    new_params = np.linspace(0, 1, num_points)
    
    # Interpolate each dimension
    for dim in range(dimensions):
        if len(curve) > 1:  # Only interpolate if we have at least 2 points
            interp_func = interp1d(normalized_arc_lengths, curve[:, dim], kind='cubic')
            resampled_curve[:, dim] = interp_func(new_params)
        else:
            resampled_curve[:, dim] = curve[0, dim]
    
    return resampled_curve

def improved_frechet(curve1, curve2, num_points=100, closed=False):
    """
    Calculate the Frechet distance between two curves after resampling them
    to have evenly spaced points according to arc length.
    
    Parameters:
    curve1 (numpy.ndarray or list): First curve
    curve2 (numpy.ndarray or list): Second curve
    num_points (int): Number of points to resample to
    closed (bool): Whether the curves are closed (loops)
    
    Returns:
    float: Frechet distance between the resampled curves
    """
    # Convert to numpy arrays if necessary
    curve1 = np.array(curve1)
    curve2 = np.array(curve2)
    
    # Resample curves
    resampled_curve1 = resample_curve(curve1, num_points)
    resampled_curve2 = resample_curve(curve2, num_points)
    
    # For closed curves, add the first point to the end and handle rotations
    if closed:
        # Add first point to the end to close the curve
        resampled_curve1 = np.vstack((resampled_curve1, resampled_curve1[0]))
        resampled_curve2 = np.vstack((resampled_curve2, resampled_curve2[0]))
        
        # Try different rotations of one curve and find the minimum distance
        min_distance = np.inf
        for j in range(num_points):
            # Rotate the points of the first curve
            rotated_curve1 = np.roll(resampled_curve1[:-1], j, axis=0)
            rotated_curve1 = np.vstack((rotated_curve1, rotated_curve1[0]))  # Re-close the curve
            
            # Calculate Frechet distance
            distance = frechet_dist(rotated_curve1, resampled_curve2)
            min_distance = min(min_distance, distance)
        
        return min_distance
    else:
        return frechet_dist(resampled_curve1, resampled_curve2)
