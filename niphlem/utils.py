def resample_signal(y_old, t_old, t_new):
    """
    
    Function to resample the data to ECG Frequency.
    It is basically an interpolation operation using
    a cubic spline function.
    
    """
    from scipy.interpolate import interp1d
    
    interp = interp1d(t_old, y_old, kind='cubic', fill_value="extrapolate")
    y_new = interp(t_new) 
    return y_new
