"""Module containing functions for the preparation of EEG data for use with ML models.

    Usage Example: [TODO]
"""

#TODO: should "windows" be called "epochs"?

import numpy as np
from scipy import stats

def create_windows(data, window_size, inter_window_interval):
    """
    Creates overlapping windows of EEG data from a single recording.
    
    Arguments:
        data: array of timeseries EEG data
        window_size(int): desired size of each window in number of samples
        inter_window_interval(int): interval between each window in number of samples
        
    Returns: 
        numpy array object with the windows along its first dimension
    """
    
    #Calculate the number of valid windows of the specified size which can be retrieved from data
    #Explanation: the max overflow possible is window_size, so we check how many 
    num_windows = 1 + (len(data) - window_size) // inter_window_interval
    
    windows = []
    for i in range(num_windows):
        windows.append(data[i*inter_window_interval:i*inter_window_interval + window_size])
    
    return np.array(windows)

def labels_from_timestamps(timestamps, sampling_rate, length):
    """takes an array containing timestamps (as floats) and
    returns a labels array of size 'length' where each index 
    corresponding to a timestamp via the 'samplingRate'.
    
    Arguments:
        timestamps: an array of floats containing the timestamps for each event (units must match sampling_rate).
        sampling_rate(float): the sampling rate of the EEG data.
        length(int): the number of samples of the corresponing EEG run. 
        
    Returns:
        an integer array of size 'length' with a '1' at each time index where a corresponding timestamp exists, and a '0' otherwise.
    """
    #TODO: function currently assumes binary labels. Generalize to arbitrary labels.


    #create labels array template
    labels = np.zeros(length)
    
    #calculate corresponding index for each timestamp
    labelIndices = np.array(timestamps * sampling_rate)
    labelIndices = labelIndices.astype(int)
    
    #flag label at each index where a corresponding timestamp exists
    labels[labelIndices] = 1
    labels = labels.astype(int)
    
    return np.array(labels)


def label_windows(labels, window_size, inter_window_interval, label_method):
    """create labels for individual windows of EEG data based on the label_method.
    For now, the "containment" and "count" label_methods assume binary labels.

    Arguments:
        labels: an integer array indicating a class for each time bin 
        window_size(int): size of each window in number of samples (matches window_size in EEG data)
        inter_window_interval(int): interval between each window in number of samples (matches inter_window_interval in EEG data)
        label_method(str): method of consolidating labels contained in window into a single label:
            "containment": whether a positive label is contained in the window,
            "count": the count of positive labels in the window,
            "mode": the most common label in the window
    
    Returns:
        an array with a label correponding to each window
    """

    windows = create_windows(labels, window_size, inter_window_interval)

    if label_method == "containment":
        window_labels = [int(1 in window) for window in windows]

    elif label_method == "count":
        # counts the number of occurences of the positive label (1)
        #TODO: maybe have have the label to be counted specified by the function
        #perhaps these should be broken up to different functions?
        window_labels = [window.count(1) for window in windows]

    elif label_method == "mode":
        #choose the most common occurence in the window and default to 0 if multiple exist
        window_labels = [stats.mode(window)[0][0] for window in windows]


    return np.array(window_labels)

def label_windows_from_timestamps(timestamps, sampling_rate, length, window_size, inter_window_interval, label_method="containment", raw_labels=False):
    """
    Directly creates labels for individual windows of EEG data from timestamps of events.

    Arguments:
        timestamps: an array of floats containing the timestamps for each event (units must match sampling_rate).
        sampling_rate(float): the sampling rate of the EEG data.
        length(int): the number of samples of the corresponing EEG run. 
        window_size(int): size of each window in number of samples (matches window_size in EEG data)
        inter_window_interval(int): interval between each window in number of samples (matches inter_window_interval in EEG data)
        label_method(str): method of consolidating labels contained in window into a single label:
            "containment": whether a positive label is contained in the window,
            "count": the count of positive labels in the window,
            "mode": the most common label in the window

    Returns:
        an array with a label correponding to each window
    """
    labels = labels_from_timestamps(timestamps, sampling_rate, length)

    window_labels = label_windows(labels, window_size, inter_window_interval, label_method)

    if not raw_labels: return window_labels
    else: return labels