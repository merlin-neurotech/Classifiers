"""Module containing functions for the preparation of EEG data for use with ML models.

    Usage Example: [TODO]
"""

import numpy as np
from scipy import stats


def epoch(signal, window_size, inter_window_interval):
    """
    Creates overlapping windows/epochs of EEG data from a single recording.
    
    Arguments:
        signal: array of timeseries EEG data of shape [n_samples, n_channels]
        window_size(int): desired size of each window in number of samples
        inter_window_interval(int): interval between each window in number of samples (measured start to start)
        
    Returns: 
        numpy array object with the epochs along its first dimension
    """
    
    #Calculate the number of valid windows of the specified size which can be retrieved from data
    num_windows = 1 + (len(signal) - window_size) // inter_window_interval
    
    epochs = []
    for i in range(num_windows):
        epochs.append(signal[i*inter_window_interval:i*inter_window_interval + window_size])
    
    return np.array(epochs)


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


def label_epochs(labels, window_size, inter_window_interval, label_method):
    """create labels for individual eoicgs of EEG data based on the label_method.
    Note: For now, the "containment" and "count" label_methods assume binary labels.

    Arguments:
        labels: an integer array indicating a class for each sample measurement
        window_size(int): size of each window in number of samples (matches window_size in EEG data)
        inter_window_interval(int): interval between each window in number of samples (matches inter_window_interval in EEG data)
        label_method(str): method of consolidating labels contained in epoch into a single label:
            "containment": whether a positive label is contained in the epoch,
            "count": the count of positive labels in the epoch,
            "mode": the most common label in the epoch
    
    Returns:
        a numpy array with a label correponding to each epoch
    """

    # epoch the labels themselves so each epoch contains a label at each sample measurement
    epochs = epoch(labels, window_size, inter_window_interval)

    #if a positive label [1] is occurs in the epoch, give epoch positive label
    if label_method == "containment":
        epoch_labels = [int(1 in epoch) for epoch in epochs]

    elif label_method == "count":
        # counts the number of occurences of the positive label [1]]
        #TODO: maybe have have the label to be counted specified by the function
        #perhaps these should be broken up to different functions?
        epoch_labels = [epoch.count(1) for epoch in epochs]

    elif label_method == "mode":
        #choose the most common label occurence in the epoch and default to the smallest if multiple exist
        epoch_labels = [stats.mode(epoch)[0][0] for epoch in epochs]


    return np.array(epoch_labels)


def label_epochs_from_timestamps(timestamps, sampling_rate, length, window_size, inter_window_interval,
                                 label_method="containment"):
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

    epoch_labels = label_epochs(labels, window_size, inter_window_interval, label_method)

    return epoch_labels


def epoch_and_label(data, sampling_rate, timestamps, window_size, inter_window_interval, label_method="containment"):
    """
    Epochs a signal (single EEG recording) and labels each epoch using timestamps of events
    and a chosen labelling method.
    
    Arguments:
        data: array of timeseries EEG data of shape [n_samples, n_channels]
        timestamps: an array of floats containing the timestamps for each event (units must match sampling_rate).
        sampling_rate(float): the sampling rate of the EEG data.
        window_size(float): desired size of each window in units of time (matches sampling_rate)
        inter_window_interval(float): interval between each window in units of time (measured start to start; matches sampling_rate)
        label_method(str): method of consolidating labels contained in window into a single label:
            "containment": whether a positive label is contained in the window,
            "count": the count of positive labels in the window,
            "mode": the most common label in the window

        
    Returns: 
        epochs: an array of the epochs with shape [n_epochs, n_channels]
        labels: an array of the labels corresponding to each epoch of shape [n_epochs, ]
    """
    
    epochs = epoch(data, int(window_size*sampling_rate), int(inter_window_interval*sampling_rate))
    labels = label_epochs_from_timestamps(timestamps, sampling_rate, len(data), 
                                          int(window_size*sampling_rate), int(inter_window_interval*sampling_rate), 
                                          label_method=label_method)
    
    return epochs, labels