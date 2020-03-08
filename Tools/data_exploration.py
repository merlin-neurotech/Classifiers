"""
Module containing functions to study and analyze neural signals, especially to provide insights for building
machine learning models to perform classification relevant to Brain-Computer Interfaces. 
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
import biosppy.signals as bsig

from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

def plot_signal(signal, sr, signal_type=None, ch_names=None, event_timestamps=None, **plt_kwargs):
    """
    Plots signal.

    Arguments: 
        signal: signal as array of shape [n_samples, n_channels].
        sr(float): sampling rate in samples per second.
        signal_type: (optional) gives a title for the y-axis.
        ch_names: (optional) array of names for each channel (used for legend).
        event_timestamps: (optional) 1-D array of times at which an event/stimulus occured.
        **plt_kwargs: matplotlib keyword args
    """
    
    ts = np.linspace(0, len(signal)/sr, len(signal))
    
    plt.plot(ts, signal, **plt_kwargs)

    if ch_names is not None: plt.legend(ch_names)
    
    if event_timestamps is not None:
        for timestamp in event_timestamps:
            plt.axvline(timestamp, ls='--', lw=1, c='violet')
    
    plt.xlabel("time [s]")
    plt.ylabel(signal_type)


def plot_grid(signals, num_signals=None, sr=1,cols=4, fig_size=(10,6), 
                    sharey=True, sharex=True, random=True, fig_axes=None, show=True):
    """
    Plot an (optionally random) set of signals [epochs] in a grid from an array of signals.

    Arguments: 
        signals: array of signals to plot from (num_signals, num_samples).
        num_signals(int): the number of siganls to plot.
        sr(float): sampling rate of signals.
        cols(int): the number of columns in the grid.
        fig_size: tuple (x,y) of figure size in inches.
        sharey(bool): whether to share scale on y-axis (see matplotlib).
        sharex(bool): whether to share scale on x-axis (see matplotlib).
        random(bool): whether to choose signals randomly or just use the first num_signals.
        fig_axes: optionally, an existing tuple of (fig,axes) to plot on (see matplotlib).
        show(bool): whether to show plot inline.

    Returns:
        fig, axes: matplotlib figure and axes with sample of signals plotted in a grid
    """
    
    # if num_signals to be plotted is not given, plot all signals
    if num_signals is None:
        num_signals = signals.shape[0]
    
    rows = int(np.ceil(num_signals / cols))
    
    if fig_axes is None:
        fig, axes = plt.subplots(rows, cols, sharey=sharey, sharex=sharex, figsize=fig_size)
    else:
        fig, axes = fig_axes

    if random:
        #choose a random set of signals to plot
        sampled_signals = signals[np.random.choice(range(len(signals)), num_signals, replace=False)]
    else: sampled_signals = signals[:num_signals] #choose the first num
    

    for r in range(rows):
        for c in range(cols):
            if (r*cols + c) >= num_signals: break
            sampled_signal = sampled_signals[r*cols+c]
            axes[r][c].plot(np.linspace(0, len(sampled_signal)/sr, len(sampled_signal)), sampled_signal)
        
    if show: plt.show()
    return fig, axes


def stim_triggered_average(signal, sr, timestamps, duration_before, duration_after, 
                           channels=None, std_multiplier=1, plot_range=True):
    """
    Inspired by the computational neuroscience concept of a spike-triggered average,
    this function computes the average signal characteristic around known events.

    Arguments:
        signal: signal in the form of an array of shape [samples, channels].
        sr(float): sampling rate of the signal.
        timestamps: array of floats containing the timestamps for each event (units must match sampling_rate).
        duration_before: the duration to be considered before each event.
        duration_after: the duration to be considered after each event.
        channels: list of channel names(str) corresponding to each channel in `signal`. used for plotting legend.
        std_multiplier(float): multiplier of standard deviation for computing upper and lower bounds of signal.
        plot_range: whether to also plot the std-based range for the stim-triggered average.

    Returns:
        relative_time: relative time of each sample in `sta` with respect to event.
        sta: average signal characteristic around event (stim-triggered-average).
        st_std: standard deviation of stim-triggered signal with respect to `relative_time`.
        figax: tuple of matplotlib figure and axis objects of plot.
    """
    
    # compute indices of of the stimulus event using the given timestamps 
    stim_indices = (timestamps*sr).astype(int)

    # compute number of indices to go forward and backward
    ind_before = int(sr * duration_before)
    ind_after = int(sr * duration_after)

    # extract "sub-signals" aligned and centered on stimulus event for computation of stim-triggered average and standard deviation
    stim_signals = np.array([signal[i-ind_before:i+ind_after] for i in stim_indices])

    # compute stim-triggered average `sta` and stim-trigerred standard deviation `st_std`
    sta = np.mean(stim_signals, axis=0)
    st_std = np.std(stim_signals, axis=0)

    # generate time space relative to stimulus event (event occurs at t=0) for plotting. 
    relative_time = np.linspace(-duration_before, duration_after, len(sta))

    # PLOTTING
    fig, ax = plt.subplots(1)

    # each channel is handled seperately so that the resultant plot is clean and interpretable. 
    for ch in range(sta.shape[1]):
        sta_line, = ax.plot(relative_time, sta[:, ch])
        if channels is not None:
            sta_line.set_label(channels[ch])
        if plot_range:
            color = sta_line.get_color()
            # compute the upper and lower bounds on the stim_triggered average using the stim_triggered standard deviation
            # `std_multiplier` controls sensitivity to deviation in the plot
            upper = sta[:, ch] + std_multiplier * st_std[:, ch]
            lower = sta[:, ch] - std_multiplier * st_std[:, ch]

            ax.plot(relative_time, upper, ls='--', c=color)
            ax.plot(relative_time, lower, ls='--', c=color)

            ax.fill_between(relative_time, lower, upper, facecolor=color, alpha=0.25)

    # indicate where stimulus occurs
    ax.axvline(0, c='violet', ls='--', lw=2, label='stim')

    ax.set_xlabel('Time Relative to Stim')
    ax.set_title('Stimulus-Triggerred Average')

    ax.legend()

    
    return relative_time, sta, st_std, (fig, ax)



def plot_PCA(epochs, sr=1, n_components=None, return_PCA=False, PCA_kwargs={}, plot_grid_kwargs={}):
    """
    performs independent component analysis and plots independent components of epochs of a signal.
    
    Arguments:
        epochs: array of epochs (n_epochs, n_samples).
        sr(float): sampling rate.
        num_components(int): number of components to use. If none is passed, all are used.
        return_PCA(bool): whether to return the independent components.
        PCA_kwargs(dict): dictionary containing kwargs for PCA function (see scikit-learn).
        plot_grid_kwargs(dict): dictionary contianing kwargs for plot_grid function.
 
    """
    
    pca = PCA(n_components=n_components, **PCA_kwargs)
    principle_components = pca.fit_transform(epochs.T).T
    
    plot_grid(principle_components, sr=sr, random=False, **plot_grid_kwargs)
    
    if return_PCA:
        return principle_components


def plot_ICA(epochs, sr=1, n_components=None, return_ICA=False, FastICA_kwargs={}, plot_grid_kwargs={}):
    """
    performs independent component analysis and plots independent components of epochs of a signal.
    
    Arguments:
        epochs: array of epochs (n_epochs, n_samples).
        sr(float): sampling rate.
        num_components(int): number of components to use. If none is passed, all are used.
        return_ICA(bool): whether to return the independent components.
        FastICA_kwargs(dict): dictionary containing kwargs for FastICA function (see scikit-learn).
        plot_grid_kwargs(dict): dictionary contianing kwargs for plot_grid function.
 
    """
    
    ICA = FastICA(n_components=n_components, **FastICA_kwargs)
    independent_components = ICA.fit_transform(epochs.T).T
    
    plot_grid(independent_components, sr=sr, random=False, **plot_grid_kwargs)
    
    if return_ICA:
        return independent_components


