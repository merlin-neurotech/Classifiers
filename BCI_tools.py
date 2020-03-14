from Tools.preprocessing import threshold_clf, epoch_band_features, epoch
from Tools.classification_tools import get_channels, softmax_predict, encode_ohe_prediction, decode_prediction, DEVICE_SAMPLING_RATE

import numpy as np
import time

def band_power_calibrator(inlet, channels, device, bands=['alpha_high'], percentile=50,
                          recording_length=10, epoch_len=1, inter_window_interval=0.2):
    '''
    Calibrator for `generic_BCI.BCI` which computes a given `percentile` for the power of each frequency band
    across epochs channel-wise. Useful for calibrating a concentration-based BCI.

    Arguments:
        inlet: a pylsl `StreamInlet` of the brain signal.
        channels: array of strings with the names of channels to use.
        device(str): device name for use by `classification_tools`
        bands: the frequency bands to get power features for.
            'all': all of ['theta', 'alpha_low', 'alpha_high', 'beta', 'gamma']
            otherwise an array of strings of the desired bands.
        percentile: the percentile of power distribution across epochs to return for each band
        recording_length(float): length of recording to use for calibration in seconds
        epoch_len(float): the length of each epoch in seconds
        inter_window_interval(float): interval between each window/epoch in units of time (measured start to start)

    Returns:
        clb_info: array of shape [n_bands, n_channels] of the `percentile` of the power of each band
    '''
    sr = DEVICE_SAMPLING_RATE[device] # get sampling_rate
    ws = int(epoch_len * sr) # calculate window size in # of samples
    iwi = int(inter_window_interval * sr) # calculate inter_window_interval in # of samples


    input("Press Enter to begin calibration...")

    print(f"Recording for {recording_length} seconds...")

    # sleep for recording_length while inlet accumulates chunk
    # necessary so that no data is used before the indicated start of recording
    time.sleep(recording_length)


    recording, _ = inlet.pull_chunk(max_samples=sr*recording_length) # get accumulated data
    recording = get_channels(np.array(recording), channels, device=device) # get appropriate channels

    # epoch the recording to compute percentile across distribution
    epochs = epoch(recording, ws, iwi)

    # compute band power for each epoch
    band_power = np.array([epoch_band_features(epoch, sr, bands=bands, return_dict=False) for epoch in epochs])

    # calculate given percentile of band power
    clb_info = np.squeeze(np.percentile(band_power, percentile, axis=0))

    print(f'\nComputed the following power percentiles: \n{clb_info}')
    input("\nCalibration complete. Press Enter to start BCI...")


    return clb_info