"""
Post-processing: PhysNet's DiffNormalized rPPG output -> heart rate (BPM).

Ported from rPPG-Toolbox's evaluation/post_process.py, specifically the
diff_flag=True path: cumsum (integrate the differenced signal) -> detrend ->
bandpass filter -> FFT peak in the physiological HR band.
"""
import numpy as np
from scipy import signal


def _detrend(input_signal, lambda_value=100):
    """Smoothness-priors detrending (Tarvainen et al.), matches
    post_process.py's _detrend exactly."""
    signal_length = input_signal.shape[0]
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags_dense(diags_data, diags_index, signal_length - 2, signal_length)
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal
    )
    return filtered_signal


def spdiags_dense(data, diags, m, n):
    """Dense equivalent of scipy.sparse.spdiags(...).toarray(), avoiding a
    sparse-matrix dependency mismatch; identical numerical result."""
    from scipy.sparse import spdiags
    return spdiags(data, diags, m, n).toarray()


def _calculate_fft_hr(ppg_signal, fs=30, low_pass=0.75, high_pass=2.5, min_nfft=4096):
    """FFT-based HR: periodogram peak within [low_pass, high_pass] Hz,
    converted to BPM. Matches post_process.py's recommended band (paper
    uses 0.75-2.5 Hz rather than the wider toolbox default of 0.6-3.3 Hz).

    min_nfft: the reference implementation uses nfft = next_power_of_2(N),
    which for a single 128-frame chunk gives nfft=128 -- no zero-padding at
    all, so peaks can only land on ~14 BPM-wide bins. Zero-padding the FFT
    well beyond the signal length doesn't add information, but it does
    interpolate the periodogram much more finely, letting the argmax peak
    picker land close to the true underlying frequency instead of snapping
    to the nearest coarse bin. This matters most for short (128-sample)
    chunks; for longer concatenated signals N may already exceed min_nfft.
    """
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = max(_next_power_of_2(ppg_signal.shape[1]), min_nfft)
    f_ppg, pxx_ppg = signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


def _next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _butter_bandpass(sig, fs=30, low_pass=0.75, high_pass=2.5):
    b, a = signal.butter(2, [low_pass / (fs / 2), high_pass / (fs / 2)], btype="bandpass")
    return signal.filtfilt(b, a, np.double(sig))


def rppg_to_bpm(rppg_waveform, fs=30, low_pass=0.75, high_pass=2.5):
    """Convert one chunk's raw PhysNet output (DiffNormalized rPPG, shape [T])
    into a single BPM estimate.

    Steps (matching calculate_metric_per_video's diff_flag=True branch):
      1. cumsum to integrate the differenced signal back into a waveform
      2. detrend (smoothness priors)
      3. Butterworth bandpass filter in the physiological HR band
      4. FFT peak -> BPM
    """
    waveform = np.cumsum(rppg_waveform)
    waveform = _detrend(waveform, 100)
    waveform = _butter_bandpass(waveform, fs=fs, low_pass=low_pass, high_pass=high_pass)
    hr = _calculate_fft_hr(waveform, fs=fs, low_pass=low_pass, high_pass=high_pass)
    return float(hr)


def aggregate_bpm(bpm_list):
    """Aggregate per-chunk BPM estimates into one final value (median is
    robust to a single bad chunk, e.g. from motion or a blink)."""
    if len(bpm_list) == 0:
        raise ValueError("No BPM estimates to aggregate.")
    return float(np.median(bpm_list))
