import pywt

def extract_channel_features(frames, sfreq=512):
    """
    Extracts features for each frame, channel, and sample from EEG data.
    Parameters:
    - frames: ndarray of shape (samples, n_frames, n_channels, n_samples).
    - sfreq: Sampling frequency of the EEG data (default 512 Hz).
    Returns:
    - features: ndarray of shape (samples, n_frames, n_channels, n_features).
    """
    samples, n_frames, n_channels, n_samples = frames.shape
    all_features = []
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50)
    }
    def bandpower(signal, band, sfreq):
        fmin, fmax = band
        freqs, psd = welch(signal, sfreq, nperseg=n_samples)
        return np.mean(psd[(freqs >= fmin) & (freqs <= fmax)])
    for sample in frames:
        sample_features = []
        for frame in sample:
            frame_features = []
            for ch in range(n_channels):
                ch_features = []
                for band_name, band in bands.items():
                    power = bandpower(frame[ch], band, sfreq)
                    ch_features.append(power)  # Power in each band
                beta_power = bandpower(frame[ch], bands["beta"], sfreq)
                alpha_power = bandpower(frame[ch], bands["alpha"], sfreq)
                ch_features.append(beta_power / alpha_power if alpha_power != 0 else 0)
                freqs, psd = welch(frame[ch], sfreq, nperseg=n_samples)
                cumulative_psd = np.cumsum(psd) / np.sum(psd)
                median_freq = freqs[np.where(cumulative_psd >= 0.5)[0][0]]
                ch_features.append(median_freq)
                spectral_entropy = -np.sum((psd / np.sum(psd)) * np.log2(psd / np.sum(psd)))
                ch_features.append(spectral_entropy)
                coeffs = pywt.wavedec(frame[ch], 'db4', level=4)
                ch_features.extend([np.mean(np.abs(c)) for c in coeffs])  # Average abs of wavelet coefficients
                approx_entropy = stats.entropy(np.abs(frame[ch]))
                ch_features.append(approx_entropy)
                skewness = stats.skew(frame[ch])
                kurtosis = stats.kurtosis(frame[ch])
                zcr = np.mean(np.diff(np.sign(frame[ch])) != 0)  # Zero-crossing rate
                ch_features.extend([skewness, kurtosis, zcr])
                frame_features.append(ch_features)
            sample_features.append(frame_features)
        all_features.append(sample_features)
    all_features = np.array(all_features)
    return all_features
extracted_features = extract_channel_features(all_sample_frames)
print("Extracted features shape:", extracted_features.shape)  # Expected shape: (samples, n_frames, n_channels, n_features)

np.shape(extracted_features)

samples, n_frames, n_channels, n_features = extracted_features.shape
reshaped_features = extracted_features.reshape(samples * n_frames, n_channels, n_features)
print("Reshaped features shape:", reshaped_features.shape)  # Expected shape: (samples * n_frames, n_channels, n_features)
expanded_labels = np.repeat(all_labels, n_frames)
print("Expanded labels shape:", expanded_labels.shape)  # Expected shape: (samples * n_frames,)

import pickle
save_path = "/content/drive/MyDrive/eeg_features_and_labels.pkl"
samples, n_frames, n_channels, n_features = extracted_features.shape
reshaped_features = extracted_features.reshape(samples * n_frames, n_channels, n_features)
expanded_labels = np.repeat(all_labels, n_frames)
with open(save_path, "wb") as f:
    pickle.dump({"features": reshaped_features, "labels": expanded_labels}, f)
print(f"Data saved successfully to {save_path}")

import pickle
save_path = "/content/drive/MyDrive/eeg_features_and_labels.pkl"
with open(save_path, "rb") as f:
    data = pickle.load(f)
reshaped_features = data["features"]
expanded_labels = data["labels"]
print("Loaded features shape:", reshaped_features.shape)  # Expected: (samples * n_frames, n_channels, n_features)
print("Loaded labels shape:", expanded_labels.shape)      # Expected: (samples * n_frames,)