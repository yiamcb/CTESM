import seaborn as sns
feature_names = [
    "Delta Power", "Theta Power", "Alpha Power", "Beta Power", "Gamma Power",
    "Beta-to-Alpha Ratio", "Median Frequency", "Spectral Entropy",
    "Wavelet Coef L1", "Wavelet Coef L2", "Wavelet Coef L3", "Wavelet Coef L4", "Wavelet Coef L5",
    "Approximate Entropy", "Skewness", "Kurtosis", "Zero-Crossing Rate"
]
healthy_features = reshaped_features[expanded_labels == 'Healthy']
pd_features = reshaped_features[expanded_labels == 'PD']
healthy_mean = np.mean(healthy_features, axis=0)  # Shape: (40, 17)
pd_mean = np.mean(pd_features, axis=0)            # Shape: (40, 17)
feature_diff = np.abs(healthy_mean - pd_mean)     # Absolute difference between classes
plt.figure(figsize=(16, 8))
for i, feature_name in enumerate(feature_names):
    plt.subplot(3, 6, i + 1)
    sns.boxplot(data=[healthy_features[:, :, i].flatten(), pd_features[:, :, i].flatten()], palette="Set2")
    plt.xticks([0, 1], ['Healthy', 'PD'])
    plt.legend([feature_name], loc='upper right')
plt.tight_layout(pad=2.0)
plt.savefig("/content/drive/MyDrive/feature_distributions.eps", format="eps", bbox_inches="tight")
plt.savefig("/content/drive/MyDrive/feature_distributions.pdf", format="pdf", bbox_inches="tight")
plt.show()
plt.figure(figsize=(10, 6))
sns.heatmap(feature_diff, cmap="coolwarm", cbar=True, yticklabels=[f"Channel {i+1}" for i in range(healthy_mean.shape[0])], xticklabels=feature_names, annot=False)
plt.xlabel("Features")
plt.ylabel("Channels")
plt.savefig("/content/drive/MyDrive/class_difference_heatmap.eps", format="eps", bbox_inches="tight")
plt.savefig("/content/drive/MyDrive/class_difference_heatmap.pdf", format="pdf", bbox_inches="tight")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
feature_names = [
    "Delta Power", "Theta Power", "Alpha Power", "Beta Power", "Gamma Power",
    "Beta-to-Alpha Ratio", "Median Frequency", "Spectral Entropy",
    "Wavelet Coef L1", "Wavelet Coef L2", "Wavelet Coef L3", "Wavelet Coef L4", "Wavelet Coef L5",
    "Approximate Entropy", "Skewness", "Kurtosis", "Zero-Crossing Rate"
]
feature_units = [
    "μV²/Hz", "μV²/Hz", "μV²/Hz", "μV²/Hz", "μV²/Hz",
    "Ratio", "Hz", "", "Coeff", "Coeff", "Coeff", "Coeff", "Coeff",
    "", "", "", "Rate"
]
healthy_features = reshaped_features[expanded_labels == 'Healthy']
pd_features = reshaped_features[expanded_labels == 'PD']
rescaled_healthy = healthy_features.copy()
rescaled_pd = pd_features.copy()
scaling_factors = [1e8, 1e8, 1e8, 1e8, 1e8] + [1] * 12
for i in range(17):
    rescaled_healthy[:, :, i] *= scaling_factors[i]
    rescaled_pd[:, :, i] *= scaling_factors[i]
plt.figure(figsize=(16, 9))
for i, feature_name in enumerate(feature_names):
    ax = plt.subplot(3, 6, i + 1)
    sns.boxplot(
        data=[rescaled_healthy[:, :, i].flatten(), rescaled_pd[:, :, i].flatten()],
        palette="Set2",
        ax=ax
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Healthy', 'PD'])
    ax.set_ylabel(feature_units[i])
    ax.legend(
        [feature_name],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        fontsize=8,
        frameon=False
    )
plt.tight_layout(pad=2.0)
plt.savefig("/content/drive/MyDrive/feature_distributions.png", format="png", dpi=300, bbox_inches="tight")
plt.show()



import numpy as np
def DataAugmentation(Data, Labels, n_augmentations=50, noise_scale=0.05, contrast_range=(0.9, 1.1), scale_range=(0.9, 1.1)):
    """
    Performs augmentation on EEG feature data using NumPy operations.
    Parameters:
    - Data: ndarray of shape (samples, n_channels, n_features)
    - Labels: ndarray of shape (samples,)
    - n_augmentations: number of augmentations per original sample
    - noise_scale: std dev for Gaussian noise
    - contrast_range: range for contrast scaling
    - scale_range: range for multiplicative scaling
    Returns:
    - augmented_features: ndarray of shape (~samples * (1 + n_augmentations), ...)
    - augmented_labels: corresponding labels
    """
    augmented_features = []
    augmented_labels = []
    for i in range(len(Data)):
        original = Data[i]
        label = Labels[i]
        augmented_features.append(original)
        augmented_labels.append(label)
        for _ in range(n_augmentations):
            sample = original.copy()
            noise = np.random.normal(0, noise_scale, sample.shape)
            sample += noise
            contrast_factor = np.random.uniform(*contrast_range)
            sample = (sample - np.mean(sample)) * contrast_factor + np.mean(sample)
            scale_factor = np.random.uniform(*scale_range)
            sample *= scale_factor
            augmented_features.append(sample)
            augmented_labels.append(label)
    augmented_features = np.array(augmented_features)
    augmented_labels = np.array(augmented_labels)
    print("Augmented features shape:", augmented_features.shape)
    print("Augmented labels shape:", augmented_labels.shape)
    return augmented_features, augmented_labels
reshaped_features = reshaped_features.astype(np.float32)
augmented_features, augmented_labels = DataAugmentation(reshaped_features, expanded_labels)

from sklearn.preprocessing import StandardScaler
def normalize_features(EEG_augmented_features):
    scaler = StandardScaler()
    features_reshaped = EEG_augmented_features.reshape(EEG_augmented_features.shape[0], -1)
    features_scaled = scaler.fit_transform(features_reshaped)
    features_scaled = features_scaled.reshape(EEG_augmented_features.shape)
    return features_scaled
normalized_features = normalize_features(augmented_features)
print("Normalized features shape:", normalized_features.shape)  # Should match EEG_augmented_features.shape

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded_labels = encoder.fit_transform(augmented_labels.reshape(-1, 1))
normalized_features = np.array(normalized_features)
X_train, X_test, y_train, y_test = train_test_split(
    normalized_features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)