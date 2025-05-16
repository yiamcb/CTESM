eeg_means = np.mean(eeg_data, axis=2)  # Resulting shape: (46, 40)
healthy_data = eeg_means[np.array(all_labels) == 'Healthy']
pd_data = eeg_means[np.array(all_labels) == 'PD']
ttest_pvalues = []
anova_pvalues = []
for ch in range(eeg_means.shape[1]):
    healthy_channel_data = healthy_data[:, ch]
    pd_channel_data = pd_data[:, ch]
    t_stat, t_pvalue = ttest_ind(healthy_channel_data, pd_channel_data, equal_var=False)
    ttest_pvalues.append(t_pvalue)
    anova_stat, anova_pvalue = f_oneway(healthy_channel_data, pd_channel_data)
    anova_pvalues.append(anova_pvalue)
results_df = pd.DataFrame({
    'Channel': [f'Channel {i+1}' for i in range(eeg_means.shape[1])],
    'T-test p-value': ttest_pvalues,
    'ANOVA p-value': anova_pvalues
})
print("Statistical Analysis of PD vs Healthy EEG Data")
print(results_df)

for idx, ch_name in enumerate(raw_data.ch_names):
    print(f"Index {idx}: {ch_name} (Type: {raw_data.get_channel_types(picks=[idx])[0]})")

eeg_data = eeg_data[:, :-1, :]

np.shape(eeg_data)

sampling_rate = 512  # Hz
frame_length_seconds = 2  # in seconds
frame_length_samples = frame_length_seconds * sampling_rate  # 1024 samples per frame
overlap_percentage = 0.5  # 50% overlap
overlap_samples = int(frame_length_samples * overlap_percentage)
all_sample_frames = []
for sample in eeg_data:
    sample_frames = []
    for start in range(0, sample.shape[1] - frame_length_samples + 1, frame_length_samples - overlap_samples):
        end = start + frame_length_samples
        sample_frames.append(sample[:, start:end])
    all_sample_frames.append(np.array(sample_frames))
all_sample_frames = np.array(all_sample_frames)
print("Framed data shape:", all_sample_frames.shape)  # Expected shape: (samples, num_frames, num_channels, 1024)

pip install PyWavelets