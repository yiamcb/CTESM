"""Importing Libraries"""
import pandas as pd
import numpy as np
import scipy
import scipy.io
import os
import zipfile
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import random
from scipy.stats import ttest_ind, f_oneway
import scipy.stats as stats
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Input, Conv2D, Conv1D, MaxPooling1D, BatchNormalization, Dense, MaxPooling2D, Flatten, Dense, Dropout, concatenate, LSTM, Reshape, Concatenate, Activation, Permute, Multiply
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import gc

"""Mounting Google Drive"""
from google.colab import drive
drive.mount('/content/drive', force_remount = True)

!pip install mne

import mne
main_folder = "/content/drive/MyDrive/BIDA Validation/"
all_data = []
all_labels = []
def get_label(folder_name):
    if "sub-hc" in folder_name:
        return "Healthy"
    elif "sub-pd" in folder_name:
        return "PD"
    else:
        return None
for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)
    if os.path.isdir(subfolder_path) and subfolder.startswith("sub-"):
        label = get_label(subfolder)
        if label:
            for sub_subfolder in os.listdir(subfolder_path):
                eeg_folder = os.path.join(subfolder_path, sub_subfolder, "eeg")
                if os.path.isdir(eeg_folder):
                    for file_name in os.listdir(eeg_folder):
                        if file_name.endswith(".bdf"):
                            file_path = os.path.join(eeg_folder, file_name)
                            raw_data = mne.io.read_raw_bdf(file_path, preload=True)
                            data = raw_data.get_data()  # Shape (n_channels, n_samples)
                            all_data.append(data)
                            all_labels.append(label)
                            print(f"Loaded: {file_path}, Label: {label}, Data Shape: {data.shape}")

min_sample_length = min(data.shape[1] for data in all_data)
homogenized_data = np.array([data[:, :min_sample_length] for data in all_data])
print("All data shape after homogenization:", homogenized_data.shape)

eeg_data = np.array(homogenized_data)
all_labels = np.array(all_labels)
print("All data shape:", eeg_data.shape)
print("All labels:", all_labels)