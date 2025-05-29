# CTESM
**A Hybrid Convolutional-Transformer Approach for Accurate EEG-Based Parkinson’s Disease Detection**

This repository contains the Python implementation of the methodology described in our research article on **EEG-based classification of Parkinson’s Disease (PD)** using a deep learning architecture called the **Convolutional Transformer Enhanced Sequential Model (CTESM)**.

## Overview
CTESM integrates convolutional neural networks (CNN), transformer blocks, and long short-term memory (LSTM) layers to capture spatial, temporal, and sequential patterns in EEG data for robust and accurate PD classification.

## Repository Structure
The original Jupyter notebook has been modularized into the following standalone Python scripts to enhance readability and reproducibility:

- **1_data_loading_and_preprocessing.py**  
  Loads EEG datasets and performs preprocessing steps, including normalization and artifact handling.

- **2_feature_extraction.py**  
  Extracts biologically informed features such as spectral power, band ratios, wavelet coefficients, and statistical measures from EEG signals.

- **3_model_architecture.py**  
  Defines the CTESM model architecture, integrating CNN layers with transformer-based attention mechanisms and LSTM layers.

- **4_training_loop_and_callbacks.py**  
  Implements the training loop, loss functions, and callback mechanisms such as early stopping and model checkpointing.

- **5_evaluation_and_metrics.py**  
  Evaluates the trained model using metrics including accuracy, F1-score, precision, recall, and confusion matrix.

- **6_visualization_and_results.py**  
  Visualizes training history, plots key EEG features, and generates result summaries for interpretation.

## Citation
If you find this work useful, please cite our article:

Bunterngchit, C., Baniata, L. H., Albayati, H., Baniata, M. H., Alharbi, K., Alshammari, F. H., & Kang, S. (2025). A Hybrid Convolutional–Transformer Approach for Accurate Electroencephalography (EEG)-Based Parkinson’s Disease Detection. *Bioengineering*, 12(6), 583. https://doi.org/10.3390/bioengineering12060583

## Datasets
The following publicly available datasets were used in this study:

- **Dataset 1**: UC San Diego Resting State EEG Data from Patients with Parkinson's Disease from Rockhill et al.  
  [DOI: 10.18112/openneuro.ds002778.v1.0.5](https://doi.org/10.18112/openneuro.ds002778.v1.0.5)

- **Dataset 2**: University of Iowa PD EEG Dataset from Anjum et al.  
  [DOI: 10.1016/j.parkreldis.2020.08.001](https://doi.org/10.1016/j.parkreldis.2020.08.001)
---

