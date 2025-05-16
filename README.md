# CTESM
A Hybrid Convolutional-Transformer Approach for Accurate EEG-Based Parkinson’s Disease Detection

This repository contains the Python implementation of the methodology used in the research article focused on **Parkinson's Disease classification using EEG signals** and a deep learning-based architecture called **Convolutional Transformer Enhanced Sequential Model (CTESM)**.

## Repository Structure
The original Jupyter notebook has been modularized into the following standalone Python scripts for better readability and reproducibility:

- **1_data_loading_and_preprocessing.py**  
  Loads EEG datasets, performs preprocessing including normalization and artifact handling.

- **2_feature_extraction.py**  
  Extracts temporal and spectral features from the EEG signals for downstream modeling.

- **3_model_architecture.py**  
  Defines the CTESM deep learning architecture, integrating convolutional layers with temporal and spatial encoding modules.

- **4_training_loop_and_callbacks.py**  
  Implements the model training logic, loss functions, and callback mechanisms like early stopping and checkpointing.

- **5_evaluation_and_metrics.py**  
  Evaluates the trained model using accuracy, F1-score, confusion matrix, and other classification metrics.

- **6_visualization_and_results.py**  
  Visualizes training history, plots EEG signal patterns, and generates result summaries for interpretation.

# If you find this work useful, please cite our article:

# Publicly available datasets used in the article:
Dataset 1 from
Dataset 2 from

