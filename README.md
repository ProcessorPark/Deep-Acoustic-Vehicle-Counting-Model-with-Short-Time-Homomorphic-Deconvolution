# Deep Acoustic Vehicle Counting Model with Short-Time Homomorphic Deconvolution

acoustic based traffic monitoring system for the [DCASE 2024 Task 10](https://dcase.community/challenge2024/).

This code also accompanies the publication of the DCASE2024-Task10 paper "Deep Acoustic Vehicle Counting Model with Short-Time Homomorphic Deconvolution" by Y.Park

## Sound data Analysis (MATLAB)
The data used for the Deep Acoustic Vehicle Counting Model was analyzed using MATLAB. The following scripts were used:
- feature_extraction.m : Extracts features (such as spectrograms and STHD)
- gen_sound_analysis.m : Analyzes the “gen_sound” data
- real_data_analysis.m : Analyzes real-world data
- simulation_sound_analysis.m :  Analyzes simulation data
- sound_generator.m : Generates sound data by synthesizing simulation sounds to create “gen_sound”

## Deep Learning Model (Python)
The training and evaluation of the Deep Acoustic Vehicle Counting Model were conducted using Python. The following notebooks and modules were used:
- feature_mesh_plot.ipynb : Plots and analyzes features (spectrograms, STHD)
- inference_and_evaluation_model.ipynb : Performs model inference and evaluation
- preprocessing_sig2feat.py : A preprocessing module that computes features (spectrograms, STHD) from raw signals. The functions within this module were originally written in MATLAB and later converted to Python
- train_dual_resnet_gru.ipynb : Contains the training code for the Deep Acoustic Vehicle Counting Model, including data loading and feature computation
- train_dual_resnet_gru(npy input).ipynb : (Not used) Training code for the Deep Acoustic Vehicle Counting Model. It extracts features using MATLAB and saves them to files before training
- transfer_dual_resnet_gru.ipynb : Code for transfer learning with the Deep Acoustic Vehicle Counting Model
- ysp_func.py : Utility functions


## Project Requirements

To run this project, you'll need MATLAB and Python.

### MATLAB
- MATLAB 2024a
- Deep Learning Toolbox
- Statistics and Machine Learning Toolbox
- Signal Processing Toolbox
- DSP System Toolbox
- Audio Toolbox
- Phased Array System Toolbox
- Computer Vision Toolbox
- Image Processing Toolbox
- Parallel Computing Toolbox (optional)

### Python
- numpy
- pandas
- scipy
- librosa
- matplotlib
- tqdm
- torch
- torchvision
- lightning
- einops


## patent
Parts of this code have been submitted for patent approval or are currently in progress.
