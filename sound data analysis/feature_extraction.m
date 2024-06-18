%% Feature Extraction
% This code converts audio signals into features.
% These features are saved .mat files for the spectrogram and the STFT (Short-Time Fourier Transform) algorithm.
%
% * feature1 : sig (4ch) to spectrogram (4ch)
% * feature2 : sig (4ch) to sthd (6ch)
%% Simulation data path

% clear, clc, close all
% path0 = "C:/Users/" + getenv('username') +"/Desktop/DCASE2024-Task10-Dataset/simulation/";
% gen_name = "gen_sound_v1/";

%% Real data path
% loc1~6
% train/, val/

clear, clc, close all
path0 = "C:/Users/" + getenv('username') +"/Desktop/DCASE2024-Task10-Dataset/loc1/";
gen_name = "train/";

%% read table
train_name = replace(gen_name, '/' ,'') + ".csv";
train_path = path0 + train_name;
data_info = readtable(train_path);
len_data = height(data_info);

%% create feat folder
mkdir(path0 + replace(gen_name, '/', '_feat1/'))
mkdir(path0 + replace(gen_name, '/', '_feat2/'))

%% Signal parameter

fframe = 2^10;
delay = fframe/4;
num_ch1 = 4;         % spectrogram 4ch
num_ch2 = 6;         % sthd 6ch

% sig to feature
tic
parfor data_nn = 1:len_data     % parfor, parallel for
    %- load audio file -%
    data_info = readtable(train_path);
    filepath = path0 + data_info.path{data_nn};
    [sig, fs] = audioread(filepath);
    sig = sig/(max(abs(sig(:))));
    
    %- feature1 : spectrogram -%
    feat1 = [];
    for ch_nn1 = 1:num_ch1        
        S = stft(sig(:,ch_nn1)+eps, fs, 'Window', hann(fframe,'periodic'), 'OverlapLength', fframe/2, 'FFTLength', fframe, 'FrequencyRange', 'onesided');      
        spectrum = mag2db(abs(S.^2)+eps);
    
        mesh_xlim = 1:fframe/8;
        spectrum = spectrum(mesh_xlim,:);
        
        feat1 = cat(3, feat1, spectrum);
    end
        
    feat1 = standardization(feat1);
    
    feat1_path = replace(replace(filepath, gen_name, replace(gen_name, '/', '_feat1/')), '.flac', '.mat');
    save(feat1_path, "-fromstruct", struct("feat1", feat1));    % only R2024a version

    %- feature2 : sthd -%
    feat2 = [];
    sum_sig = sum_2sig_with_delay(sig, delay);
    
    for ch_nn2 = 1:num_ch2
        sthd = stHD(sum_sig(:,ch_nn2), fs, fframe);

        mesh_xlim = fframe/4-fframe/16+1:fframe/4+fframe/16;
        sthd = sthd(mesh_xlim,:);
        
        feat2 = cat(3, feat2, sthd);
    end
       
    feat2 = standardization(feat2);
    
    feat2_path = replace(replace(filepath, gen_name, replace(gen_name, '/', '_feat2/')), '.flac', '.mat');    
    save(feat2_path, "-fromstruct", struct("feat2", feat2))     % only R2024a version
end
toc

%% standardization function

function data = standardization(data)
    mean_data = mean2(data);
    std_data = std2(data);
    data = (data - mean_data) / std_data;
end