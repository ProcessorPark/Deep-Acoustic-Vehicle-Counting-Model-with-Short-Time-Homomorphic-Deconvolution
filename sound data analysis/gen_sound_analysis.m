%% gen sound Analysis
% “gen_sound” is a one-minute sound data created by synthesizing simulation data

clear, clc, close all

% simulation data path
path0 = "C:/Users/" + getenv('username') +"/Desktop/DCASE2024-Task10-Dataset/simulation/";
gen_name = "gen_sound_v3/";
train_name = replace(gen_name, '/' ,'') + ".csv";

train_path = path0 + train_name;
data_info = readtable(train_path);
len_data = height(data_info);

%% select data
data_nn = 1; % 1~600(len_data)
filepath = path0 + data_info.path{data_nn}

%% load audio file
[sig, fs] = audioread(filepath);
tt = (0:length(sig)-1)/fs;

sig = sig/(max(abs(sig(:))));

%% plot 4ch sound signal
figure("Name", gen_name+"1"), sgtitle("4ch sound data - " + data_nn)
for nn = 1:4
    subplot(4,1,nn), plot(tt, sig(:,nn)), grid
end

%% parameter and sum the signal
fframe = 2^10;
delay = fframe/4;
sum_sig = sum_2sig_with_delay(sig, delay);

%% calculate the STHD (Short-Time Homomorphic Deconvolution)
sthds = {};

tic
parfor nn = 1:6
    sthds{nn} = stHD(sum_sig(:,nn), fs, fframe);
end
toc
size(sthds{1})

%% plot STHD mesh
sig_name = {'sig1 + sig2', 'sig1 + sig3', 'sig1 + sig4', 'sig2 + sig3', 'sig2 + sig4', 'sig3 + sig4'};
mesh_xlim = fframe/4-fframe/16+1:fframe/4+fframe/16;    % mesh_xlim = 1:fframe/2;

figure("Name", gen_name+"2"), sgtitle("STHD - " + data_nn)
for nn = 1:6
    sthd = sthds{nn};
    sthd = sthd(mesh_xlim,:);
    
    subplot(3,2,nn),
    mesh(sthd, 'FaceColor', 'flat'), colormap(jet), colorbar, view([0 90]),
    title(sig_name{nn}), xlabel('Time'), ylabel('Time of Flight'), xlim([1 1874]), ylim([1 128])    
end

%% check label of the data
label_name = ["car_l2r", "car_r2l", "cv_l2r", "cv_r2l"];
label = [data_info.car_left(data_nn) data_info.car_right(data_nn) data_info.cv_left(data_nn) data_info.cv_right(data_nn)]

%% calculate the Spectrogram
% calculate using STFT and convert magnitude to decibels (mag2db)
stfts = {};

tic
parfor nn = 1:4
     S = stft(sig(:,nn)+eps, fs, 'Window', hann(fframe,'periodic'), 'OverlapLength', fframe/2, 'FFTLength', fframe, 'FrequencyRange', 'onesided');      
     stfts{nn} = mag2db(abs(S.^2)+eps);
end
toc

%% Plot Spectrogram
figure("Name", gen_name+"3"), sgtitle("Spectrogram - " + data_nn)
mesh_xlim = 1:fframe/8;

for nn = 1:4    
    subplot(2,2,nn),
    mesh(stfts{nn}(mesh_xlim,:), 'FaceColor', 'flat'), colormap(jet), colorbar, view([0 90]),
    title("ch" + nn), xlabel('Time'), ylabel('Fequency'), xlim([1 1874]), ylim([1 128])    
end