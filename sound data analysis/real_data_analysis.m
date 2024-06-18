%% real data Analysis
% The real data is 1 minute long and contains many sounds.

clear, clc, close all

% Train set path
path0 = "C:/Users/" + getenv('username') +"/Desktop/DCASE2024-Task10-Dataset/";
loc_name = "loc1/";     % "loc1/", "loc2/", "loc3/", "loc4/", "loc5/", "loc6/"
train_name = "train.csv";

train_path = path0 + loc_name + train_name;
data_info = readtable(train_path);
len_data = height(data_info);

%%  select data
data_nn = 1; % 20;
label_name = ["car_l2r", "car_r2l", "cv_l2r", "cv_r2l"];
label = [data_info.car_left(data_nn) data_info.car_right(data_nn) data_info.cv_left(data_nn) data_info.cv_right(data_nn)]
filepath = path0 + loc_name + data_info.path{data_nn}

%% load audio file
[sig, fs] = audioread(filepath);
tt = (0:length(sig)-1)/fs;

sig = sig/(max(abs(sig(:))));

%% plot 4ch sound signal
disp("Total Vehicle count : " + sum(label));
disp(label_name + " = " + label);
sig_name = {'sig1', 'sig2', 'sig3', 'sig4'};
figure(1), sgtitle(filepath + " 4ch sound data")
for nn = 1:4
    subplot(2,2,nn), plot(tt, sig(:,nn)), grid
    title(sig_name{nn}), xlabel('Time (sec)'), ylabel('Magnitude')
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

figure('Name', "STHD" + nn),
for nn = 1:6
    sthd = sthds{nn};
    sthd = sthd(mesh_xlim,:);

    subplot(3,2,nn),
    mesh(sthd, 'FaceColor', 'flat'), colormap(jet), colorbar, view([0 90]),
    title(sig_name{nn}), xlabel('Time'), ylabel('Time of Flight')
    xlim([1 1874]), ylim([1 128])   
end

disp("Total Vehicle count : " + sum(label));
disp(label_name + " = " + label);
%% calculate the Spectrogram
% calculate using STFT and convert magnitude to decibels (mag2db)

stfts = {};
tic
parfor nn = 1:4
     S = stft(sig(:,nn)+eps, fs, 'Window', hann(fframe,'periodic'), 'OverlapLength', fframe/2, 'FFTLength', fframe, 'FrequencyRange', 'onesided');      
     stfts{nn} = mag2db(abs(S)+eps);
end
toc

%% Plot Spectrogram
sig_name = {'sig1', 'sig2', 'sig3', 'sig4'};

figure("Name", loc_name+"3"), %sgtitle("Spectrogram - " + label_name)
mesh_xlim = 1:fframe/8;

for nn = 1:4
    subplot(2,2,nn),
    mesh(stfts{nn}(mesh_xlim,:), 'FaceColor', 'flat'), colormap(jet), colorbar, view([0 90]),
    title(sig_name{nn}), xlabel('Time'), ylabel('Frequency')
    xlim([1 1874]), ylim([1 128])
end