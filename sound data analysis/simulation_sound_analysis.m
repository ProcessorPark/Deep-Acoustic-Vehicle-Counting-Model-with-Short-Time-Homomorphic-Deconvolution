%% simulation sound Analysis
% The simulation data is 30 seconds long and contains only one sound.

clear, clc, close all

% simulation data path
path0 = "C:/Users/" + getenv('username') +"/Desktop/DCASE2024-Task10-Dataset/simulation/";
loc_name = "loc1/";         % "loc1/", "loc2/", "loc3/", "loc4/", "loc5/", "loc6/"
label_name = "car/left/";   % "car/left/", "car/right/", "cv/left/", "cv/right/";
loc_name = loc_name + label_name;
train_name = "events-0000-0049.csv";

train_path = path0 + loc_name + train_name;
data_info = readtable(train_path);
len_data = height(data_info);

%% select data
data_nn = 25; % 1~50(len_data)  % 10;
filepath = path0 + loc_name + data_info.path{data_nn}

%% load audio file
[sig, fs] = audioread(filepath);
tt = (0:length(sig)-1)/fs;

sig = sig/(max(abs(sig(:))));

%% plot 4ch sound signal
figure("Name", loc_name+"1"), sgtitle("4ch sound data - " + label_name)
for nn = 1:4
    subplot(4,1,nn), plot(tt, sig(:,nn)), grid
end

%% parameter and sum the signal
fframe = 2^10;
delay = fframe/4;
sum_sig = sum_2sig_with_delay(sig, delay);

%% plot STHD mesh
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

figure("Name", loc_name+"2"), sgtitle("STHD - " + label_name)
for nn = 1:6
    sthd = sthds{nn};
    sthd = sthd(mesh_xlim,:);
    
    subplot(3,2,nn),
    mesh(sthd, 'FaceColor', 'flat'), colormap(jet), colorbar, view([0 90]),
    title(sig_name{nn}), xlabel('Time'), ylabel('Time of Flight')%, xlim([1 186])    
end

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
figure("Name", loc_name+"3"), sgtitle("Spectrogram 그림 - " + label_name)
mesh_xlim = 1:fframe/8;

for nn = 1:4    
    subplot(2,2,nn),
    mesh(stfts{nn}(mesh_xlim,:), 'FaceColor', 'flat'), colormap(jet), colorbar, view([0 90]),
    title("ch" + nn), xlabel('Time'), ylabel('Fequency')%, xlim([1 186])
    % drawnow
end

%%
(512/fframe)*fs
(128/fframe)*fs
(54/fframe)*fs
(33/fframe)*fs
(22/fframe)*fs