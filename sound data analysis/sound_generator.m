%% Simulation Sound Generator
% Each of loc1~6 independently generates sound.
% Use only the middle 10 seconds from a 30-second simulation sound.
% The number of sounds for each category is random: car ranges from 0 to 12, and cv (presumably “commercial vehicle”) ranges from 0 to 3

clear, clc

% Simulation data path
path0 = "C:/Users/" + getenv('username') + "/Desktop/DCASE2024-Task10-Dataset/simulation/";
new_path0 = "gen_sound_v3/";
csv_name = "events-0000-0049.csv";

fs = 16000;

%% create new folder
mkdir(path0+new_path0)

%% generate gen_sound
% “gen_sound” is a one-minute sound data created by synthesizing simulation data
tic
num_gen_sound = 1000;    % The number of sounds generated per loc

loc_names = ["loc1/", "loc2/", "loc3/", "loc4/", "loc5/", "loc6/"];

loc_label_cells = cell(1, length(loc_names));

parfor nn0 = 1:6
    loc_name = loc_names(nn0);
    loc_label_cell = {};

    for sound_nn = 1:num_gen_sound        
        label_names = ["car/left/", "car/right/", "cv/left/", "cv/right/"];   % ["car_l2r", "car_r2l", "cv_l2r", "cv_r2l"];
        
        % Labels. Various the number of car sounds
        labels = rand_labels(num_gen_sound, sound_nn);       

        gen_sig = randn(fs*60, 4)/1000;     %zeros(fs*60, 4);  % 1 minute
        
        for nn1 = 1:4       % repeat for each label (vehicle).
            label_name = label_names(nn1);
            label = labels(nn1);
        
            data_path = path0 + loc_name + label_name + csv_name;
            data_info = readtable(data_path);
            len_data = height(data_info);
            
            rand_data_num = randperm(len_data, label);              % singal selction % ransomly select 1 ~ len_data for each label
            rand_sig_start_num = (randperm(500, label)-1)*fs/10;    % signal start sample % randomly choose 0 ~ (500-1)ms for each label
        
            for nn2 = 1:label
                rand_data_nn = rand_data_num(nn2);        
        
                filepath = path0 + loc_name + label_name + data_info.path{rand_data_nn};
        
                sig = audioread(filepath);
                sig = sig(10*fs+1:20*fs,:);
                len_sig = length(sig);
        
                start_idx = rand_sig_start_num(nn2) + 1;
                end_idx = start_idx + len_sig - 1;
                
                gen_sig(start_idx:end_idx, :) = gen_sig(start_idx:end_idx, :) + sig;                
            end            
        end
        
        gen_sig = gen_sig/(max(abs(gen_sig(:))));

        new_filename = new_path0 + replace(loc_name,'/','_') + sound_nn + ".flac";
        new_filepath = path0 + new_filename;
        audiowrite(new_filepath, gen_sig, fs)   % save gen_sound
        
        loc_label_cell(sound_nn, 1:4) = num2cell(labels);
        loc_label_cell(sound_nn, 5) = {new_filename};                
    end
    loc_label_cells(nn0) = {loc_label_cell};
end
toc

% convert cell to table
label_table = table;
varNames = ["car_left", "car_right", "cv_left", "cv_right", "path"];

for nn0 = 1:6
    temp_table = cell2table(loc_label_cells{nn0}, 'VariableNames', varNames);
    
    label_table = vertcat(label_table, temp_table);
end

new_label_path = path0 + replace(new_path0, '/', '') + ".csv";
writetable(label_table, new_label_path)     % save .csv file

%% plot gen_sig
plot(gen_sig)

%% label generator function
function labels = rand_labels(num_gen_sound, sound_nn)   
    if sound_nn < num_gen_sound*0.4
        labels = [randi(6, 1, 2)-1, randi(3, 1, 2)-1];      % label - car: 0~5, cv: 0~2
    elseif sound_nn < num_gen_sound*0.8
        labels = [randi(11, 1, 2)-1, randi(4, 1, 2)-1];     % label - car: 0~10, cv: 0~3
    else
        car_l2r = randi(21, 1)-1;   % car: 0~20
        cv_l2r = randi(6, 1)-1;     % cv: 0~5
        labels = [car_l2r, 20-car_l2r, cv_l2r, 5-cv_l2r];   % label - car: 0~20, cv: 0~5
    end
end