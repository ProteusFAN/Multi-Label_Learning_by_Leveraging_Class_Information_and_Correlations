%% batch
clear
close all
base_path = '/Users/Proteus/Documents/MATLAB/Internship/Experiments';
chdir(base_path)
addpath(genpath(pwd))

for ii = 9:15
    name = strcat('ica_bounds_',num2str(ii),'.mat');
    load(fullfile(base_path, 'data', name));
    ica_bounds_manual
end