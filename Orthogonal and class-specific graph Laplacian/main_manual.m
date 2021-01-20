%% Multi-label learning model
clear
close all

%% ==== configuration ====
configuration

%% ==== Main algorithm ====
% ==== class-specific graph laplacian ==== 
[ data_info.L_x, data_info.W_x ] = graphLaplacian(data_info, options);

% ==== co-occurrence matrix ====
data_info.L_c = co_occurrence(data_info, options);

% ==== ADMM ====
[Z_admm, Z_cell, obj_Z, obj_L, log] = ADMM(data_info, options);

% ==== evaluation ====
evaluation

% ==== monitor ====
monitor_process

%% ==== save result ====
result_struct = struct( 'result', result_cell, 'parameter', options,...
    'monitor', monitor);

save_path = fullfile(path, 'result');
if ~exist(save_path, 'dir')
    mkdir(save_path)
end

save_name = strcat(generate_clock_str,'_Fscore_',...
    num2str(result_test_admm_dis_vec(end-3)), '_',...
    num2str(result_test_admm_dis_vec(end-2)), '_',...
    num2str(result_test_admm_dis_vec(end-1)), '_',...
    num2str(result_test_admm_dis_vec(end)), '.mat');
save(fullfile(save_path, save_name), 'result_struct');