function [ result_struct] = grid_search_task( task_id )
%% grid search task

%% ==== grid search setting ====
% ==== path ====
basepath = '/Users/proteusfan/Documents/MATLAB/Multi_Label_Learning_Model';
save_folder = 'grid_search_type';
chdir(basepath)
addpath(genpath(pwd))

% ==== data loading ====
data_info = dataLoading('emotions');  % input: yeast, emotions, scence

% ==== parameters ====
% ---- activation function ----
para_space.activation_func = true;

% ---- class-specific graph laplacian ----
para_space.L_x_specificLaplacian = false;
para_space.L_x_neighbor_size = 20;
para_space.L_x_kernel_size = 7;
para_space.L_x_normalized = true;

% ---- co-occurrence matrix ----
para_space.L_c_neighbor_size = 0;  % default '0' will use 25% labels as neighbor size.
para_space.L_c_normalized = true;

% ---- initialization ----
para_space.init = [1, 2] ;  % intialize test label: 1.zero, 2.random, 3.allZero, 4.allRandom
para_space.init_amplifier = 0.7;  % amplify scalar for initializing label matrix randomly

% ---- bound setting ----
para_space.bound_CIB_1 = [ false ];  % true for adaptive CIB-1
para_space.bound_CIB_2 = [ false ];  % true for given CIB-2
para_space.bound_ICA = false;
para_space.bound_orth = [ false, true ];  % orthogonality of rows of prediction label matrix

% ---- AMDD & PGD ----
para_space.beta = [1e-2, 1e-1, 1e0, 1e2, 1e4, 1e6];  % weight of graph laplacian(uniform to classes)
para_space.gamma = [1e-2, 1e-1, 1e0, 1e2, 1e4, 1e6];  % weight of co-occurrence matrix
% ADMM
para_space.ADMM_max_iter = [50, 100];
para_space.ADMM_rho = [1e-4, 1e-2, 1e0];
para_space.ADMM_rho_orth = [1e-4, 1e-2, 1e0];
para_space.ADMM_rho_gap = [1, 3];
para_space.ADMM_rho_rate = [5, 10];
% PGD:
para_space.PGD_max_iter = [10, 20];
para_space.PGD_gap_compute = [1, 3];
para_space.PGD_rate_step = 1;
para_space.PGD_alpha_rate = 1;
para_space.PGD_alpha_init = 1e0;


%% create save folder
save_path = fullfile(basepath, 'result_grid_search', save_folder);
save_path_data = fullfile(save_path, 'data');
if ~exist(save_path, 'dir')
    mkdir(save_path_data);
end
% ---- save para_sapce to result/save_folder/data ----
if ~exist(fullfile(save_path_data, 'parameter_space.mat'), 'file')
    save(fullfile(save_path_data, 'parameter_space.mat'), 'para_space');
end

%% ==== options ====
para_candidate = combvec(para_space.activation_func,...
    para_space.L_x_specificLaplacian, para_space.L_x_neighbor_size, para_space.L_x_kernel_size, para_space.L_x_normalized,...
    para_space.L_c_neighbor_size, para_space.L_c_normalized,...
    para_space.init, para_space.init_amplifier,...
    para_space.bound_CIB_1, para_space.bound_CIB_2, para_space.bound_ICA, para_space.bound_orth,...
    para_space.beta, para_space.gamma,...
    para_space.ADMM_max_iter, para_space.ADMM_rho, para_space.ADMM_rho_orth, para_space.ADMM_rho_gap, para_space.ADMM_rho_rate,...
    para_space.PGD_max_iter, para_space.PGD_gap_compute, para_space.PGD_rate_step, para_space.PGD_alpha_rate, para_space.PGD_alpha_init);

para_candidate_vec = para_candidate(:, task_id);

options.activation_func = para_candidate_vec(1);
options.L_x_specificLaplacian = candidate_vec(2);
options.L_x_neighbor_size = candidate_vec(3);
options.L_x_kernel_size = candidate_vec(4);
options.L_x_normalized = candidate_vec(5);
options.L_c_neighbor_size = candidate_vec(6);
options.L_c_normalized = candidate_vec(7);
options.init = candidate_vec(8);
options.init_amplifier = candidate_vec(9);
options.bound_CIB_1 = candidate_vec(10);
options.bound_CIB_2 = candidate_vec(11);
options.bound_ICA = candidate_vec(12);
options.bound_orth = candidate_vec(13);
options.beta = candidate_vec(14);
options.gamma = candidate_vec(15);
options.ADMM_max_iter = candidate_vec(16);
options.ADMM_rho = candidate_vec(17);
options.ADMM_rho_orth = candidate_vec(18);
options.ADMM_rho_gap = candidate_vec(19);
options.ADMM_rho_rate = candidate_vec(20);
options.PGD_max_iter = candidate_vec(21);
options.PGD_gap_compute = candidate_vec(22);
options.PGD_rate_step = candidate_vec(23);
options.PGD_alpha_rate = candidate_vec(24);
options.PGD_alpha_init = candidate_vec(25);
options.data_type = data_info.type;

%% ==== bounds setting ====
% ==== CIB_1 ====
if options.bound_CIB_1
    CIB_1 = adaptiveCIB1(data_info);
else
    CIB_1{1} = 0 * ones(1, data_info.num_sample);
    CIB_1{2} = data_info.num_c * ones(1, data_info.num_sample);
end

% ==== CIB_2 ====
if options.bound_CIB_2
    CIB_2 = CIB2(data_info);
else
    CIB_2{1} = 0 * ones(data_info.num_c, 1);
    CIB_2{2} = data_info.num_sample * ones(data_info.num_c ,1);
end

% ==== ICA bound ====
if options.bound_ICA

end

% ==== record all constrains in options ====
constrain.CIB_1 = CIB_1;
constrain.CIB_2 = CIB_2;
if options.bound_ICA
    constrain.ICA = ICA_bound;
end
options.constrain = constrain;
clear constrain

%% ==== class-specific graph laplacian ====
save_L_x = strcat('L_x_', 'normalized_', num2str(options.L_x_normalized),...
    'neighbor_', num2str(options.L_x_neighbor_size),...
    'kernel_', num2str(options.L_x_kernel_size),...
    'specific', num2str(options.L_x_specificLaplacian), '.mat');
if ~exist(fullfile(save_path_data, save_L_x), 'file')
    [ L_x, W_x ] = graphLaplacian(data_info, options);
    data_info.L_x = L_x;
    data_info.W_x = W_x;
    save(fullfile(save_path_data, save_L_x), 'L_x', 'W_x');
    clear L_x W_x
else
    temp = load(fullfile(save_path_data, save_L_x));
    data_info.L_x = temp.L_x;
    data_info.W_x = temp.W_x;
    clear temp
end

%% ==== co-occurrence matrix ====
save_L_c = strcat('L_c_', 'normalized_', num2str(options.L_c_normalized),...
    'neighbor_', num2str(options.L_c_neighbor_size), '.mat');
if ~exist(fullfile(save_path_data, save_L_c), 'file')
    L_c = co_occurrence(data_info, options);
    data_info.L_c = L_c;
    save(fullfile(save_path_data, save_L_c), 'L_c');
    clear L_c
else
    temp = load(fullfile(save_path_data, save_L_c));
    data_info.L_c = temp.L_c;
    clear temp
end

%% ==== ADMM ====
[Z_admm, Z_cell, obj_Z, obj_L, log] = ADMM(data_info, options);

%% ==== evaluation ====
Z_admm_discrete = sign(Z_admm);
[result_train_admm_dis_vec, result_test_admm_dis_vec] = ...
    evaluation_discrete_multi_label(Z_admm_discrete, data_info.label_train, data_info.label_test);
fprintf('The discrete evaluation results of ADMM are: \n')
disp([result_train_admm_dis_vec, result_test_admm_dis_vec]');
result_cell = [result_train_admm_dis_vec, result_test_admm_dis_vec]';

%% ==== monitor ====
num_cell = length(Z_cell);

%---- Evaluation of Z in each interation and see how it changes ----
eval_train = zeros(4, num_cell);
eval_test = zeros(4, num_cell);
for ii = 1:num_cell
    Z_temp = sign(Z_cell{ii});
    [eval_train_temp, eval_test_temp] = evaluation_discrete_multi_label(Z_temp, data_info.label_train, data_info.label_test);
    eval_train(:,ii) = eval_train_temp(end-3:end);
    eval_test(:,ii) = eval_test_temp(end-3:end);
end

% ---- recording ----
monitor.Z_cell = Z_cell;
monitor.obj_Z = obj_Z;
monitor.obj_L = obj_L;
monitor.log = log;

monitor.eval_train = eval_train;
monitor.eval_test = eval_test;

%% ==== save result ====
result_struct = struct( 'result', result_cell, 'parameter', options,...
    'monitor', monitor);

save_name = strcat('task_', num2str(task_id), '.mat');
save(fullfile(save_path, save_name), 'result_struct');

end

