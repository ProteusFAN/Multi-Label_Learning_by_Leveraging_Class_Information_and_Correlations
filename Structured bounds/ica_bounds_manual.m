clear
close all
chdir('/Users/Proteus/Documents/MATLAB/Internship/Experiments')
addpath(genpath(pwd))

%% Parameters setting

% ==== Randomness of generating Z ====
% options.initialization = 'zero';
options.initialization = 'random';

% ==== Orthogonality (rows of Z) ====
options.orth = true;

% ==== ICA_bounds ====
load('ICA_bounds.mat');
options.ICA_valid = false;

% ==== CIB ====
% load('CIB.mat');  % consists of CIB_1 and CIB_2

% CIB-1
CIB_1{1} = 0 * ones(1,2417);
CIB_1{2} = 14 * ones(1,2417);
% CIB_1{1} = CIB_1_bounds(:,1);  % use CIB_1 bounds itself
% CIB_1{2} = CIB_1_bounds(:,2);

% CIB-2
CIB_2{1} = 0 * ones(14, 1);
CIB_2{2} = 2417 * ones(14 ,1);
% CIB_2{1}(end-2:end-1) = cubic_bounds(end-2:end-1,1);
% CIB_2{2}(end-2:end-1) = cubic_bounds(end-2:end-1,2);

% CIB_2{1} = cubic_bounds(:,1);  % use cubic_bounds of ICA_bounds
% CIB_2{2} = cubic_bounds(:,2);
% CIB_2{1} = center;  % Baoyuan's paper
% CIB_2{2} = min(2*center, 0.8*2417);
% CIB_2{1} = CIB_2_bounds(:,1);  % use CIB_2 bounds itself
% CIB_2{2} = CIB_2_bounds(:,2);

% ==== AMDD & PGD ====
% L_x
L_x_num_neighbor = 20;
L_x_num_kernel = 7;

% L_c
L_c_num_neighbor = 7;

% MMIB
MMIB_lambda_x =  1e5; % 0.5
MMIB_lambda_c =  0; % 1e-3

% PGD:
PGD_max_iter = 10; % 20
PGD_gap_compute = 1;
PGD_rate_step = 1;
PGD_alpha_rate = 5;  % 1

% ADMM
ADMM_max_iter = 100; % 100
ADMM_rho = 1e-3;  % 1e-2
ADMM_rho_orth = 1e-3;
ADMM_rho_gap = 10;  % 50
ADMM_rho_rate = 5;  % 5

% ==== recording ====
% CIB and ICA
CIB = struct('CIB_1', CIB_1, 'CIB_2', CIB_2);
ICA = struct('ICA_width', ICA_width, 'center', center);
ICA.ICA_bounds = ICA_bounds;
ICA.P_temp = P_temp;

% parameterStruct
parameterStruct = struct( 'L_x_num_neighbor', L_x_num_neighbor,...
    'L_x_num_kernel', L_x_num_kernel, 'L_c_num_neighbor', L_c_num_neighbor,...
    'MMIB_lambda_x', MMIB_lambda_x, 'MMIB_lambda_c', MMIB_lambda_c,...
    'PGD_max_iter', PGD_max_iter, 'PGD_gap_compute', PGD_gap_compute,...
    'PGD_rate_step', PGD_rate_step, 'PGD_alpha_rate', PGD_alpha_rate,...
    'ADMM_max_iter', ADMM_max_iter, 'ADMM_rho', ADMM_rho,...
    'ADMM_rho_orth', ADMM_rho_orth, 'ADMM_rho_gap', ADMM_rho_gap,...
    'ADMM_rho_rate', ADMM_rho_rate,...
    'ICA_valid', options.ICA_valid,...
    'Z_initialization', options.initialization,...
    'orth_row', options.orth);
parameterStruct.ICA = ICA;
parameterStruct.CIB = CIB; 

% ==== display parameter ====
fprintf(['Z_initialization: ',options.initialization, '.\n']);
fprintf('ICA_valid = %d.\n', options.ICA_valid);
fprintf('MMIB_lambda_x = %f.\n', MMIB_lambda_x);
fprintf('MMIB_lambda_c = %f.\n', MMIB_lambda_c);
fprintf('PGD_max_iter = %f.\n', PGD_max_iter);
fprintf('PGD_gap_compute = %f.\n', PGD_gap_compute);
fprintf('PGD_rate_step = %f.\n', PGD_rate_step);
fprintf('PGD_alpha_rate = %f.\n', PGD_alpha_rate);
fprintf('ADMM_max_iter = %f.\n', ADMM_max_iter);
fprintf('ADMM_rho = %f.\n', ADMM_rho);
fprintf('ADMM_rho_orth = %f.\n', ADMM_rho_orth);
fprintf('ADMM_rho_gap = %f.\n', ADMM_rho_gap);
fprintf('ADMM_rho_rate = %f.\n', ADMM_rho_rate);

%% ========================================================================
%  Part 1 --- preprocessing, prepare the feature, label and some constant parameter matrices
%  ========================================================================

%% load the train and test data, 2417 = 1500 + 917 samples, 14 classes, 103 features

dataset_train=load('yeast_train.txt');
label_train=(dataset_train(:,end-13:end))';  % 14 x 1500  matrix
dataset_train(:,end-13:end) = [];  % 1500 * 103 matrix

dataset_test=load('yeast_test.txt');
label_test=(dataset_test(:,end-13:end))';  % 14 x 917  matrix
dataset_test(:,end-13:end) = [];  % 917 * 103 matrix

dataset_matrix=[dataset_train;dataset_test];  % 2417 * 103 matrix
[num_sample,num_dimension] = size(dataset_matrix);

label_train(label_train==0)=-1;
label_test(label_test==0)=-1;
[num_c, num_sample_train] = size(label_train);
[~, num_sample_test] = size(label_test);

% normalize each feature to [-1,1]
for ii = 1:num_dimension
    ma = max(dataset_matrix(:,ii));
    mi = min(dataset_matrix(:,ii));
    range = ma - mi;
    dataset_matrix(:,ii) = 2.*((dataset_matrix(:,ii) - mi)./range-0.5);
end
dataset_train = dataset_matrix(1:num_sample_train, :);
dataset_test = dataset_matrix(1+num_sample_train:end, :);

%% statistics of the label matrix
% label_whole = [label_train, label_test]; 
% cib_vec_1 = sum(label_whole==1, 1)./num_c; 
% cib_vec_2 = sum(label_whole==1, 2)./num_sample; 
% [min(cib_vec_1), median(cib_vec_1), max(cib_vec_1), std(cib_vec_1)].*100;
% [min(cib_vec_2), median(cib_vec_2), max(cib_vec_2), std(cib_vec_2)].*100;

%% compute V_x and L_x, based on kd-tree
%run('D:\matlab code\vlfeat-0.9.19\toolbox\vl_setup')
num_neighbor_size = L_x_num_neighbor;
num_kernel_size = L_x_num_kernel;
batch = 10;

V_x_kdtree_compute % return V_x and L_x

%% set some common constant matries (based on V_x), return the structure 'constant_matrix'
part_1_constant_matrices

%% ========================================================================
%  Part 2 --- the main part of experiments
%  ========================================================================

global positive_label_weight_vector positive_label_weight
positive_label_weight = 2; % this value is used to caculate the weighted hamming loss of the final result

unlabel_value=0;  % missing label value
thresh_vector=1;  %[0.2:0.2:0.8];  % the proportion of the provided labels, '0.2' means 20% labels are provided, '1' means there is no missing labels in training label matrix

max_iter=1; % in each iteration, we randomly generate different missing labels; when set 'thresh_vector=1',  you should set max_iter=1, because the training label matrix is full, i.e., no random missing labels
len_thresh = length(thresh_vector);

% define the result cell
%result_cell_stcut = cell(len_thresh, max_iter);
result_cell_MMIB = cell(len_thresh, max_iter);

for iter_thresh=1:len_thresh
    Thresh=thresh_vector(iter_thresh);
    rate = (num_sample/num_sample_train) * ( 1/Thresh ); 

    for iter=1:max_iter

        % generating the missing labels
        label_train_missing=label_train;
        hide = rand(num_c,num_sample_train)>Thresh;   %The element at the position whose value is larger than Thresh will be deleted
        [m,n] = find(hide);
        for k=1:length(m)
            label_train_missing(m(k), n(k)) = unlabel_value;
        end
        initial_assign_matrix=[label_train_missing, unlabel_value.*ones(num_c,num_sample_test)];
%         fprintf('In the full label matrix, #positive is %i, #negative is %i \n', sum(sum(initial_assign_matrix==1)), sum(sum(initial_assign_matrix==-1)) );

        positive_sum_vector = sum(label_train_missing==1,2); 
        negative_sum_vector = sum(label_train_missing==-1,2); 
        positive_label_weight_vector = negative_sum_vector./positive_sum_vector; % the ratio between negative and positive labels in each class, it will be used later to determine the weighted loss function

        % class-level similarity matrix
        num_neighbor = L_c_num_neighbor;
        [V_c_normalized, V_c, ~, ~] = Vc_compute(initial_assign_matrix, num_neighbor);
        L_c = eye(num_c, num_c) - V_c_normalized;

        % set some common constant matries (based on V_c), return the structure 'constant_matrix' 
        part_2_constant_matrices
            
        %% Method 1,  call st-cut method, without class cardinality constraints
%        lambda_x = 1.3e0;  % i.e., beta in our paper
%        lambda_c = 1e-2;  % i.e., gamma in our paper
%        st_cut_part
% 
%        result_cell_stcut{iter} = [result_train_stcut_vec, result_test_stcut_vec]'; 
           
        %% Method 2, call MMIB, with class cardinality constraints
        lambda_x = MMIB_lambda_x;  % i.e., beta in our paper
        lambda_c = MMIB_lambda_c;  % i.e., gamma in our paper    
           
        % CIB1 -- the cardinality bounds of how many classes of each instance
        options.cardinanity.class_lower = CIB_1{1}; % this two values are determined according to the statistics of the training label matrix 
        options.cardinanity.class_upper = CIB_1{2};    
        
        % CIB2 --  the cardinality bounds of how many instances of each class
        options.cardinanity.instance_lower = CIB_2{1}; 
        options.cardinanity.instance_upper = CIB_2{2};
        
        % set some common constant matries (based on 'options.cardinanity'), return the structure 'constant_matrix' 
        part_3_constant_matrices

        % set some common constant matries (based on 'ICA_constrains.mat'), return the structure 'constant_matrix' 
        part_4_constant_matrices
        
        % set some common constant matries (based on L_x and orthogonality), return the structure 'constant_matrix' 
        part_5_constant_matrices
        
        % parameters of optimization algorithm
        options.PGD.max_iter = PGD_max_iter; % the maximal iterations of PGD, this value can be set be 1 or a small value
        options.PGD.gap_compute = PGD_gap_compute; % call the exact line search to compute the step size in every 'gap_compute' iterations in PGD, if the 'options.PGD.max_iter' is large
        options.PGD.rate_step = PGD_rate_step;  % if not calling the line search, the step size size of current iteration is the product of the one in last iteration and 'options.PGD.rate_step'
        options.PGD.alpha_rate = PGD_alpha_rate; % this value is to adjust the value computed by exact line search, often >= 1

        options.ADMM.max_iter_overall = ADMM_max_iter; % the maximal iterations of ADMM
        options.ADMM.rho_0 = ADMM_rho; % the initial rho value
        options.ADMM.rho_orth = ADMM_rho_orth;
        options.ADMM.rho_gap = ADMM_rho_gap; % enlarge the rho value once in every 'options.ADMM.rho_gap' iterations
        options.ADMM.rho_rate = ADMM_rho_rate; % the rate to enlarge the rho value
            
        %% call MMIB Algorithm
        [Z_admm, Z_cell, obj_L, obj_Z, Ux_final, Uc_final, log] = MMIB(initial_assign_matrix,edgeStructure, lambda_x, lambda_c, constant_matrix, options);

        % evaluation of the predicted discrete label matrix
        Z_admm_discrete = sign(Z_admm); 
        [result_train_admm_dis_vec, result_test_admm_dis_vec, result_train_all, result_test_all] = evaluation_discrete_multi_label(Z_admm_discrete, label_train, label_test);
        fprintf('The discrete evaluation results of ADMM are: \n')
        disp([result_train_admm_dis_vec, result_test_admm_dis_vec]');
        result_cell_MMIB{iter} = [result_train_admm_dis_vec, result_test_admm_dis_vec]';

%         %% Obvsering the objective difference btween MMIB and st-cut
%         % the following constant is removed in the Lovasz extension, please see Proposition 2 for defition. It will be used to compute the objective difference between MMIB and st-cut   
%         const_weight = full(num_c*lambda_x.*sum( V_x(sub2ind([num_sample, num_sample], node1_x, node2_x)) .* ( dx_sqrt_inver(node1_x) - dx_sqrt_inver(node2_x) ).^2 )...
%                                + num_sample * lambda_c.* sum( V_c(sub2ind([num_c, num_c], node1_c, node2_c)) .* ( dc_sqrt_inver(node1_c) - dc_sqrt_inver(node2_c) ).^2 ) );
%         fprintf('The objective difference between MMIB and st-cut is %f \n', const_weight + obj_Z(end) - obj_stcut)

    end
end

%% monitor process

monitor_process % return monitor

%% The final results
% result_train = []; result_test = [];
% for i = 1:max_iter
%     result_train = [result_train; result_cell_stcut{i}(1,:)];
%     result_test = [result_test; result_cell_stcut{i}(2,:)];
% end
% result_final_stcut = [];
% result_final_stcut(1,:) = mean(result_train);
% result_final_stcut(2,:) = std(result_train, 0,1);
% result_final_stcut(3,:) = mean(result_test);
% result_final_stcut(4,:) = std(result_test, 0,1);
% result_final_stcut'

% result_train = []; result_test = [];
% for i = 1:max_iter
%     result_train = [result_train; result_cell_MMIB{i}(1,:)];
%     result_test = [result_test; result_cell_MMIB{i}(2,:)];
% end
% result_final_MMIB = [];
% result_final_MMIB(1,:) = mean(result_train,1);
% result_final_MMIB(2,:) = mean(result_test,1);
% disp(result_final_MMIB');

%% save result

result_struct = struct( 'result', result_cell_MMIB, 'parameter', parameterStruct,...
    'monitor', monitor);

base_path = '/Users/Proteus/Documents/MATLAB/Internship/Experiments';
save_path = fullfile(base_path, 'result','single_strike');
if ~exist(save_path, 'dir')
    mkdir(save_path)
end

save_name = strcat(generate_clock_str,'_Fscore_',...
    num2str(result_test_admm_dis_vec(end-3)), '_',...
    num2str(result_test_admm_dis_vec(end-2)), '_',...
    num2str(result_test_admm_dis_vec(end-1)), '_',...
    num2str(result_test_admm_dis_vec(end)), '.mat');
save(fullfile(save_path, save_name), 'result_struct');

