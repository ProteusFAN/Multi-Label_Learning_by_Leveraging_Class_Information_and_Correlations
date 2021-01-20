%%  configuration

%% ==== path ====
path = '/Users/proteusfan/Dropbox/Github/Summer-Internship-at-AI-Lab/Multi_Label_Learning_Model';
chdir(path)
addpath(genpath(pwd))

%% ==== data loading ====
data_info = dataLoading('yeast');
% data_info = dataLoading('emotions');
% data_info = dataLoading('scence');

%% ==== options & parameters ====
% option will be used to store parameters in result files.

% ==== activation function ====
options.activation_func = true;

% ==== class-specific graph laplacian ====
options.L_x_specificLaplacian = false;
options.L_x_neighbor_size = 20;
options.L_x_kernel_size = 7;
options.L_x_normalized = true;

% ==== co-occurrence matrix ====
options.L_c_neighbor_size = 0;  % default '0' will use 25% labels as neighbor size.
options.L_c_normalized = true;

% ==== initialization ====
options.data_type = data_info.type;
options.init = 1;  % intialize test label: 1.zero, 2.random, 3.allZero, 4.allRandom
options.init_amplifier = 0.7;  % amplify scalar for initializing label matrix randomly

% ==== bound setting ====
options.bound_CIB_1 = true;  % true for adaptive CIB-1
options.bound_CIB_2 = true;  % true for given CIB-2
options.bound_ICA = false;
options.bound_orth = true;  % orthogonality of rows of prediction label matrix

% ==== AMDD & PGD ====
options.beta = 0.1;  % weight of graph laplacian(uniform to classes)
options.gamma = 0.1;  % weight of co-occurrence matrix
% ADMM
options.ADMM_max_iter = 100;
options.ADMM_rho = 1e-4;
options.ADMM_rho_orth = 1e-3;
options.ADMM_rho_gap = 3;
options.ADMM_rho_rate = 10;
% PGD:
options.PGD_max_iter = 20;
options.PGD_gap_compute = 1;
options.PGD_rate_step = 1;
options.PGD_alpha_rate = 1;
options.PGD_alpha_init = 1e0;

%% ==== bound setting ====
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