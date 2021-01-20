clear
close all
chdir('D:\matlab code\Multilabel_propagation\MLML_my_models\MLML-Lovasz\demo-MMIB-AAAI2016-BYWU')
addpath(genpath(pwd))

%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%% Part 1 --- preprocessing, prepare the feature, label and some constant parameter matrices
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%% load the train and test data, 593 = 391 + 202 samples, 6 classes, 72 features

dataset_train=load('emotions_train.txt');
label_train=(dataset_train(:,73:78))';   % 6 x 391  matrix
dataset_train(:,73:78) = [];  % 391 * 72 matrix

dataset_test=load('emotions_test.txt');
label_test=(dataset_test(:,73:78))';   % 6 x 202  matrix
dataset_test(:,73:78) = [];  % 202 * 72 matrix

dataset_matrix=[dataset_train;dataset_test]; % 593 x 72
[num_sample,num_dimension] = size(dataset_matrix);

label_train(label_train==0)=-1;
label_test(label_test==0)=-1;
[num_c, num_sample_train] = size(label_train);
[~, num_sample_test] = size(label_test);

% normalize each feature to [-1,1]
for ii=1:num_dimension
    ma=max(dataset_matrix(:,ii));
    mi=min(dataset_matrix(:,ii));
    range=ma-mi;
    dataset_matrix(:,ii)=2.*((dataset_matrix(:,ii)-mi)./range-0.5);
end
dataset_train = dataset_matrix(1:num_sample_train,:);
dataset_test = dataset_matrix(1+num_sample_train:end,:);

%% statistics of the label matrix 
label_whole = [label_train, label_test]; 
cib_vec_1 = sum(label_whole==1, 1)./num_c; 
cib_vec_2 = sum(label_whole==1, 2)./num_sample; 
[min(cib_vec_1), median(cib_vec_1), max(cib_vec_1), std(cib_vec_1)].*100;
[min(cib_vec_2), median(cib_vec_2), max(cib_vec_2), std(cib_vec_2)].*100;

%% compute V_x and L_x, based on kd-tree
%run('D:\matlab code\vlfeat-0.9.19\toolbox\vl_setup')
num_neighbor_size = 20;  num_kernel_size = 7; batch = 10; 

V_x_kdtree_compute % return V_x and L_x

%% set some common constant matries (based on V_x), return the structure 'constant_matrix' 
part_1_constant_matrices


%+++++++++++++++++++++++++++++++++++++++++++++++++++
%% Part 2 --- the main part of experiments 
%+++++++++++++++++++++++++++++++++++++++++++++++++++

global positive_label_weight_vector positive_label_weight
positive_label_weight = 2; % this value is used to caculate the weighted hamming loss of the final result

unlabel_value=0;  % missing label value
thresh_vector=1;%[0.2:0.2:0.8]; % the proportion of the provided labels, '0.2' means 20% labels are provided, '1' means there is no missing labels in training label matrix

max_iter=1; % in each iteration, we randomly generate different missing labels; when set 'thresh_vector=1',  you should set max_iter=1, because the training label matrix is full, i.e., no random missing labels
len_thresh = length(thresh_vector);

% define the result cell
result_cell_stcut = cell(len_thresh, max_iter);
result_cell_MMIB = cell(len_thresh, max_iter);

for iter_thresh=1:len_thresh
        Thresh=thresh_vector(iter_thresh);
        rate = (num_sample/num_sample_train) * ( 1/Thresh ); 

        for iter=1:max_iter

            % generating the missing labels
            label_train_missing=label_train;
            hide = rand(num_c,num_sample_train)>Thresh;   %The element at the position whose value is larger than Thresh will be deleted
            [m,n] = find( hide);
            for k=1:length(m)
                label_train_missing(m(k), n(k)) = unlabel_value;
            end
            initial_assign_matrix=[label_train_missing, unlabel_value.*ones(num_c,num_sample_test)];
            fprintf('In the full label matrix, #positive is %i, #negative is %i \n', sum(sum(initial_assign_matrix==1)), sum(sum(initial_assign_matrix==-1)) );
            
            positive_sum_vector = sum(label_train_missing==1,2); 
            negative_sum_vector = sum(label_train_missing==-1,2); 
            positive_label_weight_vector = negative_sum_vector./positive_sum_vector; % the ratio between negative and positive labels in each class, it will be used later to determine the weighted loss function
            
            % class-level similarity matrix
            num_neighbor =3;
            [V_c_normalized,V_c, Vc_cell]=Vc_compute(initial_assign_matrix, num_neighbor);
            L_c = eye(num_c, num_c) - V_c_normalized;   
            
           % set some common constant matries (based on V_c), return the structure 'constant_matrix' 
            part_2_constant_matrices
            
    %% Method 1,  call st-cut method, without class cardinality constraints
           lambda_x = 1.3e0;  % i.e., beta in our paper
           lambda_c = 1e-2;  % i.e., gamma in our paper
           st_cut_part
           
           result_cell_stcut{iter} = [result_train_stcut_vec, result_test_stcut_vec]'; 
           
    %% Method 2, call MMIB, with class cardinality constraints
           lambda_x = 1.3e0;  % i.e., beta in our paper
           lambda_c = 1e-2;  % i.e., gamma in our paper    
           
           % CIB1 -- the cardinality bounds of how many classes of each instance
            options.cardinanity.class_lower = 1; % this two values are determined according to the statistics of the training label matrix 
            options.cardinanity.class_upper = 5;    
            
            % ---- if using the following values, then it means CIB1 is not used
            %options.cardinanity.class_lower = 0;
            %options.cardinanity.class_upper = num_c; 
            
           % CIB2 --  the cardinality bounds of how many instances of each class
            options.cardinanity.instance_lower = max(1, positive_sum_vector.*rate.*0.8); 
            options.cardinanity.instance_upper = max(options.cardinanity.instance_lower + 1, min(1.* num_sample, positive_sum_vector.*rate.*2));   
            
            % ---- if using the following values, then it means CIB1 is not used
            %options.cardinanity.instance_lower = 0.*ones(num_c, 1); 
            %options.cardinanity.instance_upper = 1.*num_sample.* ones(num_c, 1);
            
           % set some common constant matries (based on 'options.cardinanity'), return the structure 'constant_matrix' 
            part_3_constant_matrices
            
           % parameters of optimization algorithm
           options.PGD.max_iter = 10; % the maximal iterations of PGD, this value can be set be 1 or a small value
           options.PGD.gap_compute = 1; % call the exact line search to compute the step size in every 'gap_compute' iterations in PGD, if the 'options.PGD.max_iter' is large
           options.PGD.rate_step = 1;  % if not calling the line search, the step size size of current iteration is the product of the one in last iteration and 'options.PGD.rate_step'
           options.PGD.alpha_rate = 1; % this value is to adjust the value computed by exact line search, often >= 1
           
           options.ADMM.max_iter_overall =100; % the maximal iterations of ADMM
           options.ADMM.rho_0 = 1e-2; % the initial rho value
           options.ADMM.rho_gap = 50; % enlarge the rho value once in every 'options.ADMM.rho_gap' iterations
           options.ADMM.rho_rate = 5; % the rate to enlarge the rho value

            
            %% call MMIB Algorithm
            [Z_admm, Z_cell, obj_L, obj_Z, Ux_final, Uc_final] =MMIB(initial_assign_matrix,edgeStructure, lambda_x, lambda_c, constant_matrix, options);

            % evaluation of the predicted discrete label matrix
            Z_admm_discrete = sign(Z_admm); 
            [result_train_admm_dis_vec, result_test_admm_dis_vec, result_train_all, result_test_all] = evaluation_discrete_multi_label(Z_admm_discrete, label_train, label_test);
            result_train_admm_dis_vec = [ lambda_x; lambda_c; result_train_admm_dis_vec];
            result_test_admm_dis_vec = [ lambda_x; lambda_c; result_test_admm_dis_vec];
            fprintf('The discrete evaluation results of ADMM are: \n')
            [result_train_admm_dis_vec, result_test_admm_dis_vec]'
            result_cell_MMIB{iter} = [result_train_admm_dis_vec, result_test_admm_dis_vec]';

%             %% Obvsering the objective difference btween MMIB and st-cut
%             % the following constant is removed in the Lovasz extension, please see Proposition 2 for defition. It will be used to compute the objective difference between MMIB and st-cut   
%             const_weight = full(num_c*lambda_x.*sum( V_x(sub2ind([num_sample, num_sample], node1_x, node2_x)) .* ( dx_sqrt_inver(node1_x) - dx_sqrt_inver(node2_x) ).^2 )...
%                                    + num_sample * lambda_c.* sum( V_c(sub2ind([num_c, num_c], node1_c, node2_c)) .* ( dc_sqrt_inver(node1_c) - dc_sqrt_inver(node2_c) ).^2 ) );
%             fprintf('The objective difference between MMIB and st-cut is %f \n', const_weight + obj_Z(end) - obj_stcut)
        

        end
end


%% The final results 
result_train = []; result_test = [];
for i = 1:max_iter
    result_train = [result_train; result_cell_stcut{i}(1,:)];
    result_test = [result_test; result_cell_stcut{i}(2,:)];
end
result_final_stcut = [];
result_final_stcut(1,:) = mean(result_train);
result_final_stcut(2,:) = std(result_train, 0,1);
result_final_stcut(3,:) = mean(result_test);
result_final_stcut(4,:) = std(result_test, 0,1);
result_final_stcut'


result_train = []; result_test = [];
for i = 1:max_iter
    result_train = [result_train; result_cell_MMIB{i}(1,:)];
    result_test = [result_test; result_cell_MMIB{i}(2,:)];
end
result_final_MMIB = [];
result_final_MMIB(1,:) = mean(result_train);
result_final_MMIB(2,:) = std(result_train, 0,1);
result_final_MMIB(3,:) = mean(result_test);
result_final_MMIB(4,:) = std(result_test, 0,1);
result_final_MMIB'

