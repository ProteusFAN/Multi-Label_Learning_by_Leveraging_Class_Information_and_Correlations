function [train_result_vector, test_result_vector, train_result, test_result] = evaluation_discrete_multi_label(label_prediction, label_train, label_test, topk)
% label_prediction, num_c x num_sample matrix
% label_train, num_c x num_sample_train matrix
% label_test, num_c x num_sample_test matrix

if nargin == 4
   options.topk = topk;
elseif nargin ==3
   options.topk = 5;
else
    fprintf('Error: the number of input varaibles is incorrect..\n')
end

num_sample_train = size(label_train,2);
label_prediction_train = label_prediction(:,1:num_sample_train); % num_c x num_sample_train
label_prediction_test = label_prediction(:,1+num_sample_train:end); % num_c x num_sample_train

%% evaluation on training data 

options.type= 'example-based-partition'; % for discrete labels
result_struct_train_3 = multi_label_evaluation(label_prediction_train,label_train, options);

options.type= 'label-based-partition'; % for discrete labels
result_struct_train_4 = multi_label_evaluation(label_prediction_train,label_train, options);

%% evaluation on testing data 

options.type= 'example-based-partition'; % for discrete labels
result_struct_test_3 = multi_label_evaluation(label_prediction_test,label_test, options);

options.type= 'label-based-partition'; % for discrete labels
result_struct_test_4 = multi_label_evaluation(label_prediction_test,label_test, options);

%% summarization 
train_result.accuracy = result_struct_train_3.topk_example_accuracy;
train_result.precision = result_struct_train_3.topk_example_precision;
train_result.recall = result_struct_train_3.topk_example_recall;
train_result.F1 = result_struct_train_3.topk_example_F1;
train_result.F1_macro = result_struct_train_4.topk_label_F1_macro_average;
train_result.F1_micro = result_struct_train_4.topk_label_F1_micro;
train_result.HammingLoss = result_struct_train_3.topk_example_hamming_loss;
train_result.HammingLoss_weighted = result_struct_train_3.topk_example_hamming_loss_weighted;
train_result.F_average = mean([train_result.F1, train_result.F1_macro, train_result.F1_micro]);
train_result.F1_macro_vector = result_struct_train_4.topk_label_F1_macro_vector;

train_result_vector = [train_result.HammingLoss; train_result.HammingLoss_weighted; train_result.accuracy; train_result.precision; train_result.recall; train_result.F1; train_result.F1_macro; 
                                     train_result.F1_micro; train_result.F_average]; 

                                 
test_result.accuracy = result_struct_test_3.topk_example_accuracy;
test_result.precision = result_struct_test_3.topk_example_precision;
test_result.recall = result_struct_test_3.topk_example_recall;
test_result.F1 = result_struct_test_3.topk_example_F1;
test_result.F1_macro = result_struct_test_4.topk_label_F1_macro_average;
test_result.F1_micro = result_struct_test_4.topk_label_F1_micro;
test_result.HammingLoss = result_struct_test_3.topk_example_hamming_loss;
test_result.HammingLoss_weighted = result_struct_test_3.topk_example_hamming_loss_weighted;
test_result.F_average = mean([test_result.F1, test_result.F1_macro, test_result.F1_micro]); 
test_result.F1_macro_vector = result_struct_test_4.topk_label_F1_macro_vector;

test_result_vector = [test_result.HammingLoss; test_result.HammingLoss_weighted; test_result.accuracy; test_result.precision; test_result.recall; 
                               test_result.F1; test_result.F1_macro; test_result.F1_micro; test_result.F_average];
                                 




function result_struct = multi_label_evaluation(label_matrix_prediction,label_matrix_gt, options)
% this function implements different types of evaluation metrics of multi-label learning
% please see NIPS 2011 "A Literature Survey on Algorithms for Multi-label
% Learning" for detailed definitions of all metrics

% Input: 
%       label_matrix_prediction, m x n matrix
%       label_matrix_gt, m x n matrix, each entry \in {-1, 1}
%       options, has the following fields
%            -- type, 'ranking', 'example-based-partition', 'label-based-partition'

if nargin == 2
    options.type = 'ranking';
    options.topk = 1;
end

[m, n] = size(label_matrix_prediction);

%%  transform the ranking matrix to discrete top-k label matrix
if (~strcmp(options.type, 'ranking')) && (~strcmp(options.type, 'AP-information-retrieval'))
    % if the prediction is already discrete
    if length(unique(label_matrix_prediction(:))) <= 2  
        label_matrix_prediction_binary = false(m,n);
        label_matrix_prediction_binary(label_matrix_prediction==1) = 1;
    else
        label_matrix_prediction_binary = false(m,n); % logical matrix
        topk = options.topk;
            parfor i = 1:n
                label_vector_i = label_matrix_prediction(:,i);
                [~, indx] = sort(label_vector_i, 'descend');
                binary_label_i = false(m,1);
                binary_label_i(indx(1:topk)) = 1;
                label_matrix_prediction_binary(:,i) = binary_label_i;
            end
    end
end

%% Type 1, ranking based evaluation metrics, including average-AUC, AP, Ranking loss, Coverage, One error
switch options.type
    
    case 'ranking'
        % compute the average AUC
%         tic
         [tpr,fpr] = mlr_roc(label_matrix_prediction', label_matrix_gt');
%         toc
       result_struct.AUC =  mlr_auc(fpr,tpr);  % this measure is costly, you can comment it if unnecessary
        % result_struct.AUC =  0;  
        % compute AP 
%         tic
        result_struct.AP = Average_precision(label_matrix_prediction, label_matrix_gt);
%         toc
        % compute Ranking Loss
%         tic
%         result_struct.RL=Ranking_loss(label_matrix_prediction,label_matrix_gt);
%         toc
        
    case 'AP-information-retrieval'
%         tic
        [result_struct.AP_vector, result_struct.MAP] = AP_information_retrieval(label_matrix_prediction,label_matrix_gt);
%         toc
%         result_struct.AP_vector = 0; 
%         result_struct.MAP = 0;
        
    case 'example-based-partition'
        label_matrix_gt(label_matrix_gt==-1) = 0; % transform to binary ground-truth label matrix
        label_matrix_gt = logical(label_matrix_gt); % transform to logical
        
        global positive_label_weight edgeMatrix
        weight_matrix = ones(m,n); weight_matrix(label_matrix_gt) = positive_label_weight;
        numEdges = size(edgeMatrix, 1); 
        
        accuracy_vector = zeros(1,n);
        precision_vector = zeros(1,n);
        recall_vector = zeros(1,n);
        %F1_vector = zeros(1,n); 
        hamming_loss_vector = zeros(1,n);
        hamming_loss_weighted_vector = zeros(1,n);
%              tic
             for i = 1:n
                    z_i = label_matrix_prediction_binary(:,i); % m x 1 vector, prediction label vector of instance i
                    y_i = label_matrix_gt(:,i); % m x 1 vector, ground-truth label vector of instance i
                    w_i = weight_matrix(:,i); % m x 1 vector, the weight of each ground-truth label
                    
                    sum_zi = sum(z_i);
                    sum_yi = sum(y_i); 
                    sum_wi = sum(w_i);

                    zi_and_yi = sum(z_i & y_i);
                    zi_or_yi = sum(z_i | y_i);
                    zi_xor_yi = sum(xor(z_i, y_i));
                    zi_xor_yi_weighted = sum(xor(z_i, y_i) .* w_i);

                    accuracy_vector(i) = zi_and_yi / (zi_or_yi+eps);
                    precision_vector(i) = zi_and_yi / (sum_zi + eps);
                    recall_vector(i) = zi_and_yi / (sum_yi + eps);
                    %F1_vector(i) = 2*zi_and_yi / (sum_yi + sum_zi+eps);
                    hamming_loss_vector(i) = zi_xor_yi / m; 
                    %hamming_loss_weighted_vector(i) = zi_xor_yi_weighted / sum_wi; 
                    hamming_loss_weighted_vector(i) = zi_xor_yi_weighted / m; 
                    
                    
             end
             F1_vector = 2.*precision_vector.*recall_vector ./(precision_vector + recall_vector+eps);
%              toc
        result_struct.topk_example_accuracy = mean(accuracy_vector);
        result_struct.topk_example_precision = mean(precision_vector);
        result_struct.topk_example_recall = mean(recall_vector);
        result_struct.topk_example_F1 = mean(F1_vector);
        result_struct.topk_example_hamming_loss = mean(hamming_loss_vector);  
        result_struct.topk_example_hamming_loss_weighted = mean(hamming_loss_weighted_vector);
        
    case 'label-based-partition'
        label_matrix_gt(label_matrix_gt==-1) = 0; % transform to binary ground-truth label matrix
        label_matrix_gt = logical(label_matrix_gt); % transform to logical        
        
        % macro averaged metrics
        precision_macro_vector = zeros(1,m);
        recall_macro_vector = zeros(1,m);
        %F1_macro_vector = zeros(1,m);
        for c = 1:m
            z_c = label_matrix_prediction_binary(c,:); % 1 x n vector, prediction label vector of instance i
            y_c = label_matrix_gt(c,:); % 1 x n vector, ground-truth label vector of instance i
            
            zc_and_yc = sum(z_c & y_c);
            sum_zc = sum(z_c);
            sum_yc = sum(y_c);
            
            precision_macro_vector(c) = zc_and_yc / ( sum_zc+eps);
            recall_macro_vector(c) = zc_and_yc / (sum_yc+eps);
            %F1_macro_vector(c) = 2*zc_and_yc / (sum_zc+sum_yc+eps);
        end
        F1_macro_vector = 2.*precision_macro_vector.*recall_macro_vector ./ (precision_macro_vector+recall_macro_vector+eps);
        result_struct.topk_label_precision_macro_vector = precision_macro_vector;
        result_struct.topk_label_precision_macro_average = mean(precision_macro_vector);
        result_struct.topk_label_recall_macro_vector = recall_macro_vector;
        result_struct.topk_label_recall_macro_average = mean(recall_macro_vector);  
        result_struct.topk_label_F1_macro_vector = F1_macro_vector;
        result_struct.topk_label_F1_macro_average = mean(F1_macro_vector);       
        
        % micro averaged metrics
        sum_Z = sum(label_matrix_prediction_binary(:));
        sum_Y = sum(label_matrix_gt(:));
        Z_and_Y = sum(sum(label_matrix_prediction_binary & label_matrix_gt));
        
        result_struct.topk_label_precision_micro = Z_and_Y / (sum_Z+eps);
        result_struct.topk_label_recall_micro = Z_and_Y / (sum_Y+eps);
        result_struct.topk_label_F1_micro = 2*Z_and_Y / (sum_Z+sum_Y+eps);
end


function [AP_vector, MAP] = AP_information_retrieval(label_matrix_prediction,label_matrix_gt)

% Input: 
%       label_matrix_prediction, m x n matrix, continuous ranking values
%       label_matrix_gt, m x n matrix, each entry \in {-1, 1}

[m,n] = size(label_matrix_prediction);

label_matrix_gt(label_matrix_gt==-1)=0;
positive_instance_vector = sum(label_matrix_gt,2); % the number of positive instances of each class

AP_vector = zeros(1,m);
for c = 1:m
    n_c = positive_instance_vector(c); % the number of positive instances of the c-th class
    positive_instance_location_c = find(label_matrix_gt(c,:)); % the location of the positive instances of the c-th class
    
    [~, ranking_c_index] = sort(label_matrix_prediction(c,:),'descend');
    precision_vector = zeros(1,n_c);
    for i = 1: positive_instance_vector(c)
        precision_vector(i) = length( intersect( ranking_c_index(1:i), positive_instance_location_c) ) / i;
    end
    AP_vector(c) = mean(precision_vector);
end
MAP = mean(AP_vector);
