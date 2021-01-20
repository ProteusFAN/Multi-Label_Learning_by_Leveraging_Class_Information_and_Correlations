function [ data_info ] = dataLoading( type )
%% ==== data loading ====
%   Description: loading feature and label (feature matrix: sample as row,
%   label matrix: sample as column)
%       standardize label to {-1, +1} 
%       normalizing each feature to [-1, 1]
%       'type': 'yeast', 'emotions', 'scene'

%% ==== loading data as type ====
switch type
    case 'yeast'
        dataset_train = load('yeast_train.txt');
        label_train = (dataset_train(:,end-13:end))';
        dataset_train(:,end-13:end) = [];
        
        dataset_test = load('yeast_test.txt');
        label_test = (dataset_test(:,end-13:end))';
        dataset_test(:,end-13:end) = [];
        
    case 'emotions'
        dataset_train = load('Emotions_data_train.txt');
        dataset_test = load('Emotions_data_test.txt');
        label_train = load('Emotions_label_train.txt');
        label_test = load('Emotions_label_test.txt');
        
    case 'scene'
        dataset_train = load('data_train.txt');
        dataset_test = load('data_test.txt');
        label_train = load('label_train.txt');
        label_test = load('label_test.txt');        
end   

dataset_matrix = [dataset_train; dataset_test];
[num_sample, num_dimension] = size(dataset_matrix);

%% ==== standardize feature to {-1, +1} ====
label_train(label_train == 0) = -1;
label_test(label_test == 0) = -1;
[num_c, num_sample_train] = size(label_train);
[~, num_sample_test] = size(label_test);

%% ==== normalize each feature to [ -1,1] ====
dataset_matrix = mapminmax(dataset_matrix')';
dataset_train = dataset_matrix(1:num_sample_train,:);
dataset_test = dataset_matrix(1+num_sample_train:end,:);

%% ==== data_info ====
data_info = struct('dataset_matrix', dataset_matrix,...
    'dataset_train', dataset_train, 'dataset_test', dataset_test,...
    'label_train', label_train, 'label_test', label_test,...
    'num_sample', num_sample, 'num_sample_train', num_sample_train,...
    'num_sample_test', num_sample_test, 'num_c', num_c,...
    'num_dimension', num_dimension, 'type', type);

end

