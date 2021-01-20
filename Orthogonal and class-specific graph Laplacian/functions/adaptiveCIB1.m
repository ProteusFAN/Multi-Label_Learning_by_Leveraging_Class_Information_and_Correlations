function [ CIB_1 ] = adaptiveCIB1( data_info )
%% ==== setting ====
num_neigbor = ceil(log(data_info.num_sample_train)^2);
percent_lower = 0.15;
percent_upper = 0.85;
batch = 10;

%% ==== Calculation of CIB-1 ====

% ==== build kd-tree ====
kdtree = vl_kdtreebuild(data_info.dataset_train');

% ==== find k-nearest neighbors and their distances based on kd-tree ====
num_batch = fix(data_info.num_sample/batch);
index_cell = cell(1,num_batch);

if data_info.num_sample > 5000
    parfor iter_batch = 1:num_batch
        i = (iter_batch - 1) * batch + 1 : iter_batch * batch;
        index_cell{iter_batch} = vl_kdtreequery(kdtree, data_info.dataset_train',...
            data_info.dataset_matrix(i,:)', 'NumNeighbors', num_neigbor+1);
    end
else
    for iter_batch = 1:num_batch %1:batch:num_sample
        i = (iter_batch - 1) * batch + 1 : iter_batch * batch;
        index_cell{iter_batch} = vl_kdtreequery(kdtree, data_info.dataset_train',...
            data_info.dataset_matrix(i,:)', 'NumNeighbors', num_neigbor+1); 
    end
end
index_matrix = [index_cell{:}];
i = num_batch * batch + 1 : data_info.num_sample;
index_matrix_end = vl_kdtreequery(kdtree, data_info.dataset_train',...
    data_info.dataset_matrix(i,:)', 'NumNeighbors', num_neigbor+1);
index_matrix = [index_matrix, index_matrix_end];
index_matrix(1,:) = [];

% ==== calculate CIB-1 ====
num_label_train = sum(max(data_info.label_train,0));
labelNeighbor = sort(num_label_train(index_matrix));

CIB_1{1} = labelNeighbor(round(max(1, num_neigbor*percent_lower)),:);
CIB_1{2} = labelNeighbor(round(num_neigbor*percent_upper),:);

end