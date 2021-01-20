function [ graphLaplacian, W ] = computeGraphLaplacian( feature, varargin )
% compute graph laplacian matrix

%% ==== setting ====

if find(strcmp(varargin, 'type'))
    type = varargin{find(strcmp(varargin, 'type')) + 1};
else
    type = 'general';  % default to compute general graph laplacian
%     type = 'specific';
end

if find(strcmp(varargin, 'normalized'))
    normalized = varargin{find(strcmp(varargin,'normalized')) + 1};
else
    normalized = true;  % default to use normalized graph laplacian
end

if find(strcmp(varargin, 'batch_size'))
    batch_size = varargin{find(strcmp(varargin, 'batch_size')) + 1};
else
    batch_size = 10;  % default batch_size
end

if find(strcmp(varargin, 'neighbor_size'))
    neighbor_size = varargin{find(strcmp(varargin,'neighbor_size')) + 1};
else
    neighbor_size = 20;  % default num_neighbor_size
end

if find(strcmp(varargin, 'kernel_size'))
    kernel_size = varargin{find(strcmp(varargin,'kernel_size')) + 1};
else
    kernel_size = 7;  % default num_kernel_size
end

if find(strcmp(varargin, 'specific_idx'))
    specific_idx = varargin{find(strcmp(varargin, 'specific_idx')) + 1};
end

%% ==== compute kd-tree and the k-nearest neighbors and distances based on the kd-tree ====
num_sample = size(feature,1);
kdtree = vl_kdtreebuild(feature'); 
num_batch = fix(num_sample/batch_size);
index_cell = cell(1,num_batch);
distance_cell = cell(1,num_batch);

for iter_batch = 1:num_batch
    i = (iter_batch - 1) * batch_size + 1 : iter_batch * batch_size;
    [index_cell{iter_batch}, distance_cell{iter_batch} ] = vl_kdtreequery(kdtree,...
        feature', feature(i,:)', 'NumNeighbors', neighbor_size+1) ;    
end
index_matrix = [ index_cell{:} ];
distance_matrix = [ distance_cell{:} ]; 

i = num_batch * batch_size + 1 : num_sample;
[index_matrix_end, distance_matrix_end] = vl_kdtreequery(kdtree,...
    feature', feature(i,:)', 'NumNeighbors', neighbor_size+1) ;    
index_matrix = [index_matrix, index_matrix_end];
distance_matrix = [distance_matrix, distance_matrix_end];

index_matrix(1,:) = [];
distance_matrix(1,:) = [];

%% ==== compute weight matrix in graph laplacian based on type (general or specific) ====
kernel_size_vec = distance_matrix(kernel_size,:);
V = zeros(size(distance_matrix));

switch type
    case 'general'
        for ii = 1:num_sample
            dis_vec = distance_matrix(:,ii);
            kernel_size = kernel_size_vec(ii);
            V(:,ii) = exp( -dis_vec.^2./ kernel_size^2 );
        end
        
    case 'specific'
        num_sample_train = numel(specific_idx);
        feature_train = feature(1:num_sample_train);
        feature_train_pos = feature_train(specific_idx);
        feature_train_neg = feature_train(~specific_idx);
        std_pos = std(feature_train_pos);
        std_neg = std(feature_train_neg);
        var_pos = std_pos^2;
        var_neg = std_neg^2;
        mean_neg = mean(feature_train_neg);
        
        para_a = 1/var_neg - 1/var_pos;
        para_b = -2 * mean_neg/var_neg;
        para_c = mean_neg^2/var_neg + log(std_neg) - log(std_pos);
        delta = para_b^2 - 4 * para_a * para_c;
        threshold = (-para_b - delta^0.5) / (2 * para_a);
        for ii = 1:num_sample
            score = (feature(index_matrix(:,ii)) + feature(ii)) / 2;
            ratio = tanh( (score - threshold)/ (mean_neg/2) ) + 1;
            dis_vec = distance_matrix(:,ii);
            kernel_size = kernel_size_vec(ii);
            V(:,ii) = exp( -dis_vec.^2./ kernel_size^2.* ratio );  % linear ratio
        end
end

V = single(V);
W = single(zeros(num_sample, num_sample));
for j = 1:num_sample
    W(j,index_matrix(:,j)) = V(:,j);
end
W = 0.5.*(W + W');
W = double(W);
W = sparse(W);

%% ==== compute graph laplacian ====
if normalized
    d_halfInverse = diag(sqrt(1./sum(W,1)));
    graphLaplacian = diag(ones(1,num_sample)) - d_halfInverse * W * d_halfInverse;
    graphLaplacian = sparse(graphLaplacian);
else
    D = diag(sum(W,1));
    graphLaplacian = D - W;
    graphLaplacian = sparse(graphLaplacian);
end

end
