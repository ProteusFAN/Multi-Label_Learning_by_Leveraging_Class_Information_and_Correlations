function [ graphLaplacian, W ] = computeFullGraphLaplacian( feature, GorS, Normalized )
% compute full graph laplacian matrix
if nargin == 2
    Normalized = True;
elseif nargin == 1
    GorS = True;
end

%% ==== setting ====
num_kernel = 7;

%% ==== compute the adjacency weight matrix
[num_sample, ~] = size(feature);
distance = zeros(num_sample);

for i = 1:num_sample
    temp_mat = bsxfun(@minus, feature, feature(i,:));
    distance(:,i) = sqrt(sum(temp_mat.^2,2));
end

kernel_sort = sort(distance);
kernel_vec = kernel_sort(num_kernel,:);

if GorS
    temp_mat = bsxfun(@times, distance.^2, 1./kernel_vec);
    temp_mat = bsxfun(@times, temp_mat, 1./kernel_vec');
    W = exp(-temp_mat);
else
    norm2squared = sum(feature.^2,2);
    W_sup = repmat(norm2squared, 1, n) + repmat(norm2squared', n, 1);
    temp_mat = bsxfun(@times, distance.^2 .* W_sup, 1./(kernel_vec.^2));
    temp_mat = bsxfun(@times, temp_mat, 1./(kernel_vec'.^2));
    W = exp(-temp_mat);
end
W = W - diag(ones(num_sample,1));

%% ==== compute graph laplacian ====
if Normalized
    d_halfInverse = diag(sqrt(1./sum(W,1)));
    graphLaplacian = diag(ones(1,num_sample)) - d_halfInverse * W * d_halfInverse;
    graphLaplacian = sparse(graphLaplacian);
else
    D = diag(sum(W,1)); 
    graphLaplacian = D - W;
    graphLaplacian = sparse(graphLaplacian);
end

end

