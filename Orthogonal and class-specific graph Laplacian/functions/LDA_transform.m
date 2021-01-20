function [ feature, idx ] = LDA_transform( data_info, i )
%% LDA_transform
% transform feature to one dimension decided by LDA

%% ==== preprocess ====
idx = data_info.label_train(i,:) == 1;
feature_p = data_info.dataset_train(idx,:);
feature_n = data_info.dataset_train(~idx,:);

mean_all = mean(data_info.dataset_train);
mean_p = mean(feature_p);
mean_n = mean(feature_n);

%% ==== between-class covariance matrix and within-class covariance matrix ====
mat_b = size(feature_p,1) * (mean_p-mean_all)'*(mean_p-mean_all) + size(feature_n,1) * (mean_n-mean_all)'*(mean_n-mean_all);

mat_w = zeros(size(data_info.dataset_train,2));
for i_sample = 1:size(feature_p,1)
    mat_w = mat_w + (feature_p(i_sample,:)-mean_p)' * (feature_p(i_sample,:)-mean_p);
end

for i_sample = 1:size(feature_n,1)
    mat_w = mat_w + (feature_n(i_sample,:)-mean_n)' * (feature_n(i_sample,:)-mean_n);
end
mat_w = 1/(size(data_info.dataset_train,2) - 2) .* mat_w;

mat = mat_w \ mat_b;
[V, ~] = eig(mat);
vector = V(:,1);

%% ==== new feature ====
feature = data_info.dataset_matrix * vector;

feature_train = feature(1:data_info.num_sample_train);
feature_new_p = feature_train(idx,:);
feature_new_n = feature_train(~idx,:);

feature = feature - mean(feature_new_p);
if mean(feature_new_p) > mean(feature_new_n)
    feature = -feature;  % ensure negative samples lie rightside
end

end