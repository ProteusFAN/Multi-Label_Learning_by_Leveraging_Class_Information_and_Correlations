function [ indicator ] = papca( raw )
%% parallel analysis for principal component analysis
%  feature is composed by samples as row.

%% ==== setting ====
pval_threshold = 0.05;
ndatasets = 100;

%% ==== papca algorithm ====
[n, m] = size(raw); % suppose m < n

sigval = zeros(min(n,m), ndatasets);
for iter = 1:ndatasets
    raw_perm = zeros(size(raw));
    for m_iter = 1:m
        raw_perm(:,m_iter) = raw(randperm(n),m_iter);
    end
    sigval(:,iter) = svd(raw_perm);
end

realsig = svd(raw);
pval = sum(sigval > realsig*ones(1,ndatasets), 2)/ndatasets;

indicator = sum(pval < pval_threshold);

end

% num_pc = zeros(14,1);
% 
% for i = 1:14
%     feature = data_info.dataset_train(data_info.label_train(i,:) == 1,:);
%     num_pc(i) = papca(feature);
% end