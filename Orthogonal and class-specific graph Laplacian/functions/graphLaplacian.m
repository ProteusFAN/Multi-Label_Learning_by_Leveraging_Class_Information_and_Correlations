function [ Laplacian, W ] = graphLaplacian( data_info, options )
% class-specific graph laplacian
% compute weighted graph laplacian over general one and class-specific one
% two methods to determine class-specific graph laplacian: (1) cumulative
%   variance. (2) parallel analysis of PCA.

%% ==== setting ====
threshold_weight = 1/4; % weight to combine general with specific ones

Laplacian = cell(data_info.num_c,1);
W = cell(data_info.num_c,1);

%% ==== general graph laplacian ====
[L_g, W_g] = computeGraphLaplacian(data_info.dataset_matrix,...
    'type', 'general', 'normalized', options.L_x_normalized, 'batch_size', 10,...
    'neighbor_size', options.L_x_neighbor_size,...
    'kernel_size', options.L_x_kernel_size);

if ~options.L_x_specificLaplacian
    for j = 1:data_info.num_c
        Laplacian{j} = L_g;
        W{j} = W_g;
    end
    return;
end

%% ==== specific graph laplacian ====
L_s = cell(data_info.num_c,1);
W_s = cell(data_info.num_c,1);
for i = 1:data_info.num_c
    [feature, specific_idx] = LDA_transform(data_info, i);
    [L_s{i}, W_s{i}] = computeGraphLaplacian(feature,...
        'type', 'specific', 'normalized', options.L_x_normalized, 'batch_size', 10,...
        'neighbor_size', options.L_x_neighbor_size,...
        'kernel_size', options.L_x_kernel_size,...
        'specific_idx', specific_idx);
end

alpha = min(1, sum(max(data_info.label_train,0),2)./(threshold_weight * data_info.num_sample_train));
for j = 1:data_info.num_c
    Laplacian{j} = (1-alpha(j)).* L_g + alpha(j).* L_s{j};
    W{j} = (1-alpha(j)).* W_g + alpha(j).* W_s{j};
end

end

