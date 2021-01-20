function [ L_c ] = co_occurrence( data_info, options )
%% compute co-occurrence matrix (L_c)

%% ==== parameter setting ====
num_c = data_info.num_c;
if options.L_c_neighbor_size
    L_c_neighbor_size = options.L_c_neighbor_size;
else
    L_c_neighbor_size = ceil(size(data_info.label_train,1)/4);
end

%% ==== compute full similarity matrix, based on co-occurrence ====
% asymmetric co-occurrency matrix
label_matrix = logical(max(data_info.label_train, 0));
co_occurrence_matrix = zeros(num_c);
size_vector_c = sum(label_matrix, 2) + eps;
for i = 1:num_c
    product_matrix = repmat(label_matrix(i,:), num_c, 1) & label_matrix;
    co_occurrence_matrix(i,:) = sum(product_matrix,2)' ./ size_vector_c(i);
end
co_occurrence_matrix = co_occurrence_matrix - diag(diag(co_occurrence_matrix));

%% ==== compute the k-nn sparse similarity matrix ====
co_occurrence_matrix_knn = zeros(size(co_occurrence_matrix));
for i = 1:num_c
    sim_vec_i = co_occurrence_matrix(i,:);
    sim_vec_i_sort = sort(sim_vec_i, 'descend');
    thresh_value = sim_vec_i_sort(L_c_neighbor_size); 
    sim_vec_i(sim_vec_i <= thresh_value) = 0;
    co_occurrence_matrix_knn(i,:) = sim_vec_i;
end
co_occurrence_matrix_knn = (co_occurrence_matrix_knn + co_occurrence_matrix_knn') .* 0.5;

%% ==== L_c ====
W = co_occurrence_matrix_knn;
if options.L_c_normalized
    d_halfInverse = diag(sqrt(1./sum(W,1)));
    L_c = diag(ones(1,num_c)) - d_halfInverse * W * d_halfInverse;
else
    D = diag(sum(W,1));
    L_c = D - W;
end

end

