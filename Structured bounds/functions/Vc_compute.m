function [ V_c_normalized, V_c, Vc_cell, co_occurrency_matrix ] = Vc_compute( label_matrix, num_neighbor )
%% compute the class similarity Vc
num_c = size(label_matrix,1);

Y0 = label_matrix; % 0 or 1
Y0(Y0==-1) = 0;
Y0 = sign(Y0);

%% compue the full similarity matrix, based on co-occurrency
% asymmetric co-occurrency matrix 
size_c = diag(1./sum(Y0,2)); % num_c x 1 vector, |y_i|

co_occurrency_matrix = size_c * (Y0 * Y0') * size_c;
co_occurrency_matrix = co_occurrency_matrix - diag(diag(co_occurrency_matrix));

%% compute the k-nn sparse similarity matrix
co_occurrency_matrix_knn = co_occurrency_matrix; 
Vc_cell = cell(num_c, 1); %zeros(num_c, num_neighbor);
for i = 1:num_c
    sim_vec_i = co_occurrency_matrix(i,:);
    [sim_vec_i_sort, index_sort] = sort(sim_vec_i,'descend');
    Vc_cell{i} = [index_sort(1:num_neighbor); sim_vec_i_sort(1:num_neighbor)];
    thresh_value = sim_vec_i_sort(num_neighbor); 
    sim_vec_i(sim_vec_i<thresh_value) = 0;
    co_occurrency_matrix_knn(i,:) = sim_vec_i;
end
co_occurrency_matrix_knn = (co_occurrency_matrix_knn + co_occurrency_matrix_knn') .* 0.5;

% normalized affinity matrix
AM = co_occurrency_matrix_knn;
AM = AM - diag(diag(AM));
dd_c = sum(AM, 2)+eps; % the summation of each row, n*1 vector
dd_c = sqrt(1./dd_c);
degree_matrix = sparse(diag(dd_c));
V_c_normalized = degree_matrix * AM* degree_matrix;
V_c = co_occurrency_matrix_knn;
