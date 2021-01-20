
%% using KD-tree to compute the similarity matrix
% run('D:\matlab code\vlfeat-0.9.19\toolbox\vl_setup')

t0 = cputime;
%% --------- Step 1 : generate the kd-tree
tic
kdtree = vl_kdtreebuild(dataset_matrix') ;
toc

%% -------- Step 2 : based on the kd-tree, compute the k-nearest neighbors and distances
num_batch = fix(num_sample/batch);
index_cell = cell(1,num_batch);
distance_cell = cell(1,num_batch);

if num_sample > 5000
        tic
        parfor iter_batch = 1:num_batch %1:batch:num_sample
            i = (iter_batch-1)*batch +1 : iter_batch*batch;
              tic
                [index_cell{iter_batch}, distance_cell{iter_batch} ] = vl_kdtreequery(kdtree, dataset_matrix', dataset_matrix(i,:)', 'NumNeighbors', num_neighbor_size+1) ;    
              toc
        end
        toc
else
        tic
        for iter_batch = 1:num_batch %1:batch:num_sample
            i = (iter_batch-1)*batch +1 : iter_batch*batch;
            [index_cell{iter_batch}, distance_cell{iter_batch} ] = vl_kdtreequery(kdtree, dataset_matrix', dataset_matrix(i,:)', 'NumNeighbors', num_neighbor_size+1) ;    
        end
        toc
end

index_matrix = [ index_cell{:} ];
distance_matrix = [ distance_cell{:} ]; 

i = num_batch * batch + 1 : num_sample;
tic
[index_matrix_end, distance_matrix_end] = vl_kdtreequery(kdtree, dataset_matrix', dataset_matrix(i,:)', 'NumNeighbors', num_neighbor_size+1) ;    
toc
index_matrix = [ index_matrix, index_matrix_end];
distance_matrix = [ distance_matrix, distance_matrix_end];

index_matrix(1,:) = []; 
distance_matrix(1,:) = []; 
toc

%% --------  Step 3 : compute the similarity matrix
kernel_size_vec = distance_matrix(num_kernel_size,:);
V = zeros(size(distance_matrix));
tic
for i = 1:num_sample
%       tic
    dis_vec = distance_matrix(:,i);
    kernel_size = kernel_size_vec(i); 
    V(:,i) = exp(-dis_vec.^2./kernel_size^2); % the kernel size here is different with the one used in our previous works
%       toc
end
toc
V=single(V);

 V_x = single(zeros(num_sample, num_sample));
%V_x = sparse(num_sample, num_sample);
tic
for i = 1:num_sample
%        tic
    V_x(i,index_matrix(:,i)) = V(:,i);
%        toc
end
toc
V_x = 0.5.*(V_x + V_x');
V_x = double(V_x);
V_x = sparse(V_x);

%%  -------- Step 4 : compute the normalized Laplacian matrix, L_x
dd=full(sum(V_x,1)); % the summation of each row, n*1 vector
dd=sqrt(1./dd);  
degree_matrix=diag(sparse(dd));
V_x_Normalized=sparse(degree_matrix*V_x*degree_matrix);
L_x= diag(sparse(ones(1,num_sample)))- V_x_Normalized;
sum(L_x(:))
sum(V_x_Normalized(:))
clear V_x_Normalized degree_matrix

D = diag(sum(V_x,1)); 
L_x_unnormalized = D - V_x; 


