function edges_matrix_x = Vx2edgesmatrix_original(Vx,m,lambda_x)


%% The following demo shows how to transform Vx to the Edges_matrix compatiable to Bk_matlab
n = size(Vx,1);
d_sqrt_inver = 1./ sqrt(sum(Vx,2)); 

%% determine the neigbors of each instance and its similarity
Pair_Cell = cell(1,n-1);
tic
for i = 1:n-1
    Vx_vec_i = Vx(i,:);
    neigh_index = intersect(find(Vx_vec_i), i+1:n);
    num_neigh = length(neigh_index); 
    sim_vec = Vx_vec_i(neigh_index); 
    Pair_Cell{i} = [i.*ones(num_neigh,1), neigh_index', sim_vec',...
                       d_sqrt_inver(i).*ones(num_neigh,1), d_sqrt_inver(neigh_index)]';
end
toc
edges_matrix1 = single(full([Pair_Cell{:}]')); % nEdges x 5, each row is (i,j, Vx(i,j), 1/sqrt(d_i), 1/sqrt(d_j) )
clear Pair_Cell
nEdges_single = size(edges_matrix1,1); 

%% transform 'edges_matrix1' to 'edges_matrix2', following the form of Bk_matlab
% edges_matrix2 = zeros(nEdges_single,6); 
% tic
% for e = 1:nEdges_single
%     vec_e = edges_matrix1(e,:);
%     i = sub2ind([m,n], 1, vec_e(1));
%     j = sub2ind([m,n], 1, vec_e(2));
%     cost_00 = vec_e(3) * (vec_e(4) - vec_e(5))^2; 
%     cost_01 = vec_e(3) * (vec_e(4) + vec_e(5))^2;
%     edges_matrix2(e,:) = [i, j, cost_00, cost_01, cost_01, cost_00];
% end
% toc

i_vector = sub2ind([m,n], ones(nEdges_single,1), edges_matrix1(:,1));
j_vector = sub2ind([m,n], ones(nEdges_single,1), edges_matrix1(:,2));
cost_00_vector = edges_matrix1(:,3) .* ( edges_matrix1(:,4) - edges_matrix1(:,5) ).^2; 
cost_01_vector = edges_matrix1(:,3) .* ( edges_matrix1(:,4) + edges_matrix1(:,5) ).^2;

clear edges_matrix1
edges_matrix2 = [i_vector, j_vector, cost_00_vector, cost_01_vector, cost_01_vector, cost_00_vector];

%% repeat the instance edges to all rows (classes),  
edges_matrix3 = repmat(edges_matrix2,m,1);
clear edges_matrix2
for i = min(2,m) : m
    indx = (i-1)*nEdges_single+1 : i*nEdges_single; 
    edges_matrix3( indx, 1:2 ) = edges_matrix3( indx, 1:2 ) + i -1;
end

edges_matrix_x = edges_matrix3; 
edges_matrix_x(:,3:6) = edges_matrix_x(:,3:6).*lambda_x*0.5;

