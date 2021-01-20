
function edges_matrix3 = Vc2edgesmatrix(Vc,n)

%% The following demo shows how to transform Vc to the Edges_matrix compatiable to Bk_matlab
m = size(Vc,1);
d_sqrt_inver = 1./ sqrt(sum(Vc,2)); 

%% determine the neigbors of each class and its similarity
Pair_Cell = cell(1,m-1);
for i = 1:m-1
    Vc_vec_i = Vc(i,:);
    neigh_index = intersect(find(Vc_vec_i), i+1:m);
    num_neigh = length(neigh_index); 
    sim_vec = Vc_vec_i(neigh_index); 
    Pair_Cell{i} = [i.*ones(num_neigh,1), neigh_index', sim_vec',...
                       d_sqrt_inver(i).*ones(num_neigh,1), d_sqrt_inver(neigh_index)]';
end
edges_matrix1 = single(full([Pair_Cell{:}]')); % nEdges x 5, each row is (i,j, Vc(i,j), 1/sqrt(d_i), 1/sqrt(d_j) )
clear Pair_Cell
nEdges_single = size(edges_matrix1,1); 

%% transform 'edges_matrix1' to 'edges_matrix2', following the form of Bk_matlab

i_vector = sub2ind([m,n], edges_matrix1(:,1), ones(nEdges_single,1) );
j_vector = sub2ind([m,n], edges_matrix1(:,2), ones(nEdges_single,1) );

% normalized cut
cost_00_vector = edges_matrix1(:,3) .* ( edges_matrix1(:,4) - edges_matrix1(:,5) ).^2; 
cost_01_vector = edges_matrix1(:,3) .* ( edges_matrix1(:,4) + edges_matrix1(:,5) ).^2;
% unnormalized cut
% cost_00_vector = edges_matrix1(:,3) .* 0; 
% cost_01_vector = edges_matrix1(:,3) .* 4;

edges_matrix2 = [i_vector, j_vector, edges_matrix1(:,3:5), cost_00_vector, cost_01_vector, cost_01_vector, cost_00_vector];
clear edges_matrix1

%% repeat the class edges to all columns (instances),  
edges_matrix3 = repmat(edges_matrix2,n,1);
clear edges_matrix2
for i = min(2,n) : n
    indx = (i-1)*nEdges_single+1 : i*nEdges_single; 
    edges_matrix3( indx, 1:2 ) = edges_matrix3( indx, 1:2 ) + (i -1).*m;
end
% edges_matrix_c = edges_matrix3; 
% edges_matrix_c(:,3:6) = edges_matrix_c(:,3:6).*lambda_c*0.5;

