       %% the edgeStructure of the instance-level sub-graph
        V_x_triu = triu(V_x); 
        [node1_x, node2_x] = find(V_x_triu > 0); 
        num_ex = length(node1_x); 
        edgeStructure.instance.edge_id = [1:num_ex]'; 
        edgeStructure.instance.node1_id = node1_x; 
        edgeStructure.instance.node2_id = node2_x;
        dx_sqrt_inver = 1./ sqrt(sum(V_x,2)); 
        edgeStructure.instance.similarity = 2.*V_x(sub2ind([num_sample, num_sample], node1_x, node2_x)) .*dx_sqrt_inver(node1_x) .*dx_sqrt_inver(node2_x);
   
        %% define the constant parameter matrices. please refer to our paper for the definitions of these notations        
        A = zeros(num_sample, num_ex);
        A(sub2ind([num_sample, num_ex], edgeStructure.instance.node1_id, edgeStructure.instance.edge_id)) = 1; % A(e_1, e) = 1;
        A(sub2ind([num_sample, num_ex], edgeStructure.instance.node2_id, edgeStructure.instance.edge_id)) = -1; % A(e_2, e) = -1;
        A = sparse( A ); 
        tem_vec = ones(num_sample, 1); 
        A_bar = sparse([A, -A, -tem_vec, tem_vec]); 
        M1 = A_bar * A_bar'; 
        clear tem_vec
        
        one_diag = diag( sparse(ones(1,num_ex)) );
        C_bar = [one_diag, one_diag, sparse(zeros(num_ex, 2))]; 
        clear one_diag
        
        %% save the constant matrices to the unified struct 'constant_matrix'
        constant_matrix.A = A; 
        constant_matrix.A_bar = A_bar; 
        constant_matrix.C_bar = C_bar; 
        constant_matrix.M1 = M1;