        %%  the edgeStructure of the class-level similarity sub-graph
        V_c_triu = triu(V_c); 
        [node1_c, node2_c] = find(V_c_triu > 0); 
        num_ec = length(node1_c); 
        edgeStructure.class.edge_id = [1:num_ec]'; 
        edgeStructure.class.node1_id = node1_c; 
        edgeStructure.class.node2_id = node2_c; 
        dc_sqrt_inver = 1./ sqrt(sum(V_c,2)); 
        edgeStructure.class.similarity = 2.*V_c(sub2ind([num_c, num_c], node1_c, node2_c)) .*dc_sqrt_inver(node1_c) .*dc_sqrt_inver(node2_c); 
                           
        %% define the constant parameter matrices. please refer to our paper for the definitions of these notations   
        B = sparse(num_ec, num_c);
        B(sub2ind([num_ec, num_c], edgeStructure.class.edge_id, edgeStructure.class.node1_id)) = 1; % B(e, e_1) = 1;
        B(sub2ind([num_ec, num_c], edgeStructure.class.edge_id, edgeStructure.class.node2_id)) = -1; % B(e, e_2) = -1;  
        B_bar = sparse( [B; -B; -ones(1,num_c); ones(1,num_c)] );
        M2 = B_bar' * B_bar; 
        D_bar = sparse( [eye(num_ec); eye(num_ec); zeros( 2, num_ec)] );

        %% save the constant matrices to the unified structure 'constant_matrix'
        constant_matrix.B = B; 
        constant_matrix.B_bar = B_bar; 
        constant_matrix.D_bar = D_bar; 
        constant_matrix.M2 = M2; 