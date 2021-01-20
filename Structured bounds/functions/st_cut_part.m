 %% ------------------------------------ Method 1: st-cut

        % transform the initial label matrix to vector
        N = num_c*num_sample; 
        y0 = reshape(initial_assign_matrix, N, 1); 
        id_neg = y0 == -1;
        id_pos = y0 == 1;
        
        location_positive = initial_assign_matrix==1;  
        positive_label_weight_matrix = repmat(positive_label_weight_vector, 1, num_sample); 
        weight_matrix = single(ones(num_c, num_sample));
        weight_matrix(location_positive) = positive_label_weight_matrix(location_positive);
        Yw = initial_assign_matrix .* weight_matrix;
        Yw_vector = reshape(Yw, N, 1);
        clear weight_matrix

        cost_neg_vector = zeros(1, N);
        cost_neg_vector(id_pos) = 2 * Yw_vector(id_pos);  % the unary cost of incorrectly setting positive to negative labels
        cost_pos_vector = zeros(1, N);
        cost_pos_vector(id_neg) = 2;  % the unary cost of incorrectly setting negative to positive labels
        
        unary_cost_matrix = [cost_neg_vector; cost_pos_vector]; % 2 x N matrix
        

        %% define the edge_matrix based on Vx and Vc
        
        tic
        edges_matrix_x = single(Vx2edgesmatrix(V_x,num_c)); 
        toc
        edges_matrix_c = single(Vc2edgesmatrix(V_c,num_sample));
 
        edges_matrix_x(:,6:9) = edges_matrix_x(:,6:9).*(lambda_x);
        edges_matrix_c(:,6:9) = edges_matrix_c(:,6:9).*(lambda_c);
        clear edges_matrix_x_original edges_matrix_c_original

        edges_matrix_whole = double([edges_matrix_c; edges_matrix_x]);
        nEdges_whole = size(edges_matrix_whole,1); 
        clear edges_matrix_c edges_matrix_x
                     

        %% create the handle for st-cut
        %  BK_ListHandles   % when it is your first time to call this toolbox, please uncomment this line. However, it may take a long time to run it  
        h = BK_Create();
        BK_AddVars(h,num_c*num_sample);
        BK_SetUnary(h,unary_cost_matrix);  
        
        BK_SetPairwise(h, edges_matrix_whole(:, [1:2, 6:end]));
        e = BK_Minimize(h); 
        fprintf('The minimal objective function of st-cut is %f \n', e);
        tic
        labeling_stcut=BK_GetLabeling(h); 
        toc
        labeling_stcut_matrix = single(reshape(labeling_stcut, num_c, num_sample));
        Z_stcut = double(2.* (labeling_stcut_matrix -1.5)); % transform to {-1, 1}
        
        y_train = label_train; y_train(y_train==0)=-1; 
        y_test = label_test; y_test(y_test==0)=-1; 
        
        [result_train_stcut_vec, result_test_stcut_vec, result_train_stcut, result_test_stcut] = evaluation_discrete_multi_label(Z_stcut, y_train, y_test);
        result_train_stcut_vec =  [lambda_x; lambda_c; result_train_stcut_vec];
        result_test_stcut_vec =  [lambda_x; lambda_c; result_test_stcut_vec];

        %% objective function 
        const = trace(Yw * initial_assign_matrix');
        obj_stcut = const - trace(Yw * Z_stcut') + lambda_x * trace((Z_stcut * L_x) * Z_stcut') + lambda_c * trace(L_c * (Z_stcut * Z_stcut') );
