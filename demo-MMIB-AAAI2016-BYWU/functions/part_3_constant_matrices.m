       
        % constraint on the number of positive instances 
        v2_lower = 2.*options.cardinanity.instance_lower - num_sample; 
        v2_upper = 2.*options.cardinanity.instance_upper - num_sample; 
        G1_bar = sparse( [zeros(num_c, 2*num_ex), -v2_lower, v2_upper] );  
        
        % constraint on the number of positive classes 
        v1_lower = 2*options.cardinanity.class_lower - num_c; 
        v1_upper = 2*options.cardinanity.class_upper - num_c; 
        G2_bar = sparse( [zeros(2*num_ec, num_sample); -v1_lower.*ones(1,num_sample); v1_upper.*ones(1,num_sample)] ); 

        %% save the constant matrices to the unified structure 'constant_matrix'
        constant_matrix.G1_bar = G1_bar; 
        constant_matrix.G2_bar = G2_bar; 