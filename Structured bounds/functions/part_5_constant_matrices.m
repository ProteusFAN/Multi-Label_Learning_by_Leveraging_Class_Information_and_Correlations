        %% define the constant parameter matrices. please refer to our paper for the definitions of these notations
        H_bar = D_x * ones(num_sample, 1);
        
        %% save the constant matrices to the unified struct 'constant_matrix'
        constant_matrix.H_bar = H_bar;
        constant_matrix.M5 = H_bar * H_bar';