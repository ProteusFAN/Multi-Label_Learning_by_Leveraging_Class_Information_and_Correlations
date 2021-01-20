        %% compute the permutation matrix P, un_mixing matrix un_M, mean c, bound b
        % t is the number of constrains, t'(t_ in code) is the total number of sources
        % m is the number of classes(num_c), n is the number of
        % samples(num_sample)
        
        if iscell(P_temp)
            n_group = length(P_temp);
        else
            n_group = size(P_temp,1);
        end
        
        % un_M is t' * t
        un_M = ICA_bounds{1}.un_mixing;
        for ii = 2:n_group
            un_M_add = ICA_bounds{ii}.un_mixing;
            un_M = blkdiag(un_M,un_M_add);
        end
        
        % c is t' * 1
        c = ICA_bounds{1}.mean;
        for ii = 2:n_group
            c_add = ICA_bounds{ii}.mean;
            c = [c; c_add];
        end
        
        % b is t' * 1
        b = ICA_bounds{1}.bounds_(:,2);
        for ii = 2:n_group
            b_add = ICA_bounds{ii}.bounds_(:,2);
            b = [b;b_add];
        end
               
        % P is t * m
        t = size(un_M,2);
        P = zeros(t,num_c);
        line_position = 1;
        for ii = 1:n_group
            if iscell(P_temp)
                loc = P_temp{ii} + 1;
            else
                loc = P_temp(ii,:) + 1;
            end
            for jj = 1:length(loc)
                P(line_position,int64(loc(jj))) = 1;
                line_position = line_position + 1;
            end
        end
        
        %% define the constant parameter matrices. please refer to our paper for the definitions of these notations
        F_bar = un_M * P;
        E_bar = [-ones(num_sample,1),ones(num_sample,1)];
        G3_bar = [2*b - 2*un_M*c + num_sample*F_bar*ones(num_c,1),...
            2*b + 2*un_M*c - num_sample*F_bar*ones(num_c,1)];
        M3 = E_bar * E_bar';
        M4 = F_bar' * F_bar;
        
        %% save the constant matrices to the unified struct 'constant_matrix'
        constant_matrix.F_bar = F_bar;
        constant_matrix.E_bar = E_bar;
        constant_matrix.G3_bar = G3_bar;
        constant_matrix.M3 = M3;
        constant_matrix.M4 = M4;