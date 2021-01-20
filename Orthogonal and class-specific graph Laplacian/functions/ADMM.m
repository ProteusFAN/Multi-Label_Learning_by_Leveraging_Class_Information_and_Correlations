function [ Z_final, Z_cell, obj_Z, obj_L, log ] = ADMM( data_info, options )
% ADMM for multi-label learning

%% ==== computation of parameters ====
% ==== compute Y, P, Yp and tr(Yp'*Y) ====
Y = [data_info.label_train, zeros(size(data_info.label_test))];
[m, n] = size(Y);

label_positive_sum = sum(Y == 1, 2);
label_negative_sum = sum(Y == -1, 2);
label_weight = sqrt(label_negative_sum./label_positive_sum);

P = zeros(size(Y));
for ii = 1:data_info.num_c
    P(ii, Y(ii,:) == 1) = label_weight(ii);
    P(ii, Y(ii,:) == -1) = 1/label_weight(ii);
end
Yp = double(P.* Y);
const = trace(Y * Yp');

% ==== compute A, B, G1, G2 and C, G3 (ICA) and e, D_half (orthogonality) ====
e = eye(m);

if options.bound_CIB_2
    A = [-ones(n, 1), ones(n, 1)];
    G1 = [-(2*options.constrain.CIB_2{1} - n), 2*options.constrain.CIB_2{2} - n];
end

if options.bound_CIB_1
    G2 = [-(2*options.constrain.CIB_1{1} - m); 2*options.constrain.CIB_1{2} - m];
    B = [-ones(1, m); ones(1, m)];
end

if options.bound_ICA
%     C = ;
%     G3 = ;
end

if options.bound_orth
    D_half = cell(m, 1);
    for ii = 1:m
        D_half{ii} = sparse(diag(sqrt(sum(data_info.W_x{ii},1))));
    end
end

%% ==== initialization ====
% ==== Z_start ====
given_locations = Y ~= 0;   % assume no missing labels in training data
switch options.init
    case 1 % zero
        Z_start = options.init_amplifier * Y;
    case 2 % random
        Z_start = 1/5 * (2 * rand(size(Y)) - 1 );  % use Uniform(-1/5, 1/5) to initialize test label
        Z_start(given_locations) = options.init_amplifier * Y(given_locations);
    case 3 % allZero
        Z_start = zeros(size(Y));
    case 4 % allRandom
        Z_start = 1/5 * (2 * rand(size(Y)) - 1 );  % intialize randomly with small magnitude
end

if options.activation_func
    Z_start_prob = tanh(Z_start);
else
    Z_start_prob = Z_start;
end

% ==== other parameters in ADMM ====
beta = options.beta;
gamma = options.gamma;
Rho = options.ADMM_rho;

if options.bound_CIB_2
    Lambda1_old = zeros(m, 2);
    Phi1_old = max(0, G1 - Z_start_prob * A);
end

if options.bound_CIB_1
    Lambda2_old = zeros(2, n);
    Phi2_old = max(0, G2 - B * Z_start_prob);
end

if options.bound_ICA
    Lambda3_old = zeros(size(C, 1), 2);
    Phi3_old = max(0, G3 - C * Z_start_prob * A);
end

if options.bound_orth
    lambda_old = zeros(m, 1);
    rho = options.ADMM_rho_orth;
end

% ==== recording object function at intialization ====
% original object function 
obj_Z(1) = const - trace(Z_start_prob * Yp') +...
    gamma/2 * trace(data_info.L_c * (Z_start_prob * Z_start_prob'));
for ii = 1:m
    obj_Z(1) = obj_Z(1) +...
        beta/2 * e(ii,:) * Z_start_prob * data_info.L_x{ii} * Z_start_prob' * e(ii,:)';
end

% augmented lagrange
obj_L(1) = obj_Z(1);

if options.bound_CIB_2
    tem_matrix_1 = Z_start_prob * A - G1 + Phi1_old;
    aug_term_1 = trace( ( Lambda1_old + Rho/2 * tem_matrix_1 )' * tem_matrix_1 );
    obj_L(1) = obj_L(1) + aug_term_1;
end

if options.bound_CIB_1
    tem_matrix_2 = B * Z_start_prob - G2 + Phi2_old;
    aug_term_2 = trace( ( Lambda2_old + Rho/2 * tem_matrix_2 )' * tem_matrix_2 );
    obj_L(1) = obj_L(1) + aug_term_2;
end

if options.bound_ICA
    tem_matrix_3 = C * Z_start_prob * A - G3 + Phi3_old;
    aug_term_3 = trace( ( Lambda3_old + Rho/2 * tem_matrix_3 )' * tem_matrix_3 );
    obj_L(1) = obj_L(1) + aug_term_3;
end

if options.bound_orth
    for ii = 1:m
        tem_orth = e(ii,:) * Z_start * D_half{ii} * ones(n,1);
        obj_L(1) = obj_L(1) + tem_orth * (lambda_old(ii) + rho/2 * tem_orth); 
    end
end

%% ==== ADMM algorithms ====
t = 1;
Z_old = Z_start;
Z_cell{1} = Z_old;
log = struct();

for iter = 1:options.ADMM_max_iter
    % ---- update of Z{t+1} by PGD algorithm ----
    % compute some parameters and save in PGD for passing parameters
    PGD.Yp = Yp;
    PGD.Y = Y;
    PGD.L_x = data_info.L_x;
    PGD.L_c = data_info.L_c;
    PGD.e = e;
    PGD.M0 = sparse(-Yp);
    PGD.M2 = gamma * PGD.L_c;
    PGD.const = const;
    
    if options.bound_CIB_2
        PGD.M0 = PGD.M0 + sparse( (Lambda1_old + Rho * (Phi1_old - G1)) * A' );
        PGD.M1 = Rho * sparse(A * A');
        PGD.const = PGD.const + trace( (Lambda1_old + Rho/2 *(Phi1_old - G1))' * (Phi1_old - G1) );
    end
    
    if options.bound_CIB_1
        PGD.M0 = PGD.M0 + sparse( B' * (Lambda2_old + Rho * (Phi2_old - G2)) );
        PGD.M2 = PGD.M2 + Rho * sparse(B' * B);
        PGD.const = PGD.const + trace( (Lambda2_old + Rho/2 *(Phi2_old - G2)) * (Phi2_old - G2)');
    end

    if options.bound_ICA
        PGD.M0 = PGD.M0 + sparse(C' * (Lambda3_old + Rho *(Phi3_old - G3)) * A');
        PGD.M3 = sparse(C' * C);  % the scalar should be Rho_3/Rho_1 while Rho_1 = Rho_3 in this case.
        PGD.const = PGD.const + trace( (Phi3_old - G3)' * (Lambda3_old + Rho/2 *(Phi3_old - G3)));         
    end
    
    if options.bound_orth
        PGD.D_half = D_half;
        PGD.lambda = lambda_old;
        PGD.rho = rho;
    end
    
    % PGD algorithm
    [ Z_new, log_pgd, nonUpdate ] = pgd_template(Z_old, options, PGD);
    
    % if Z does not update anymore, stop iteration.
    if nonUpdate
        log.stop_iter = iter - 1;
        break
    end
    
    if options.activation_func
        Z_new_prob = tanh(Z_new);
    else
        Z_new_prob = Z_new;
    end
    
    % ---- update of Phi ----
    if options.bound_CIB_2
        Phi1_new = max(0, G1 - Z_new_prob * A - Lambda1_old/Rho);
    end
    
    if options.bound_CIB_1
        Phi2_new = max(0, G2 - B * Z_new_prob - Lambda2_old/Rho);
    end
    
    if options.bound_ICA
        Phi3_new = max(0, G3 - C * Z_new_prob * A - Lambda3_old/Rho);
    end
    
    % ---- update of Lambda (lambda)----
    if options.bound_CIB_2
        Lambda1_new = Lambda1_old + 1.618 * Rho * (Z_new_prob * A - G1 + Phi1_new);
    end
    
    if options.bound_CIB_1
        Lambda2_new = Lambda2_old + 1.618 * Rho * (B * Z_new_prob - G2 + Phi2_new);
    end
    
    if options.bound_ICA
        Lambda3_new = Lambda3_old + 1.618 * Rho * (C * Z_new_prob * A - G3 + Phi3_new);
    end
    
    if options.bound_orth
        lambda_new = zeros(size(lambda_old));
        for ii = 1:m
            lambda_new(ii) = lambda_old(ii) +...
                1.618 * rho * e(ii,:) * Z_new *  D_half{ii} * ones(n,1);
        end
    end
    
    % ---- recording object function in each iteration ----
    if ((iter <= 4) && (mod(iter,1) == 0)) || ((iter > 4) && (mod(iter,10) == 0))
        fprintf('\nIter: %d/%d.\n', iter, options.ADMM_max_iter);
        
        % original object function 
        obj_Z(t+1) = const - trace(Z_new_prob * Yp') +...
            gamma/2 * trace(data_info.L_c * (Z_new_prob * Z_new_prob'));
        for ii = 1:m
            obj_Z(t+1) = obj_Z(t+1) +...
                beta/2 * e(ii,:) * Z_new_prob * data_info.L_x{ii} * Z_new_prob' * e(ii,:)';
        end
        fprintf('obj_Z: %f.\n', obj_Z(t+1));
        
        % augmented lagrange
        obj_L(t+1) = obj_Z(t+1);
        if options.bound_CIB_2
            tem_matrix_1 = Z_new_prob * A - G1 + Phi1_new;
            aug_term_1 = trace( ( Lambda1_new + Rho/2 * tem_matrix_1 )' * tem_matrix_1 );
            obj_L(t+1) = obj_L(t+1) + aug_term_1;
        end
        
        if options.bound_CIB_1
            tem_matrix_2 = B * Z_new_prob - G2 + Phi2_new;
            aug_term_2 = trace( ( Lambda2_new + Rho/2 * tem_matrix_2 )' * tem_matrix_2 );
            obj_L(t+1) = obj_L(t+1) + aug_term_2;
        end

        if options.bound_ICA
            tem_matrix_3 = C * Z_new_prob * A - G3 + Phi3_new;
            aug_term_3 = trace( ( Lambda3_new + Rho/2 * tem_matrix_3 )' * tem_matrix_3 );
            obj_L(t+1) = obj_L(t+1) + aug_term_3;
        end

        if options.bound_orth
            for ii = 1:m
                tem_orth = e(ii,:) * Z_new * D_half{ii} * ones(n,1);
                obj_L(t+1) = obj_L(t+1) + tem_orth * (lambda_new(ii) + rho/2 * tem_orth); 
            end
        end
        fprintf('obj_L: %f.\n', obj_L(t+1));
        
%         fprintf('tem_matrix_1: %f, tem_matrix_2: %f. \n', sumabs(tem_matrix_1), sumabs(tem_matrix_2));

        % log
        log.iter(t) = iter;
        log.pgd(t) = log_pgd;
        
        if options.bound_CIB_1
            diff_CIB_1 = sumabs( tem_matrix_2 > 0 );
            scale_CIB_1 = sumabs( max(tem_matrix_2, 0) );
            n_CIB_1 = length( tem_matrix_2 );
            log.diff_CIB_1(t) = diff_CIB_1;
            log.scale_CIB_1(t) = scale_CIB_1;
            log.n_CIB_1 = n_CIB_1;
            fprintf('diff_CIB_1: %d/%d, scale_CIB_1: %f.\n', diff_CIB_1, n_CIB_1, scale_CIB_1);
        end
        
        if options.bound_CIB_2
            diff_CIB_2 = sumabs( tem_matrix_1 > 0 );
            scale_CIB_2 = sumabs( max(tem_matrix_1, 0) );
            n_CIB_2 = length( tem_matrix_1 );
            log.diff_CIB_2(t) = diff_CIB_2;
            log.scale_CIB_2(t) = scale_CIB_2;
            log.n_CIB_2 = n_CIB_2;
            fprintf('diff_CIB_2: %d/%d, scale_CIB_2: %f.\n', diff_CIB_2, n_CIB_2, scale_CIB_2);
        end
        
        if options.bound_ICA
            diff_ICA = sumabs( tem_matrix_3 > 0 );
            scale_ICA = sumabs( max(tem_matrix_3, 0) );
            n_ICA = length( tem_matrix_3 );
            log.diff_ICA(t) = diff_ICA;
            log.scale_ICA(t) = scale_ICA;
            log.n_ICA = n_ICA;
            fprintf('diff_ICA: %d/%d, scale_ICA: %f.\n', diff_ICA, n_ICA, scale_ICA);
        end
        
        if options.bound_orth
            orth_prod = zeros(1,m);
            diff_orth = 0;
            scale_orth = 0;
            n_orth = m;
            for ii = 1:m
                tem_orth = e(ii,:) * Z_new *  D_half{ii} * ones(n,1);
                diff_orth = diff_orth + (tem_orth ~= 0);
                scale_orth = scale_orth + abs(tem_orth);
                orth_prod(ii) = tem_orth;
            end
            log.diff_orth(t) = diff_orth;
            log.scale_orth(t) = scale_orth;
            log.n_orth(t) = n_orth;
            fprintf('diff_orth: %d/%d, scale_orth: %f.\n', diff_orth, n_orth, scale_orth);
            disp(['orth_prod: ', num2str(orth_prod)]);
        end
        
        Z_cell{t+1} = Z_new;
        t = t+1;
    end
    
    % ---- change the variable states for the next iteration ----
    Z_old = Z_new;
    
    if options.bound_CIB_2
        Phi1_old = Phi1_new;
        Lambda1_old = Lambda1_new;
    end
    
    if options.bound_CIB_1
        Phi2_old = Phi2_new;
        Lambda2_old = Lambda2_new;
    end
    
    if options.bound_ICA
        Phi3_old = Phi3_new;
        Lambda3_old = Lambda3_new;
    end
    
    if options.bound_orth
        lambda_old = lambda_new;
    end
    
    if (~mod(iter,options.ADMM_rho_gap))
        Rho = min(1e8, Rho * options.ADMM_rho_rate);
        if options.bound_orth
            rho = min(1e8, rho * options.ADMM_rho_rate);
            PGD.rho = rho;
        end
    end
end

Z_final = Z_new;

end


