function [ Z_new, log_pgd, nonUpdate ] = pgd_template( Z_old, options, PGD )
% PGD algorithm for ADMM

%% ==== parameter setting ====
[m, n] = size(Z_old);

max_iter = options.PGD_max_iter; 
gap_compute = options.PGD_gap_compute;
rate_step = options.PGD_rate_step;
alpha_rate = options.PGD_alpha_rate;

beta = options.beta;
if options.bound_orth
    rho = PGD.rho;
    lambda = PGD.lambda;
end

%% ==== PGD algorithm ====
alpha = options.PGD_alpha_init;
obj_Z = compute_obj_Z(Z_old, options, PGD);
obj_L = compute_obj_L(Z_old, options, PGD);
Z_set{1} = Z_old;
if options.activation_func
    Z_old_prob = tanh(Z_old);
else
    Z_old_prob = Z_old;
end
Z_gradient_set{1} = zeros(size(Z_old_prob));

for i = 1:max_iter
    % ---- step 1, gradient ----   
    Z_gradient = PGD.M0;
    for ii = 1:m
        Z_gradient = Z_gradient + beta * PGD.e(ii,:)'*PGD.e(ii,:) * Z_old * PGD.L_x{ii};
    end
    
    if options.bound_CIB_2
        Z_gradient = Z_gradient + Z_old_prob * PGD.M1;
    end
    
    % this term always exist due to co-occurrence matrix (rather use: if options.bound_CIB_1
    Z_gradient = Z_gradient + PGD.M2 * Z_old_prob;
    
    if options.bound_ICA
        Z_gradient = Z_gradient + PGD.M3 * Z_old_prob * PGD.M1;
    end
    
    % when activation function is implements, chain rule is used.
    if options.activation_func
        Z_grad_suffix = 1 - Z_old_prob.^2;
        Z_gradient = Z_gradient.* Z_grad_suffix;
    end
    
    if options.bound_orth
        for ii = 1:m
            tem_vec =  PGD.D_half{ii} * ones(n,1);
            tem_matrix = tem_vec * tem_vec';
            Z_gradient = Z_gradient +...
                lambda(ii) * PGD.e(ii,:)' * tem_vec' +...
                rho * PGD.e(ii,:)'*PGD.e(ii,:) * Z_old * tem_matrix;
        end
    end
    
    % ---- step 2, step size ----
    if (i-1)/gap_compute == fix((i-1)/gap_compute)
        [alpha(end+1), nonUpdate] = step_size_compute(Z_old, Z_gradient, options, PGD, alpha);
        alpha(end) = alpha(end) / alpha_rate;
    else
        alpha(end+1) = alpha(i) * rate_step;
    end
    
    % ---- step 3, update of Z ----
    Z_new = Z_old - alpha(i+1) * Z_gradient;
    
    % ---- compute obj_Z and obj_L ----
    if ((i <= 4) && (mod(i,1) == 0)) || ((i > 4) && (mod(i,4) == 0))
        obj_Z(end+1) = compute_obj_Z(Z_new, options, PGD);
        obj_L(end+1) = compute_obj_L(Z_new, options, PGD);
        Z_set{end+1} = Z_new;
        Z_gradient_set{end+1} = Z_gradient;
    end
    
    Z_old = Z_new;
    
    % ---- stop iteration if stepsize is too small
    if alpha(end) < 1e-5
        break
    end
end

% record pgd_template results
log_pgd.obj_Z = obj_Z;
log_pgd.obj_L = obj_L;
log_pgd.alpha = alpha;
log_pgd.Z_set = Z_set;
log_pgd.Z_gradient_set = Z_gradient_set;

end