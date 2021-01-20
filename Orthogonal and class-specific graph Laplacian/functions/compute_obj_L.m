function [ obj_L ] = compute_obj_L( Z, options, PGD )
%% compute obj_L
%% parameter setting
if options.activation_func
    Z_prob = tanh(Z);
else
    Z_prob = Z;
end

[m, n] = size(PGD.Yp);
beta = options.beta;
if options.bound_orth
    lambda = PGD.lambda;
    rho = PGD.rho;
end

%% compute obj_L
obj_L = PGD.const + trace(Z_prob * PGD.M0');
for ii = 1:m
    obj_L = obj_L + beta/2 * PGD.e(ii,:) * Z_prob * PGD.L_x{ii} * Z_prob' * PGD.e(ii,:)';
end

if options.bound_CIB_2
    obj_L = obj_L + 1/2 * trace(Z_prob * PGD.M1 * Z_prob');
end

% this term always exist due to co-occurrence matrix (rather use: if options.bound_CIB_1
obj_L = obj_L + 1/2 * trace(PGD.M2 * (Z_prob * Z_prob'));

if options.bound_ICA
    obj_L = obj_L + 1/2 * trace( Z_prob * PGD.M1 * Z_prob' * PGD.M3 );
end

if options.bound_orth
    for ii = 1:m
        tem_orth = PGD.e(ii,:) * Z * PGD.D_half{ii} * ones(n,1);
        obj_L = obj_L + tem_orth * (lambda(ii) + 1/2*rho*tem_orth);
    end
end

end

