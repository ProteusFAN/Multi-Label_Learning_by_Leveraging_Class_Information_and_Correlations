function [ obj_Z ] = compute_obj_Z( Z, options, PGD )
%% compute obj_Z
if options.activation_func
    Z_prob = tanh(Z);
else
    Z_prob = Z;
end
[m, ~] = size(PGD.Yp);

obj_Z = trace((PGD.Y - Z_prob) * PGD.Yp') + options.gamma/2 * trace( PGD.L_c * (Z_prob * Z_prob'));
for ii = 1:m
    tem_vec = PGD.e(ii,:) * Z_prob;
    obj_Z = obj_Z + options.beta/2 * tem_vec * PGD.L_x{ii} * tem_vec';
end

end