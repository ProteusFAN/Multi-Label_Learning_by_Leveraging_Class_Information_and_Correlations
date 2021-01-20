function [Z_final, Z_cell, obj_L, obj_Z, Ux_final, Uc_final, log] = MMIB(Y, edgeStructure, lambda_x, lambda_c, constant_matrix, options)

%% preprocessing 
[m, n] = size(Y);

% set the weight label matrix Yw (note that it is represented as Yp in our paper)
global positive_label_weight_vector
weight_matrix = ones(size(Y)); 
for ii = 1:m
    weight_matrix(ii, Y(ii,:)==1) = positive_label_weight_vector(ii);
end
Yw = double(weight_matrix .* Y ); 
const = trace(Yw * Y'); 

% Constant terms, A, B, Q, A_bar, B_bar, C_nar, D_bar, G2_bar, M1, M2
A = constant_matrix.A; 
B = constant_matrix.B; 
A_bar = constant_matrix.A_bar; 
B_bar = constant_matrix.B_bar; 
C_bar = constant_matrix.C_bar; 
D_bar = constant_matrix.D_bar; 
G1_bar = constant_matrix.G1_bar; 
G2_bar = constant_matrix.G2_bar;
M1 = constant_matrix.M1; 
M2 = constant_matrix.M2;

if options.ICA_valid
    E_bar = constant_matrix.E_bar;
    F_bar = constant_matrix.F_bar;
    G3_bar = constant_matrix.G3_bar;
    M3 = constant_matrix.M3;
    M4 = constant_matrix.M4;
end

if options.orth
    H_bar = constant_matrix.H_bar;
    M5 = constant_matrix.M5;
end

w_x_vec = lambda_x .* edgeStructure.instance.similarity; 
W_x_bar = repmat(w_x_vec, 1, m)'; % m * num_ex matrix 
w_c_vec = lambda_c .* edgeStructure.class.similarity;
W_c_bar = repmat(w_c_vec, 1, n); % num_ec * n matrix 

num_ex = length(edgeStructure.instance.edge_id); 
num_ec = length(edgeStructure.class.edge_id); 

%% initialization of Z0
% given_locations = (Y~=0);
% if isfield(options, 'initialization')
%     switch options.initialization
%         case 'random'
%             % generate Z randomly via proportion
%             n_pos = sum(Y == 1);
%             n_neg = sum(Y == -1);
%             n_all = n_pos + n_neg;
%             Z0 = randsrc(m,n,[1, -1; n_pos/n_all, n_neg/n_all]);
%             fprintf('random initialization of Z.\n');
%         case 'zero'
%             Z0 = zeros(m, n); 
%             fprintf('non-random initialization of Z.\n');
%     end
%     Z0(given_locations) = Y(given_locations); 
% else
%     Z0 = Y; %Z0(Z0 == 0) =-1; 
% end

Z0 = 2*rand(size(Y))-1;
%% initialization of other matricies 

Ux_old =full( abs(Z0 * A) );
Uc_old = full( abs(B * Z0) );

Lambda1_old = sparse(m, 2*num_ex+2); 
Lambda2_old = sparse(2*num_ec + 2, n);

Phi1_old = max(0, Ux_old * C_bar + G1_bar - Z0 * A_bar); 
Phi2_old = max(0, D_bar*Uc_old + G2_bar - B_bar * Z0);

rho = options.ADMM.rho_0;
rho_1 = rho;
rho_2 = rho;

if options.ICA_valid
    Lambda3_old = sparse(size(F_bar,1), 2);
    Phi3_old = max(0, G3_bar - F_bar * Z0 * E_bar);
    rho_3 = rho;
end

if options.orth
    Lambda4_old = sparse(m, 1);
    rho_4 = options.ADMM.rho_orth;
end

%% the objective function of initialized variables, L(Z, Ux, Uc, Lambda1, Lambda2, Phi1, Phi2)

obj_Z(1) = const - trace(Z0 * Yw') + sum( Ux_old * w_x_vec ) + sum( w_c_vec' * Uc_old); 

tem_matrix_1 = Z0 * A_bar - Ux_old * C_bar- G1_bar + Phi1_old; 
tem_matrix_2 = B_bar * Z0 - D_bar*Uc_old - G2_bar + Phi2_old;
aug_term_1 = trace( tem_matrix_1 * ( Lambda1_old + 0.5*rho_1.*tem_matrix_1 )' ); 
aug_term_2 = trace( tem_matrix_2 * ( Lambda2_old + 0.5*rho_2.*tem_matrix_2 )' );
obj_L(1) = obj_Z(1) + aug_term_1 + aug_term_2;

if options.ICA_valid
    tem_matrix_3 = F_bar * Z0 * E_bar - G3_bar + Phi3_old;
    aug_term_3 = trace( tem_matrix_3 * ( Lambda3_old + 0.5*rho_3.*tem_matrix_3 )' );
    obj_L(1) = obj_L(1) + aug_term_3;
end

if options.orth
    tem_matrix_4 = Z0 * H_bar;
    aug_term_4 = trace( tem_matrix_4 * ( Lambda4_old + 0.5*rho_4.*tem_matrix_4 )' );
    obj_L(1) = obj_L(1) + aug_term_4;
end
    
clear tem_matrix_1 tem_matrix_2 tem_matrix_3 tem_matrix_4 aug_term_1 aug_term_2 aug_term_3 aug_term_4
t = 1; 

%% start ADMM algorithm

Z_old = Z0; 
Z_cell{1} = Z_old;
max_iter = options.ADMM.max_iter_overall;
log = struct();

for iter = 1:max_iter
    
    %% update of Z^{t+1}
    M0 = - Yw + (Lambda1_old - rho_1 .* (Ux_old * C_bar + G1_bar - Phi1_old) ) * A_bar' +...
        B_bar' * (Lambda2_old - rho_2 .* (D_bar * Uc_old + G2_bar - Phi2_old) );
    
    if options.orth
        M0 = M0 + Lambda4_old *  H_bar';
        M1 = M1 + rho_4/rho_1 *M5;
    end
    
    if options.ICA_valid
        M0 = M0 + F_bar' * (Lambda3_old - rho_3 .* (G3_bar - Phi3_old)) * E_bar';
        [Z_new, ~, ~ ] = mlml_pgd_template(Z_old, options, M0, 0.5*rho_1.*M1 , 0.5*rho_2.*M2, 0.5*rho_3*M3, M4); 
    else
        [Z_new, ~, ~ ] = mlml_pgd_template(Z_old, options, M0, 0.5*rho_1.*M1 , 0.5*rho_2.*M2); 
    end
     
    %% update of U_x^{t+1}
    tem_matrix_x = Z_new * A_bar - G1_bar + Phi1_old; 
    Mx = W_x_bar - ( Lambda1_old + rho_1 .* tem_matrix_x) * C_bar'; 
    Ux_new = max(0, -(0.5/rho_1) .* Mx); 
    
    %% update of U_c^{t+1}
    tem_matrix_c = B_bar * Z_new -G2_bar + Phi2_old; 
    Mc = W_c_bar - D_bar' * ( Lambda2_old + rho_2 .* tem_matrix_c ); 
    Uc_new = max(0, -(0.5/rho_2) .* Mc); 
    
    %% update of Phi1_^{t+1} and Phi2_^{t+1} (and Phi3_^{t+1})
    tem_matrix_1 = Z_new * A_bar - Ux_new * C_bar - G1_bar; 
    tem_matrix_2 = B_bar * Z_new - D_bar * Uc_new - G2_bar;
    Phi1_new = max(0, -( Lambda1_old./rho_1 + tem_matrix_1 ) );
    Phi2_new = max(0, -( Lambda2_old./rho_2 + tem_matrix_2 ) );
    if options.ICA_valid
        tem_matrix_3 = F_bar * Z_new * E_bar - G3_bar;
        Phi3_new = max(0, -( Lambda3_old./rho_3 + tem_matrix_3 ) );
    end
    
    if options.orth
        tem_matrix_4 = Z_new * H_bar;
    end
    
    %% update of Lambda1_^{t+1} and Lambda2_^{t+1} (and Lambda3_^{t+1})
    Lambda1_new = Lambda1_old + 1.618.*rho_1.* ( Phi1_new + tem_matrix_1 ); 
    Lambda2_new = Lambda2_old + 1.618.*rho_2.* ( Phi2_new + tem_matrix_2 );
    if options.ICA_valid
        Lambda3_new = Lambda3_old + 1.618.*rho_3.* ( Phi3_new + tem_matrix_3 );
    end
    
    if options.orth
        Lambda4_new = Lambda4_old + 1.618.*rho_4.* tem_matrix_4;
    end
    
    %% compute the objective function of L(Z, Ux, Uc, Lambda1, Lambda2, Phi1, Phi2)
    if ((iter <= 8) && (mod(iter,2) == 0)) || ((iter > 8) && (mod(iter,10) == 0))
        log.iter(t) = iter;
        obj_Z(t+1) = const - trace(Yw' * Z_new) + sum( Ux_new * w_x_vec ) + sum( w_c_vec' * Uc_new); 
        
        tem_matrix_1_1 = tem_matrix_1 + Phi1_new; 
        tem_matrix_2_2 = tem_matrix_2 + Phi2_new;
        aug_term_1 = trace( tem_matrix_1_1 * ( Lambda1_new + 0.5*rho_1.*tem_matrix_1_1 )' ); 
        aug_term_2 = trace( ( Lambda2_new + 0.5*rho_2.*tem_matrix_2_2 ) * tem_matrix_2_2' );
        obj_L(t+1) = obj_Z(t+1) + aug_term_1 + aug_term_2;

        if options.ICA_valid
            tem_matrix_3_3 = tem_matrix_3 + Phi3_new;
            aug_term_3 = trace( tem_matrix_3_3 * ( Lambda3_new + 0.5*rho_3.*tem_matrix_3_3 )' );
            obj_L(t+1) = obj_L(t+1) + aug_term_3; 
        end
        
        if options.orth
            tem_matrix_4_4 = tem_matrix_4;
            aug_term_4 = trace( tem_matrix_4_4 * ( Lambda4_new + 0.5*rho_4.*tem_matrix_4_4 )' );
            obj_L(t+1) = obj_L(t+1) + aug_term_4;
        end
        
        Z_cell{t+1} = Z_new;

%         dist1 = norm(tem_matrix_1_1,'fro');  
%         dist2 = norm(tem_matrix_2_2,'fro');
%         if options.ICA_valid
%             dist3 = norm(tem_matrix_3_3,'fro');
%             text_display = sprintf('iter=%d, dist1=%f, dist2=%f, dist3=%f, rho1=%f, rho2=%f, rho3=%f, obj_L=%f, obj_Z=%f.',iter,dist1, dist2, dist3, rho_1, rho_2, rho_3, obj_L(t+1), obj_Z(t+1));
%         else
%             text_display = sprintf('iter=%d, dist1=%f, dist2=%f, rho1=%f, rho2=%f, obj_L=%f, obj_Z=%f.',iter,dist1, dist2, rho_1, rho_2, obj_L(t+1), obj_Z(t+1));
%         end
%         disp(text_display)
        
        diff_Ux_Uc = sumabs(tem_matrix_1(:,1:end-2) > 0) + sumabs(tem_matrix_2(1:end-2,:) > 0);
        diff_CIB_1 = sumabs( tem_matrix_2(end-1:end,:) > 0 );
        diff_CIB_2 = sumabs( tem_matrix_1(:,end-1:end) > 0 );
        scale_Ux_Uc = sumabs( max(tem_matrix_1(:,1:end-2),0) ) + sumabs( max(tem_matrix_2(1:end-2,:),0) );
        scale_CIB_1 = sumabs( max(tem_matrix_2(end-1:end,:), 0) );
        scale_CIB_2 = sumabs( max(tem_matrix_1(:,end-1:end), 0) );
        n_Ux_Uc = length( tem_matrix_1(:,1:end-2) ) + length( tem_matrix_2(1:end-2,:) );
        n_CIB_1 = length( tem_matrix_2(end-1:end,:) );
        n_CIB_2 = length( tem_matrix_1(:,end-1:end) );
        
        log.diff_Ux_Uc(t) = diff_Ux_Uc; log.scale_Ux_Uc(t) = scale_Ux_Uc; log.n_Ux_Uc(t) = n_Ux_Uc;
        log.diff_CIB_1(t) = diff_CIB_1; log.scale_CIB_1(t) = scale_CIB_1; log.n_CIB_1(t) = n_CIB_1;
        log.diff_CIB_2(t) = diff_CIB_2; log.scale_CIB_2(t) = scale_CIB_2; log.n_CIB_2(t) = n_CIB_2;
        
        str1 = sprintf('iter = %d, obj_L = %f, obj_Z = %f', iter ,obj_L(t+1), obj_Z(t+1));
        str2 = sprintf('diff_Ux_Uc = %d/%d, diff_CIB_1 = %d/%d, diff_CIB_2 = %d/%d', diff_Ux_Uc, n_Ux_Uc, diff_CIB_1, n_CIB_1, diff_CIB_2, n_CIB_2);
        str3 = sprintf('scale_Ux_Uc = %f, scale_CIB_1 = %f, scale_CIB_2 = %f', scale_Ux_Uc, scale_CIB_1, scale_CIB_2);
        str4 = sprintf('rho1 = %f, rho2 = %f', rho_1, rho_2);
        
        if options.ICA_valid
            diff_ICA = sumabs( tem_matrix_3 > 0 );
            scale_ICA = sumabs( max(tem_matrix_3, 0) );
            n_ICA = length( tem_matrix_3 );
            log.diff_ICA(t) = diff_ICA; log.scale_ICA(t) = scale_ICA; log.n_ICA(t) = n_ICA;
            
            str2 = sprintf([str2, ', diff_ICA = %d/%d'], diff_ICA, n_ICA);
            str3 = sprintf([str3, ', scale_ICA = %f'], scale_ICA);
            str4 = sprintf([str4, ', rho3 = %f'], rho_3);
        end
        
        if options.orth
            diff_orth = sumabs( tem_matrix_4 > 0 );
            scale_orth = sumabs( tem_matrix_4 );
            n_orth = length( tem_matrix_4 );
            log.diff_orth(t) = diff_orth; log.scale_orth(t) = scale_orth; log.n_orth(t) = n_orth;
            
            str2 = sprintf([str2, ', diff_orth = %d/%d'], diff_orth, n_orth);
            str3 = sprintf([str3, ', scale_orth = %f'], scale_orth);
            str4 = sprintf([str4, ', rho4 = %f'], rho_4);
        end
        
        fprintf([str1,'.\n']);
        fprintf([str2,'.\n']);
        fprintf([str3,'.\n']);
        fprintf([str4,'.\n\n']);
        
%         if options.ICA_valid
%             if( iter>20 && scale_Ux_Uc<1e-8 && scale_CIB_1<1e-8 && scale_CIB_2<1e-8 && scale_ICA<1e-8 ) 
%                 break; 
%             else
%                 t = t+1;
%             end
%         else
%             if( iter>20 && scale_Ux_Uc<1e-8 && scale_CIB_1<1e-8 && scale_CIB_2<1e-8 ) 
%                 break; 
%             else
%                 t = t+1;
%             end
%         end
        t = t+1;
    end
    
    %% change the variable states for the next iteration
    Z_old = Z_new; 
    Ux_old = Ux_new; 
    Uc_old = Uc_new; 
    Phi1_old = Phi1_new;
    Phi2_old = Phi2_new;
    Lambda1_old = Lambda1_new;
    Lambda2_old = Lambda2_new;
    if options.ICA_valid
        Phi3_old = Phi3_new;
        Lambda3_old = Lambda3_new;
    end
    
    if options.orth
        Lambda4_old = Lambda4_new;
    end
    
    if(~mod(iter,options.ADMM.rho_gap))
        rho = min(1e8, rho * options.ADMM.rho_rate);
        rho_1 = 1.*rho;
        rho_2 = 1.*rho;
        if options.ICA_valid
            rho_3 = 1.*rho;
        end
        if options.orth
            rho_4 = min(1e8, rho_4 * options.ADMM.rho_rate);
        end
    end   
    
end
Z_final = Z_new;     
Ux_final = Ux_new;    
Uc_final = Uc_new;    


function [Z_final, Z_cell, obj, alpha] = mlml_pgd_template(Z0, options, A, B, C, varargin)

% varargin consists of D and E, if options.ICA_valid is true
% min_Z { tr(A' * Z) + tr(Z B Z') + tr(Z' C Z) + tr(D Z' E Z) }, st. -1 <= Z <= 1

if options.ICA_valid
    D = varargin{1};
    E = varargin{2};
end

max_iter = options.PGD.max_iter; 
gap_compute = options.PGD.gap_compute;
rate_step = options.PGD.rate_step;
alpha_rate = options.PGD.alpha_rate;

B = full(B); C = full(C);

obj = zeros(1, max_iter);
if options.ICA_valid
    obj(1) = obj_compute(Z0, A, B, C, D, E);
else
    obj(1) = obj_compute(Z0, A, B, C);
end

alpha = zeros(1, max_iter);

Z_old = Z0;
Z_cell = cell(1,max_iter);
Z_cell{1} = Z_old;

for i=1:max_iter
    
    % step 1, gradient
    if options.ICA_valid
        Z_gradient = A + 2.* Z_old *  B  + 2.* C * Z_old + 2.* E * Z_old * D; 
    else
        Z_gradient = A + 2.* Z_old * full( B ) + 2.* full(C) * Z_old; 
    end
    
    % step 2, step size by exact line search
    if ((i-1)/gap_compute) == fix((i-1)/gap_compute)
        if options.ICA_valid
            alpha(i) = ( step_size_compute(Z_old, Z_gradient, A, B, C, D, E) ) / (alpha_rate*i);
        else
            alpha(i) = ( step_size_compute(Z_old, Z_gradient, A, B, C) ) / (alpha_rate*i);
        end
    else
        alpha(i) = alpha(i-1) * rate_step;
    end
    
    % step 3, update of Z
    Z_new = Z_old - alpha(i) .* Z_gradient;
    
    % step 4, projection to [-1,1] 
%     Z_new = min(1,  max(-1, Z_new) );

    for ii = 1:size(Z_new, 1)
        Z_new(ii,:) = Z_new(ii,:).* sqrt(size(Z_new, 2))./ sqrt(Z_new(ii,:) * Z_new(ii,:)');
    end
    
%     Z_new = Z_new.* sqrt(size(Z_new, 1)*size(Z_new, 2)) ./ norm(Z_new,'fro');
    
    Z_cell{i+1} = Z_new;
    
    % step 5, objective function and check convergence
%     if ((i)/5) == fix((i)/5)
%         if options.ICA_valid
%             obj(i+1) = obj_compute(Z_new, A, B, C, D, E);
%         else
%             obj(i+1) = obj_compute(Z_new, A, B, C);
%         end
%         obj_diff=abs(obj(i+1)-obj(i+1 - 1))/abs(obj(i));
%          if (obj_diff<1*10^(-8))
%            break;
%          end    
%     end
    Z_old = Z_new;
    
end

Z_final = Z_new;
obj( obj == 0) = [];



function obj = obj_compute(Z, A, B, C, varargin)

% varargin consists of D and E, if options.ICA_valid is true
if length(varargin) == 2
    D = varargin{1};
    E = varargin{2};
    obj = trace(Z * A') + trace( (Z * full(B)) * Z') + trace( full(C) * (Z * Z') ) +...
        trace( D * Z' * E * Z );
else
    obj = trace(Z * A') + trace( (Z * full(B)) * Z') + trace( full(C) * (Z * Z') );
end

function alpha = step_size_compute(Z, Z_gradient, A, B, C, varargin)

% varargin consists of D and E, if options.ICA_valid is true
% the step size by exact line search for tr(A' * Z) + tr(Z B Z') + tr(Z' C Z) + tr(D Z' E Z)

if length(varargin) == 2
    D = varargin{1};
    E = varargin{2};
    numerator = 0.5 * trace( Z_gradient * A') + trace( (Z * B) * Z_gradient' ) +...
        trace( C * (Z_gradient * Z') ) + ...
        0.5 * ( trace( D * Z_gradient' * E * Z) + trace( D * Z' * E * Z_gradient) );
    dom = trace( (Z_gradient * B) * Z_gradient' ) + trace( C * (Z_gradient * Z_gradient') ) +...
        trace( D * Z_gradient' * E * Z_gradient );
else
    numerator = 0.5 * trace( Z_gradient * A') + trace( (Z * B) * Z_gradient' ) + trace( C * (Z_gradient * Z') );
    dom = trace( (Z_gradient * B) * Z_gradient' ) + trace( C * (Z_gradient * Z_gradient') );
end

alpha = numerator / dom; 
