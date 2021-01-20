function [Z_final, Z_cell, obj_L, obj_Z, Ux_final, Uc_final] = MMIB(Y,edgeStructure, lambda_x, lambda_c, constant_matrix, options)

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

w_x_vec = lambda_x .* edgeStructure.instance.similarity; 
W_x_bar = repmat(w_x_vec, 1, m)'; % m x num_ex matrix 
w_c_vec = lambda_c .* edgeStructure.class.similarity;
W_c_bar = repmat(w_c_vec, 1, n); % num_ec x n matrix 

num_ex = length(edgeStructure.instance.edge_id); 
num_ec = length(edgeStructure.class.edge_id); 

%% initialization of Z0
given_locations = (Y~=0);
if isfield(options, 'initialization')
    switch options.initialization
        case  'random'
            Z0 = rand(m, n);
            
        case 'zero'
            Z0 = zeros(m, n); 
    end
    Z0(given_locations) = Y(given_locations); 
else
    Z0 = Y; %Z0(Z0 == 0) =-1; 
end

%% initialization of other matricies 

Ux_old =full( abs(Z0 * A) );
Uc_old = full( abs(B * Z0) );

Lambda1_old = sparse(m, 2*num_ex+2); 
Lambda2_old = sparse(2*num_ec + 2, n); 

Phi1_old = max(0, Ux_old * C_bar + G1_bar - Z0 * A_bar); 
Phi2_old = max(0, D_bar*Uc_old + G2_bar - B_bar * Z0); 

rho = options.ADMM.rho_0;
rho_1 = 1.*rho;
rho_2 = rho;

%% the objective function of initialized variables, L(Z, Ux, Uc, Lambda1, Lambda2, Phi1, Phi2)
obj_Z(1) = const - trace(Z0 * Yw') + sum( Ux_old * w_x_vec ) + sum( w_c_vec' * Uc_old); 

tem_matrix_1 = Z0 * A_bar - Ux_old * C_bar- G1_bar + Phi1_old; 
tem_matrix_2 = B_bar * Z0 - D_bar*Uc_old - G2_bar + Phi2_old; 
aug_term_1 = trace( tem_matrix_1 * ( Lambda1_old + 0.5*rho_1.*tem_matrix_1 )' ); 
aug_term_2 = trace( tem_matrix_2 * ( Lambda2_old + 0.5*rho_2.*tem_matrix_2 )' );
obj_L(1) = obj_Z(1) + aug_term_1 + aug_term_2;
clear tem_matrix_1 tem_matrix_2 aug_term_1 aug_term_2
t = 1; 


%% start ADMM algorithm

Z_old = Z0; 
Z_cell{1} = Z_old;
max_iter = options.ADMM.max_iter_overall;
for iter =1:max_iter
    
    %% update of Z^{t+1} 
    M0 = - Yw + (Lambda1_old - rho_1 .* (Ux_old * C_bar + G1_bar - Phi1_old) ) * A_bar' +  B_bar' * ( Lambda2_old -rho_2 .* ( D_bar * Uc_old + G2_bar - Phi2_old ) );
    [Z_new, ~, obj_Lz ] = mlml_pgd_template(M0, 0.5*rho_1.*M1 , 0.5*rho_2.*M2, Z_old, options); 
    
    %% update of U_x^{t+1}
    tem_matrix_x = Z_new * A_bar - G1_bar + Phi1_old; 
    Mx = W_x_bar - ( Lambda1_old + rho_1 .* tem_matrix_x) * C_bar'; 
    Ux_new = max(0, -(0.5/rho_1) .* Mx); 
    
    %% update of U_c^{t+1}
    tem_matrix_c = B_bar * Z_new -G2_bar + Phi2_old; 
    Mc = W_c_bar - D_bar' * ( Lambda2_old + rho_2 .* tem_matrix_c ); 
    Uc_new = max(0, -(0.5/rho_2) .* Mc); 
    
    %% update of Phi1_^{t+1} and Phi2_^{t+1}  
    tem_matrix_1 = Z_new * A_bar - Ux_new * C_bar - G1_bar; 
    tem_matrix_2 = B_bar * Z_new - D_bar * Uc_new -G2_bar;
    Phi1_new = max(0, -( Lambda1_old./rho_1 + tem_matrix_1 ) ); 
    Phi2_new = max(0, -( Lambda2_old./rho_2 + tem_matrix_2 ) ); 

    
    %% update of Lambda1_^{t+1} and Lambda2_^{t+1}
    Lambda1_new = Lambda1_old + 1.618.*rho_1.* ( Phi1_new + tem_matrix_1); 
    Lambda2_new = Lambda2_old + 1.618.*rho_2.* ( Phi2_new + tem_matrix_2);  
    
    %% compute the objective function of L(Z, Ux, Uc, Lambda1, Lambda2, Phi1, Phi2)
    if ((iter)/10) == fix((iter)/10) % compute the objective function once in every 10 iterations
        obj_Z(t+1) = const - trace(Yw' * Z_new) + sum( Ux_new * w_x_vec ) + sum( w_c_vec' * Uc_new); 
        
        tem_matrix_1_1 = tem_matrix_1 + Phi1_new; 
        tem_matrix_2_2 = tem_matrix_2 + Phi2_new; 

        aug_term_1 = trace( tem_matrix_1_1 * ( Lambda1_new + 0.5*rho_1.*tem_matrix_1_1 )' ); 
        aug_term_2 = trace( ( Lambda2_new + 0.5*rho_2.*tem_matrix_2_2 ) * tem_matrix_2_2' );
        obj_L(t+1) = obj_Z(t+1) + aug_term_1 + aug_term_2; 
        
        Z_cell{t+1} = Z_new;

        dist1 = norm(tem_matrix_1_1,'fro');  
        dist2 = norm(tem_matrix_2_2,'fro'); 
        text_display = sprintf('iter=%d, dist1=%f, dist2=%f, rho1=%f, rho2=%f, obj_L=%f, obj_Z=%f\n',iter,dist1, dist2, rho_1, rho_2, obj_L(t+1), obj_Z(t+1));
        disp(text_display)
        
        if( iter>50 && dist1<1e-8 && dist2<1e-8) 
            break; 
        else
            t = t+1;
        end         
    end
    
    %% change the variable states for the next iteration
    Z_old = Z_new; 
    Ux_old = Ux_new; 
    Uc_old = Uc_new; 
    Phi1_old = Phi1_new;
    Phi2_old = Phi2_new;
    Lambda1_old = Lambda1_new;
    Lambda2_old = Lambda2_new;    

    if(~mod(iter,options.ADMM.rho_gap))
        rho = min(1e5,rho * options.ADMM.rho_rate);
        rho_1 = 1.*rho;
        rho_2 = 1.*rho;
    end     
end
Z_final = Z_new;     
Ux_final = Ux_new;    
Uc_final = Uc_new;    


function [Z_final, Z_cell, obj, alpha] = mlml_pgd_template(A, B, C, Z0, options)

% min_Z { tr(A' * Z) + tr(Z B Z') + tr(Z' C Z) }, st. -1 <= Z <= 1

max_iter = options.PGD.max_iter; 
gap_compute = options.PGD.gap_compute;
rate_step = options.PGD.rate_step;
alpha_rate = options.PGD.alpha_rate;

B = full(B); C = full(C);

obj = zeros(1, max_iter);
%obj(1) = obj_compute(A, B, C, Z0); 

alpha = zeros(1, max_iter);

Z_old = Z0;
Z_cell = cell(1,max_iter);
Z_cell{1} = Z_old;

for i=1:max_iter
     
    % step 1, gradient
    %Z_gradient = A + 2.* Z_old * full( B ) + 2.* full(C) * Z_old; 
    Z_gradient = A + 2.* Z_old *  B  + 2.* C * Z_old; 
    
    % step 2, step size by exact line search
    if ((i-1)/gap_compute) == fix((i-1)/gap_compute)
          alpha(i) = ( step_size_compute(A, B, C, Z_old, Z_gradient) ) / (alpha_rate*i); 
    else
        alpha(i) = alpha(i-1) * rate_step;
    end
    
    % step 3, update of Z
    Z_new = Z_old - alpha(i) .* Z_gradient;
    
    % step 4, projection to [-1,1] 
    Z_new = min(1,  max(-1, Z_new) );
    Z_cell{i+1} = Z_new;
    
    % step 5, objective function and check convergence
%     if ((i)/5) == fix((i)/5)
%         obj(i+1) = obj_compute(A, B, C, Z_new); 
%         obj_diff=abs(obj(i+1)-obj(i+1 - 1))/abs(obj(i));
%          if (obj_diff<1*10^(-8))
%            break;
%          end    
%     end
    Z_old = Z_new;
    
end

Z_final=Z_new;
obj( obj == 0) = [];



function obj = obj_compute(A, B, C, Z)

obj = trace(Z * A') + trace( (Z * full(B)) * Z') + trace( full(C) * (Z * Z') );

function alpha = step_size_compute(A, B, C, Z, Z_gradient)

% the step size by exact line search for   tr(A' * Z) + tr(Z B Z') + tr(Z' C Z)

numerator = 0.5 * trace( Z_gradient * A') + trace( (Z * B) * Z_gradient' ) + trace( C * (Z_gradient * Z') );
dom = trace( (Z_gradient * B) * Z_gradient' ) + trace( C * (Z_gradient * Z_gradient') );

alpha = numerator / dom; 
