%% Evaluation

Z_admm_discrete = sign(Z_admm);
[result_train_admm_dis_vec, result_test_admm_dis_vec] = ...
    evaluation_discrete_multi_label(Z_admm_discrete, data_info.label_train, data_info.label_test);
fprintf('The discrete evaluation results of ADMM are: \n')
disp([result_train_admm_dis_vec, result_test_admm_dis_vec]');
result_cell = [result_train_admm_dis_vec, result_test_admm_dis_vec]';
