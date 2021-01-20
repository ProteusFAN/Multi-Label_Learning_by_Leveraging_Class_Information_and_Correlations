%% monitor process
num_cell = length(Z_cell);

%% Evaluation of Z in each interation and see how it changes
eval_train = zeros(4, num_cell);
eval_test = zeros(4, num_cell);
for ii = 1:num_cell
    Z_temp = sign(Z_cell{ii});
    [eval_train_temp, eval_test_temp] = evaluation_discrete_multi_label(Z_temp, data_info.label_train, data_info.label_test);
    eval_train(:,ii) = eval_train_temp(end-3:end);
    eval_test(:,ii) = eval_test_temp(end-3:end);
end

%% recording
monitor.Z_cell = Z_cell;
monitor.obj_Z = obj_Z;
monitor.obj_L = obj_L;
monitor.log = log;

monitor.eval_train = eval_train;
monitor.eval_test = eval_test;