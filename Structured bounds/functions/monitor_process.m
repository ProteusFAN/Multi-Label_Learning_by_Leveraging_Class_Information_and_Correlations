%% Calculating Z * 1_n and see how it changes corrosponding bounds (CIB-2)

num_cell = length(Z_cell);
Z_cell_row_discrete_sum = zeros(num_c, num_cell);
Z_cell_row_sum = zeros(num_c, num_cell);
Z_cell_row_pos = zeros(num_c, num_cell);
for ii = 1:num_cell
    Z_cell_row_discrete_sum(:,ii) = (sign(Z_cell{ii}) + 1)/2 * ones(num_sample,1);  % discrete value
    Z_cell_row_sum(:,ii) = (Z_cell{ii} + 1)/2 * ones(num_sample,1);  % real value
    diff_left = min( Z_cell_row_sum(:,ii) - CIB_2{1}, 0 );
    diff_right = max( Z_cell_row_sum(:,ii) - CIB_2{2}, 0 );
    Z_cell_row_pos(:,ii) = diff_left + diff_right;
end

%% Evaluation of Z in each interation and see how it changes

eval_train = [];
eval_test = [];
for ii = 1:num_cell
    Z_temp = sign(Z_cell{ii});
    [eval_train_temp, eval_test_temp] = evaluation_discrete_multi_label(Z_temp, label_train, label_test);
    eval_train = [eval_train, eval_train_temp(end-3:end)];
    eval_test = [eval_test, eval_test_temp(end-3:end)];
end

%% recording

monitor.z_cell = Z_cell;
monitor.z_cell_row_discrete_sum = Z_cell_row_discrete_sum;
monitor.z_cell_row_sum = Z_cell_row_sum;
monitor.z_cell_row_pos = Z_cell_row_pos;

monitor.eval_train = eval_train;
monitor.eval_test = eval_test;

log.obj_Z = obj_Z;
log.obj_L = obj_L;
monitor.log = log;