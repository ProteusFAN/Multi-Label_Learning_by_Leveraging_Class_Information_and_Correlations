%% task 2
task = 2;
batch = 2560;

base_path = '/data1/fanmin/Experiments';
% base_path = '/Users/Proteus/Documents/MATLAB/Internship/Experiments';

matlabpool(10);
matlabpool('addAttachedFiles',{fullfile(base_path, 'para_ica_bounds.m')});

parfor ii = (task-1)*batch+1 : task*batch
    para_ica_bounds(ii);
    fprintf('Iteration: %d/%d.\n', ii-(task-1)*batch, batch);   
end
matlabpool close;