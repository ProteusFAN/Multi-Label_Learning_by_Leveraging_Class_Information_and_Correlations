function [ CIB_2 ] = CIB2( data_info )
%% compute CIB-2

% preparation
num_train = data_info.num_sample_train;
num_all = data_info.num_sample;

center = num_all/num_train * sum(max(data_info.label_train,0), 2);

% width and CIB-2
width = sqrt(num_all);

CIB_2{1} = max(center - width,0);
CIB_2{2} = min(center + width, num_all);

end