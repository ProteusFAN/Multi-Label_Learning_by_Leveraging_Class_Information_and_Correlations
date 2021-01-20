%% non-symmetric graph laplacian of class

clear
close all
base_path = '/Users/Proteus/Documents/MATLAB/Internship/Experiments';
chdir(base_path)
addpath(genpath(pwd))

dataset_train=load('yeast_train.txt');
label_train=(dataset_train(:,end-13:end))';   % 14 x 1500  matrix
label_train(label_train==0) = -1;

%% the number of ICA bounds
n_ica = 15;

%% algorithm

[~, ~, ~, W_c] = Vc_compute(label_train, 14);
W_c = triu(W_c);
candidate = sort(W_c(:),'descend');
threshold = candidate(n_ica);
L_c = W_c;
L_c(W_c < threshold) = 0;

vol = sum(L_c,2);
L_c = L_c + diag(vol);

save(fullfile(base_path,'ICA_combination','L_c.mat'), 'L_c');