function [ indicator ] = papca( feature )
%% Parallel Analysis Program For Principal Component Analysis
%   with random normal data simulation or Data Permutations.

%  This program conducts parallel analyses on data files in which
%  the rows of the data matrix are cases/individuals and the
%  columns are variables; There can be no missing values;

%  You must also specify:
%   -- ndatsets: the # of parallel data sets for the analyses;
%   -- percent: the desired percentile of the distribution of random
%      data eigenvalues [percent];
%   -- randtype: whether (1) normally distributed random data generation 
%      or (2) permutations of the raw data set are to be used in the
%      parallel analyses (default=[2]);

%  WARNING: Permutations of the raw data set are time consuming;
%  Each parallel data set is based on column-wise random shufflings
%  of the values in the raw data matrix using Castellan's (1992, 
%  BRMIC, 24, 72-77) algorithm; The distributions of the original 
%  raw variables are exactly preserved in the shuffled versions used
%  in the parallel analyses; Permutations of the raw data set are
%  thus highly accurate and most relevant, especially in cases where
%  the raw data are not normally distributed or when they do not meet
%  the assumption of multivariate normality (see Longman & Holden,
%  1992, BRMIC, 24, 493, for a Fortran version); If you would
%  like to go this route, it is perhaps best to (1) first run a 
%  normally distributed random data generation parallel analysis to
%  familiarize yourself with the program and to get a ballpark
%  reference point for the number of factors/components;
%  (2) then run a permutations of the raw data parallel analysis
%  using a small number of datasets (e.g., 10), just to see how long
%  the program takes to run; then (3) run a permutations of the raw
%  data parallel analysis using the number of parallel data sets that
%  you would like use for your final analyses; 1000 datasets are 
%  usually sufficient, although more datasets should be used
%  if there are close calls.

tic

raw = feature;

ndatsets  = 100; % Enter the desired number of parallel data sets here

percent   = 0.95; % Enter the desired percentile here

% Enter either
%  1 for normally distributed random data generation parallel analysis, or
%  2 for permutations of the raw data set (more time consuming).
randtype = 2;

%the next command can be used to set the state of the random # generator
randn('state',1953125)

%%%%%%%%%%%%%%% End of user specifications %%%%%%%%%%%%%%%

[ncases,nvars] = size(raw);

evals = []; % random eigenvalues initialization
% principal components analysis & random normal data generation
if (randtype == 1)
    realeval = sort(eig(corrcoef(raw)),'descend');    % better use corrcoef
    for nds = 1:ndatsets;
        evals(:,nds) = eig(corrcoef(randn(ncases,nvars)));
    end
end

% principal components analysis & raw data permutation
if (randtype == 2)
    %realeval = flipud(sort(eig(corrcoef(raw))));    % either cov/corrcoef
    realeval = sort(eig(cov(raw)),'descend');
    for nds = 1:ndatsets; 
    x = raw;
        for lupec = 2:nvars;
            % Here we use randperm in matlabl
            x(:,lupec) = x(randperm(ncases),lupec);
            % Below is column-wise random shufflings
            %  of the values in the raw data matrix using Castellan's (1992, 
            %  BRMIC, 24, 72-77) algorithm;
            %
            %for luper = 1:(ncases -1);
            %k = fix( (ncases - luper + 1) * rand(1) + 1 )  + luper - 1;
            %d = x(luper,lupec);
            %x(luper,lupec) = x(k,lupec);
            %x(k,lupec) = d;end;end;
        end
        %evals(:,nds) = eig(corrcoef(x));   % either cov/corrcoef
        evals(:,nds) = eig(cov(x));
    end
end

evals = sort(evals,'descend');
pvals = sum(evals>(realeval*ones(1,ndatsets)),2)/ndatsets; % p-values of observed random eigenvalues greater than real eigenvalues

indicator = pvals > (1-percent);
end

