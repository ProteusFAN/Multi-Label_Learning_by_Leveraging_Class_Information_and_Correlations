    function D = Sparse2Dense(S)
        [i,j,s] = find(S);
        z = zeros(size(s,1),1);
        D = [i,j,z,s,s,z];  % [i,j,e00,e01,e10,e11]
    end