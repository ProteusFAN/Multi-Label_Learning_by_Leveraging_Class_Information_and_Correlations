
function edges_matrix = W2edgematrix(W)
dd=sum(W,2)+eps; % the summation of each row, n*1 vector
dd=sqrt(1./dd);

mn = length(dd);

W_triu =triu(W);
[aa, bb , cc] = find(W_triu);

W_vector = W(sub2ind([mn, mn], aa, bb));
c00 = (dd(aa) - dd(bb)).^2 .* W_vector;
c01 = (dd(aa) + dd(bb)).^2 .* W_vector;

edges_matrix = full([aa, bb, c00, c01, c01, c00]);
