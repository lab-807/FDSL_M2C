function [laplacian] = Laplaican(X,W, numClust, N)
if size(X,1)>=size(X,2)
    X_new=zeros(N);
    for i = 1:N
        X_new(i,:)=X(i,:);
    end
else
    n = ceil(size(X,2)/size(X,1));
    X_new=[];
    for i = 1:n
        X_new=[X_new; X];
    end
    X_new=X_new(1:N,:);
end
% A = affinity_matrix(X_new,numClust,N);
A = W;
eye_matrix = 1 - eye(size(A));
A = A .* eye_matrix;
c_diag = sum(A, 2);
c_diag(c_diag == 0) = 1;
c_diag(c_diag < 10^(-10)) = 10^(-10);
c_diag = diag(sqrt(c_diag.^(-1)));
laplacian = eye(size(A)) - c_diag * A * c_diag;
end

function [aff_matrix] = affinity_matrix(X_new, numClust, N)
NeiNum = round(N/numClust);
[aff_matrix, indx] = genarateNeighborhood(X_new , NeiNum);
aff_matrix = (aff_matrix + aff_matrix');
end

function [aff_matrix, indx] = genarateNeighborhood(KC,tau)
aff_matrix = zeros(size(KC, 2));
smp_num = size(KC, 2);
KC0 = KC - 10^18*eye(smp_num);
% KC0 = KC;
[~,indx] = sort(KC0,'descend');
indx_0 = indx(1:tau,:);

smp_index = 1 : smp_num;
smp_index = repmat(smp_index, tau, 1);

indx_0 = indx_0(:);
smp_index = smp_index(:);
real_index = (smp_index-1) * smp_num + indx_0;
aff_matrix(real_index) = 1;
end
