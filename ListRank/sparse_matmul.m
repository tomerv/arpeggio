function [ C ] = sparse_matmul( A, B, f, S )
% Returns C = f(A * B) .* S
[fi, fj, ~] = find(S);
fv = zeros(nnz(S),1);
fprintf('sparse_matmul: multiplying %d elements...', nnz(S));
for i = 1:nnz(S)
    if mod(i,1000000)==0
        fprintf('%d...', i);
    end
    fv(i) = f( A(fi(i),:) * B(:,fj(i)) );
    if isnan(fv(i))
        fprintf('Found NaN on (%d,%d)', fi(i), fj(i));
    end
end
fprintf('\n');
C = sparse(fi, fj, fv, size(S,1), size(S,2));
end

