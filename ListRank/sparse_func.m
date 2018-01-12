function [ Sout ] = sparse_func( S, f )
% Returns B = f(S) .* (S != 0)
[fi, fj, fv] = find(S);
fv = f(fv);
Sout = sparse(fi, fj, fv, size(S,1), size(S,2));
end

