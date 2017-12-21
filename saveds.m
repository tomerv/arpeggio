function [ ] = saveds( mat, filename )
    dlmwrite(filename, size(mat), 'delimiter', ' ');
    [i, j, v] = find(mat);
    dlmwrite(filename, [i-1 j-1 v], 'delimiter', ' ', '-append');
end

