function A=readmat(path)

fprintf('Loading %s\n', path);
fid = fopen(path, 'r');
sz = fscanf(fid, '%d', 2);
data = fscanf(fid, '%f');
fclose(fid);
data = reshape(data, 3, [])';
A = sparse(data(:,1)+1, data(:,2)+1, data(:,3), sz(1), sz(2));
