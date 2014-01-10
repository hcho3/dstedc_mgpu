function write_bin(filename, A)

fid = fopen(filename, 'w');
M = int64(size(A, 1));
N = int64(size(A, 2));
fwrite(fid, M, 'integer*8');
fwrite(fid, N, 'integer*8');
fwrite(fid, A, 'double');
fclose(fid);
