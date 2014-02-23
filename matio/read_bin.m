function [A] = read_bin(filename)

fid = fopen(filename, 'r');
M = fread(fid, [1,1], 'integer*8');
N = fread(fid, [1,1], 'integer*8');
A = fread(fid, [M,N], 'double');
fclose(fid);
