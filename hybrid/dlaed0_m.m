function [D, Q] = dlaed0_m(NGPU, D, E)
% computes all eigenvalues and corresponding eigenvectors of a symmetric
% tridiagonal matrix using the divide-and-conquer eigenvalue algorithm.
% We will have
%    diag(D(in)) + diag(E, 1) + diag(E, -1) = Q * diag(D(out)) + Q'.

save('input.mat', 'D', 'E');
status = system(sprintf('./tester %d input.mat D.mat Q.mat', NGPU));
if status ~= 0
    error('something''s wrong!');
end
load('D.mat');
load('Q.mat');
