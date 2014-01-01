function [D, Q] = dlaed0(D, E)
% computes all eigenvalues and corresponding eigenvectors of a symmetric
% tridiagonal matrix using the divide-and-conquer eigenvalue algorithm.
% We will have
%    diag(D(in)) + diag(E, 1) + diag(E, -1) = Q * diag(D(out)) + Q'.

save('input.mat', 'D', 'E');
system('./tester input.mat D.mat Q.mat');
load('D.mat');
load('Q.mat');
