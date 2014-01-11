function [Dout, Q] = dlaed0_m(NGPU, D, E)
% computes all eigenvalues and corresponding eigenvectors of a symmetric
% tridiagonal matrix using the divide-and-conquer eigenvalue algorithm.
% We will have
%    diag(D(in)) + diag(E, 1) + diag(E, -1) = Q * diag(D(out)) + Q'.

write_bin('D.bin', D);
write_bin('E.bin', E);
status = system(sprintf('./tester %d D.bin E.bin Dout.bin Q.bin', NGPU));
if status ~= 0
    error('something''s wrong!');
end
Dout = read_bin('Dout.bin');
Q = read_bin('Q.bin');
