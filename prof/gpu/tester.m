function tester(NGPU, matsiz)

%{
v = randn(matsiz, 1);
v = v(:);
[U, ~] = qr(randn(matsiz));
A = U * diag(v) * U'; % symmetric system with eigenvalues v
A = triu(A) + tril(A) - diag(diag(A)); % make it perfectly symmetric
A = hess(A); % reduce A to tridiagonal using orthogonal transformations
D = diag(A);
E = diag(A, 1);
save(sprintf('input_%d.mat', matsiz), 'D', 'E');
save(sprintf('v_%d.mat', matsiz), 'v');
%}
D = read_bin(sprintf('D_%d.bin', matsiz));
E = read_bin(sprintf('E_%d.bin', matsiz));
v = read_bin(sprintf('v_%d.bin', matsiz));
[d, q] = dlaed0_m(NGPU, D, E);
A = diag(D) + diag(E,1) + diag(E,-1);
fprintf(1, 'max error in eigenvalues = %.20g\n', max(abs(d-sort(v))));
fprintf(1, 'max error in eigendecomp = %.20g\n', ...
    max(max(abs(A - q*diag(d)*q'))));
