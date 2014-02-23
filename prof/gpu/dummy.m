matsiz = 36000;
v = randn(matsiz, 1);
v = v(:);
[U, ~] = qr(randn(matsiz));
A = U * diag(v) * U'; % symmetric system with eigenvalues v
A = triu(A) + tril(A) - diag(diag(A)); % make it perfectly symmetric
A = hess(A); % reduce A to tridiagonal using orthogonal transformations
D = diag(A);
E = diag(A, 1);
write_bin(sprintf('D_%d.mat', matsiz), D);
write_bin(sprintf('E_%d.mat', matsiz), E);
write_bin(sprintf('v_%d.mat', matsiz), v);
