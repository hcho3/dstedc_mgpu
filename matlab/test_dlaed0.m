[U, ~] = qr(randn(100));
A = U * diag(2:2:200) * U'; % symmetric matrix with eigenvalues 2:2:200
A = triu(A) + tril(A) - diag(diag(A)); % make it perfectly symmetric
A = hess(A); % reduce A to tridiagonal using orthogonal transformations
D = diag(A);
E = diag(A, 1);
eigvals = (2:2:200)';
[d, q] = dlaed0(D, E);
max(abs(d - eigvals))
max(max(abs(A - q * diag(d) * q')))
