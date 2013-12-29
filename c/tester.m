for i=[1024 2048 4096 8192]
    %{
    v = randn(i, 1);
    v = v(:);
    [U, ~] = qr(randn(i));
    A = U * diag(v) * U'; % symmetric system with eigenvalues v
    A = triu(A) + tril(A) - diag(diag(A)); % make it perfectly symmetric
    A = hess(A); % reduce A to tridiagonal using orthogonal transformations
    D = diag(A);
    E = diag(A, 1);
    save(sprintf('input_%d.mat', i), 'D', 'E');
    save(sprintf('v_%d.mat', i), 'v');
    %}
    fprintf(1, 'i = %d\n', i);
    load(sprintf('input_%d.mat',i));
    load(sprintf('v_%d.mat',i));
    [d, q] = dlaed0(D, E);
    A = diag(D) + diag(E,1) + diag(E,-1);
    fprintf(1, 'max error in eigenvalues = %.20g\n', max(abs(d-sort(v))));
    fprintf(1, 'max error in eigendecomp = %.20g\n', ...
        max(max(abs(A - q*diag(d)*q'))));

    %max_eig_error = max(abs(d - sort(eig(A))))
    %max_decomp_error = max(max(abs(A - q * diag(d) * q')))
end
