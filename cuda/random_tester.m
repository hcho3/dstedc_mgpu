function random_matrices_with_eigval(filename, v, num_it)

if ~ischar(filename)
    error('The file name must be a valid string.');
    return;
end

if ~isempty(filename)
    fprintf(1, 'Creating %s...\n', filename);
    diary(filename);
else
    fprintf(1, 'You chose not to create a log.\n');
end

rs_max_eig_error = 0.0;
rs_max_decomp_error = 0.0;
rs_time = 0.0;

for i=1:num_it
    i
    [U, ~] = qr(randn(1000));
    A = U * diag(v) * U'; % symmetric system with eigenvalues v
    A = triu(A) + tril(A) - diag(diag(A)); % make it perfectly symmetric
    A = hess(A); % reduce A to tridiagonal using orthogonal transformations
    D = diag(A);
    E = diag(A, 1);
    tic;
    [d, q] = dlaed0(D, E);
    t = toc;
    max_eig_error = max(abs(d - v));
    max_decomp_error = max(max(abs(A - q * diag(d) * q')));
    rs_max_eig_error = rs_max_eig_error + max_eig_error;
    rs_max_decomp_error = rs_max_decomp_error + max_decomp_error;
    rs_time = rs_time + t;
    fprintf(1, 'Running average of max_eig_error = %.12g\n', ...
        rs_max_eig_error / i);
    fprintf(1, 'Running average of max_decomp_error = %.12g\n', ...
        rs_max_decomp_error / i);
    fprintf(1, 'Running average of running time = %.12g\n', rs_time / i);
end

if strcmp(get(0,'Diary'), 'on')
    diary off;
end
