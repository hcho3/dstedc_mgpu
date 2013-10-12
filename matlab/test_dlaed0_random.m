% running sums of errors
rs_max_eig_error = 0.0;
rs_max_decomp_error = 0.0;
rs_time = 0.0;

%% symmetric tridiagonal matrices with all eigenvalues are of multiplicity 2
%% in which the eigenvalues are clearly separated.
%{
diary('multiplicity_separated.log');
v = [(1:1:500); (1:1:500)];
v = v(:);
for i=1:200
    i
    [U, ~] = qr(randn(1000));
    A = U * diag(v) * U'; % eigenvalues 1:1:100, each with multiplicity 2
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
diary off;
%}

%% symmetric tridiagonal matrices with all eigenvalues are of multiplicity 2
%% in which the eigenvalues are not so clearly separated.
%{
% running sums of errors
rs_max_eig_error = 0.0;
rs_max_decomp_error = 0.0;
rs_time = 0.0;
diary('multiplicity_not_separated.log');
v = [(1:0.0000001:1.0000499); (1:0.0000001:1.0000499)];
v = v(:);
for i=1:200
    i
    [U, ~] = qr(randn(1000));
    A = U * diag(v) * U'; % eigenvalues 1:1:100, each with multiplicity 2
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
diary off;
%}

%% symmetric tridiagonal matrices with two clusters, one small and one large.
%{
% running sums of errors
rs_max_eig_error = 0.0;
rs_max_decomp_error = 0.0;
rs_time = 0.0;
diary('multipole.log');
v = [linspace(0, 0.0000499, 500); linspace(1000, 1000.0000499, 500)]';
v = v(:);
for i=1:200
    i
    [U, ~] = qr(randn(1000));
    A = U * diag(v) * U'; % eigenvalues 1:1:100, each with multiplicity 2
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
diary off;
%}

%% symmetric tridiagonal matrices with (uniformly) random diagonal entries
% running sums of errors
rs_max_eig_error = 0.0;
rs_max_decomp_error = 0.0;
rs_time = 0.0;
diary('uniform_random.log');
for i=1:200
    i
    D = rand(1000, 1);
    E = rand(999, 1);

    A = diag(D) + diag(E, 1) + diag(E, -1);

    tic;
    [d, q] = dlaed0(D, E);
    t = toc;
    max_eig_error = max(abs(d - sort(eig(A))));
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
diary off;
