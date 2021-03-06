addpath('../matio');
if exist('dlaed1', 'dir') ~= 7
    error('dlaed1/ directory must exist.');
end
if length(dir('dlaed1')) > 2
    fprintf(1, 'Warning: dlaed1/ directory is not empty. It may contain ');
    fprintf(1, 'some test matrices already.\n');
    prompt = 'Are you sure you want to create new test matrices? Y/N [Y]: ';
    str = input(prompt, 's');
    if isempty(str)
        str = 'Y';
    end
    if strncmpi(str, 'N', 1)
        fprintf(1, 'aborting...\n');
        return;
    end
end
for matsiz=[1024 2048 4096 8192 16384 32768 50000]
    matsiz
    v = randn(matsiz, 1);
    v = v(:);
    [U, ~] = qr(randn(matsiz));
    A = U * diag(v) * U'; % symmetric system with eigenvalues v
    A = triu(A) + tril(A) - diag(diag(A)); % make it perfectly symmetric
    A = hess(A); % reduce A to tridiagonal using orthogonal transformations
    D = diag(A);
    E = diag(A, 1);
    write_bin(sprintf('dlaed1/D_%d.bin', matsiz), D);
    write_bin(sprintf('dlaed1/E_%d.bin', matsiz), E);
    write_bin(sprintf('dlaed1/v_%d.bin', matsiz), v);
end
