for i=1:100
    i
    D = rand(1000, 1);
    E = rand(999, 1);

    A = diag(D) + diag(E, 1) + diag(E, -1);

    [d, q] = dlaed0(D, E);
    fprintf('Minimum error in eigenvalues = %.12g\n', min(abs(d - sort(eig(A)))) );
    fprintf('Maximum error in eigenvalues = %.12g\n', max(abs(d - sort(eig(A)))) );
    fprintf('Minimum residual from A to numerical decomposition = %.12g\n', ...
        min(min(abs(A - q * diag(d) * q'))) );
    fprintf('Maximum residual from A to numerical decomposition = %.12g\n', ...
        max(max(abs(A - q * diag(d) * q'))) );
end
