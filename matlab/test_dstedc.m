load A.mat
load D.mat
load E.mat

[d, q] = dstedc(D, E);
fprintf('Minimum error in eigenvalues = %.12g\n', min(abs(d - sort(eig(A)))) );
fprintf('Maximum error in eigenvalues = %.12g\n', max(abs(d - sort(eig(A)))) );
fprintf('Minimum residual from A to numerical decomposition = %.12g\n', ...
    min(min(abs(A - q * diag(d) * q'))) );
fprintf('Maximum residual from A to numerical decomposition = %.12g\n', ...
    max(max(abs(A - q * diag(d) * q'))) );
