function [lambda, Q] = dlaed3(delta, z, rho)
% stably computes the eigendecomposition Q * diag(lambda) * Q**T  of
% diag(delta) + (1/rho) * z * z**T  by solving an inverse eigenvalue problem.
n = length(delta);
tau = zeros(n, 1);
orig = zeros(n, 1);
v = zeros(n, 1);
Q = zeros(n, n);

for i=1:n
    [tau(i), orig(i)] = dlaed4(i, delta, z, rho);
end
% inverse eigenvalue problem: find v such that lambda(1), ..., lambda(n) are
% exact eigenvalues of the matrix D + v * v**T.
for i=1:n
    v(i) = ...
        sign(z(i)) * sqrt(rho * prod(delta(i) - orig(1:i-1) - tau(1:i-1)) * ...
        prod(orig(i:n) - delta(i) + tau(i:n)) / ...
        (prod(delta(i) - delta(1:i-1)) * prod(delta(i+1:n) - delta(i))) );
end
% uncomment to test the inverse eigenvalue routine
% abs((tau+orig)-eig(diag(delta) + v * v'))

% compute the eigenvectors of D + v * v**T
lambda = tau + orig;
for j=1:n
    for i=1:n
        Q(i, j) = v(i) / (orig(j) - delta(i) + tau(j));
    end
    Q(:, j) = Q(:, j) / norm(Q(:, j), 2);
end
