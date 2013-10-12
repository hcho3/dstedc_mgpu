function [lambda, Q] = dlaed3(delta, z, rho)
% stably computes the eigendecomposition Q * diag(lambda) * Q**T  of
% diag(delta) + rho * z * z**T  by solving an inverse eigenvalue problem.
n = length(delta);
tau = zeros(n, 1);
orig = zeros(n, 1);
v = zeros(n, 1);
Q = zeros(n, n);

for i=1:n
    [tau(i), orig(i)] = dlaed4(i, delta, z, rho);
end
% uncomment to test the secular equation solver
%{
t = max( tau+orig- sort(eig( diag(delta) + rho * z * z')) );
assert(t <= 1e-12, 'too much error in secular solver: %.12g', t);
fprintf('secular solver = %.12g\n', t);
%}
% inverse eigenvalue problem: find v such that lambda(1), ..., lambda(n) are
% exact eigenvalues of the matrix D + v * v**T.
for i=1:n
    v(i) = orig(i) - delta(i) + tau(i);
    for j=1:i-1
        v(i) = v(i) * ((delta(i) - orig(j) - tau(j)) / (delta(i) - delta(j)));
    end
    for j=i+1:n
        v(i) = v(i) * ((orig(j) - delta(i) + tau(j)) / (delta(j) - delta(i)));
    end
    v(i) = sign(z(i)) * sqrt(v(i));
end

% uncomment to test the inverse eigenvalue routine
%{
diff = zeros(n, 1);
ref = sort(eig(diag(delta) + v * v'));
for i=1:n
    diff(i) = abs(orig(i) - ref(i) + tau(i));
end
t = max(max(diff));
assert(t <= 1e-12, 'too much error in inv. eig. problem: %.12g', t);
fprintf('inv. eig. problem = %.12g\n', t);
fprintf('------------------------------------------------------\n');
%}

% compute the eigenvectors of D + v * v**T
lambda = tau + orig;
for j=1:n
    for i=1:n
        Q(i, j) = v(i) / (orig(j) - delta(i) + tau(j));
    end
    Q(:, j) = Q(:, j) / norm(Q(:, j), 2);
end
