function [D, Q, perm1] = dlaed1(D, Q, perm1, rho, cutpnt)
% computes the updated eigensystem of a diagonal matrix after modification by a
% rank-one symmetric matrix.
%
%   T = Q(in) ( D(in) + rho * z * z') Q'(in) = Q(out) * D(out) * Q'(out)
%     where z = Q'(in) * u, u is a vector of length N with ones in the cutpnt
%     and cutpnt + 1 th elements and zeros elsewhere. 
N = length(D);

% form the z vector
z = [Q(cutpnt, 1:cutpnt)'; Q(cutpnt+1, cutpnt+1:end)']

% deflate eigenvalues
perm1(cutpnt+1:end) = perm1(cutpnt+1:end) + cutpnt;
[K, D, Q, rho, w] = dlaed2(D, Q, cutpnt, perm1, rho, z)

% solve secular equation
if K > 0
    [D(1:K), Qhat] = dlaed3(D(1:K), w, 1/rho);
    Qhat
    % back-transformation
    Q = [Q(:, 1:K) * Qhat, Q(:, K+1:end)];

    % compute perm1 that would merge back deflated values.
    perm1 = dlamrg(K, N - K, D, 1, -1);
else
    perm1 = 1:N;
end
