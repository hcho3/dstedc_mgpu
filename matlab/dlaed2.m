function [K, D, Q, rho, w] = dlaed2(D, Q, N1, perm1, rho, z)
% merges two lists of eigenvalues and carries out deflation.

N = length(D);
N2 = N - N1;
w = zeros(N, 1);

if rho < 0  % make rho positive
    z(N1+1:end) = -z(N1+1:end);
end
% normalize z so that norm2(z) = 1. Since z is the concatenation of two
% normalized vectors, norm2(z) = sqrt(2).
z = z / sqrt(2);
rho = abs(2 * rho);  % rho = abs(norm(z)^2 * rho)

% apply perm1 to re-merge deflated eigenvalues.
z = z(perm1);
D = D(perm1);

% compute perm2 that would merge two lists of eigenvalues so that D(perm2)
% would be in ascending order.
perm2 = dlamrg(N1, N2, D, 1, 1);
perm3 = perm1(perm2);

% ---deflate---

% compute the allowable deflation tolerance
[~, imax] = max(abs(z));
[~, jmax] = max(abs(D));
tol = 8 * eps * max( abs(D(jmax)), abs(z(imax)) ); 

% if rho is small enough, then we're done; we just need to put the eigenvalues
% and eigenvectors in ascending order.
if rho * abs(z(imax)) <= tol
    K = 0; % all eigenvalues deflated
    D = D(perm2);
    Q = Q(:, perm3);
    w = zeros(0, 1);
    return;
end

perm4 = zeros(1, N);

K = 0;
K2 = N + 1;
for j=1:N
    nj = perm2(j);
    if rho * abs(z(nj)) <= tol  % deflate due to small z component.
        K2 = K2 - 1;
        perm4(K2) = j;
    else
        pj = nj;
        break;
    end
end
j = j + 1;
while j <= N
    nj = perm2(j);
    if rho * abs(z(nj)) <= tol  % deflate due to small z component.
        K2 = K2 - 1;
        perm4(K2) = j;
    else
        % check if eigenvalues are close enough to allow deflation.
        s = z(pj);
        c = z(nj);
        % find sqrt(a^2 + b^2) without overflow or destructive underflow.
        tau = dlapy2(c, s);
        t = D(nj) - D(pj);
        c = c / tau;
        s = -s / tau;
        if abs(t * c * s) <= tol  % deflation is possible.
            z(nj) = tau;
            z(pj) = 0;
            pc = Q(:, pj);
            nc = Q(:, nj);
            Q(:, pj) =  c * pc + s * nc;
            Q(:, nj) = -s * pc + c * nc;
            t = D(pj) * c^2 + D(nj) * s^2;
            D(nj) = D(pj) * s^2 + D(nj) * c^2;
            D(pj) = t;
            K2 = K2 - 1;
            i = 1;
            while K2 + i <= N 
                % find the right place for newly deflated eigenvalue.
                if D(pj) < D(perm2(perm4(K2+i)))
                    perm4(K2+i-1) = perm4(K2+i);
                    perm4(K2+i) = j - 1;
                    i = i + 1;
                else
                    break;
                end
            end
            perm4(K2+i-1) = j - 1;
            pj = nj;
        else  % found a non-deflated eigenvalue
            K = K + 1;
            w(K) = z(pj);
            perm4(K) = j - 1;
            pj = nj;
        end
    end
    j = j + 1;
end
% record the last eigenvalue.
K = K + 1;
w(K) = z(pj);
perm4(K) = j - 1;

perm5 = perm2(perm4);
perm6 = perm1(perm5);

% sort eigenvalues and eigenvectors
D = D(perm5);
Q = Q(:, perm6);
w = w(1:K);
