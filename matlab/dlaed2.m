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
% compute perm2 that merge-sorts D1, D2 into one sorted list.
perm2 = dlamrg(N1, N2, D(perm1), 1, 1);

% apply perm2.
permacc = perm1(perm2);

% compute allowable deflation tolerance.
[~, imax] = max(abs(z));
[~, jmax] = max(abs(D));
tol = 8.0 * eps * max( [abs(D(jmax)), abs(z(imax))] );

% If the rank-1 modifier is small enough, we're done: all eigenvalues deflate.
if rho * abs(z(imax)) <= tol
    K = 0;
    D = D(permacc);
    Q = Q(:, permacc);
    w = zeros(0, 1);
    return;
end

% --deflation--
perm3 = zeros(N, 1);
K = 0;
K2 = N;
for i=1:N
    pi = permacc(i);
    if rho * abs(z(pi)) <= tol
        % 1) z-component is small
        % => move D(i) to the end of list.
        perm3(K2) = i;
        K2 = K2 - 1;
    elseif i < N
        ni = permacc(i+1);
        t = abs(D(ni) - D(pi));
        tau = dlapy2(z(pi), z(ni));
        s = -z(pi) / tau;
        c = z(ni) / tau;
        if abs(t * c * s) <= tol
            % 2) D(i) and D(i+1) are close to each other compared to the
            %    z-weights given to them.
            % => zero out z(i) by applying a Givens rotation. After this step
            %    D(i) can be deflated away.
            z(ni) = tau;
            z(pi) = 0.0;
            pq = Q(:, pi);
            nq = Q(:, ni);
            Q(:, pi) =  c * pq + s * nq;
            Q(:, ni) = -s * pq + c * nq;
            t     = c^2 * D(pi) + s^2 * D(ni);
            D(ni) = s^2 * D(pi) + c^2 * D(ni);
            D(pi) = t;
            perm3(K2) = i;
            k = 0;
            while K2+k+1 <= N && ...
                D(permacc(perm3(K2+k))) < D(permacc(perm3(K2+k+1))) 
                t = perm3(K2+k);
                perm3(K2+k) = perm3(K2+k+1);
                perm3(K2+k+1) = t;
                k = k + 1;
            end
            K2 = K2 - 1;
        else
            % 3) D(i) is not deflated.
            K = K + 1;
            perm3(K) = i;
        end
    else
        % 3) D(i) is not deflated.
        K = K + 1;
        perm3(K) = i;
    end
end

% Apply perm3 to eigenpairs.
permacc = permacc(perm3);
D = D(permacc);
Q = Q(:, permacc);
z = z(permacc);
w = z(1:K);
