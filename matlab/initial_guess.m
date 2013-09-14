function [tau, orig] = initial_guess(k, delta, zeta, rho)
% make a good initial guess so that subsequent iterations do not jump out of
% the permissible range.

n = length(delta);
delta_n_1 = delta(n) + zeta' * zeta / rho; % delta(n+1)

if k >= 1 && k < n
    [fval, gval] = eval_midpoint(k, n, delta, delta_n_1, zeta, rho);
    DELTA = delta(k+1) - delta(k);
    if fval >= 0
        K = k;
        c = gval;
        a = c * DELTA + zeta(k)^2 + zeta(k+1)^2;
        b = zeta(k)^2 * DELTA;
    else
        K = k+1;
        c = gval;
        a = -c * DELTA + zeta(k)^2 + zeta(k+1)^2;
        b = -zeta(k+1)^2 * DELTA;
    end
    if a <= 0
        tau = (a - sqrt(a^2 - 4 * b * c)) / (2 * c);
    else
        tau = (2 * b) / (a + sqrt(a^2 - 4 * b * c));
    end
else
    [fval, gval] = eval_midpoint(n, n, delta, delta_n_1, zeta, rho);
    h = @(x) zeta(n-1)^2 / (delta(n-1)-x) + zeta(n)^2 / (delta(n)-x);
    if fval <= 0
        if gval <= -h(delta_n_1)
            tau = zeta' * zeta / rho;
        else
            DELTA = delta(n) - delta(n-1);
            c = gval;
            a = -c * DELTA + (zeta(n-1)^2 + zeta(n)^2);
            b = -zeta(n)^2 * DELTA;
            if a >= 0
                tau = (a + sqrt(a^2 - 4 * b * c)) / (2 * c);
            else
                tau = (2 * b) / (a - sqrt(a^2 - 4 * b * c));
            end
        end
    else
        DELTA = delta(n) - delta(n-1);
        c = gval;
        a = -c * DELTA + (zeta(n-1)^2 + zeta(n)^2);
        b = -zeta(n)^2 * DELTA;
        if a >= 0
            tau = (a + sqrt(a^2 - 4 * b * c)) / (2 * c);
        else
            tau = (2 * b) / (a - sqrt(a^2 - 4 * b * c));
        end
    end
    K = n;
end
orig = delta(K);

function [fval, gval] = eval_midpoint(k, n, delta, delta_n_1, zeta, rho)
% evaluates the values of f and g at (delta(k) + delta(k+1)) / 2
if k < n
    gval = rho;
    mpt = (delta(k) + delta(k+1)) / 2;
    for j=1:k-1
        gval = gval + zeta(j)^2 / (delta(j)-mpt);
    end
    for j=k+2:n
        gval = gval + zeta(j)^2 / (delta(j)-mpt);
    end
    fval = gval + zeta(k)^2 / (delta(k)-mpt) + zeta(k+1)^2 / (delta(k+1)-mpt);
else
    gval = rho;
    mpt = (delta(n) + delta_n_1) / 2;
    for j=1:n-2
        gval = gval + zeta(j)^2 / (delta(j)-mpt);
    end
    fval = gval + zeta(n-1)^2 / (delta(n-1)-mpt) + zeta(n)^2 / (delta(n)-mpt);
end
