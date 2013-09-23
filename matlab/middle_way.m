function [eta] = middle_way(k, delta, zeta, rho, tau, orig)

n = length(delta);
if k >= 1 && k < n
    DELTA_k = delta(k) - orig - tau;
    DELTA_k_1 = delta(k+1) - orig - tau;
    f_y = rho + sum(zeta.^2 ./ (delta - orig - tau));
    fderiv_y = sum(zeta.^2 ./ (delta - orig - tau).^2);
    psideriv_y = sum(zeta(1:k).^2 ./ (delta(1:k) - orig - tau).^2);
    phideriv_y = sum(zeta(k+1:n).^2 ./ (delta(k+1:n) - orig - tau).^2);

    a = (DELTA_k + DELTA_k_1) * f_y - DELTA_k * DELTA_k_1 * fderiv_y;
    b = DELTA_k * DELTA_k_1 * f_y;
    c = f_y - DELTA_k * psideriv_y - DELTA_k_1 * phideriv_y;
    if a <= 0
        eta = (a - sqrt(a^2 - 4 * b * c)) / (2 * c);
    else
        eta = (2 * b) / (a + sqrt(a^2 - 4 * b * c));
    end
    % if eta + y falls below delta(k) or exceeds delta(k+1), do a Newton step.
    if orig + tau + eta <= delta(k) || orig + tau + eta >= delta(k+1)
        eta = -f_y / fderiv_y;
    end
else
    delta_n_1 = delta(n) + zeta' * zeta / rho; % delta(n+1)
    DELTA_n_1 = delta(n-1) - orig - tau;
    DELTA_n = delta(n) - orig - tau;
    f_y = rho + sum(zeta.^2 ./ (delta - orig - tau));
    fderiv_y = sum(zeta.^2 ./ (delta - orig - tau).^2);
    psideriv_y = sum(zeta(1:n-1).^2 ./ (delta(1:n-1) - orig - tau).^2);

    a = (DELTA_n_1 + DELTA_n) * f_y - DELTA_n_1 * DELTA_n * fderiv_y;
    b = DELTA_n_1 * DELTA_n * f_y;
    c = f_y - DELTA_n_1 * psideriv_y - zeta(n)^2 / DELTA_n;
    if a >= 0
        eta = (a + sqrt(a^2 - 4 * b * c)) / (2 * c);
    else
        eta = (2 * b) / (a - sqrt(a^2 - 4 * b * c));
    end
    % if eta + y falls below delta(n), simply do a Newton step.
    if orig + tau + eta <= delta(n) || orig + tau + eta >= delta_n_1
        eta = -f_y / fderiv_y;
    end
end
