function [tau, orig] = dlaed4(i, delta, z, rho)
% compute the i-th eigenvalue of the perturbed matrix D + 1/rho * z * z**T
% tau+orig gives the computed eigenvalue.
n = length(delta);
f = @(tau, orig) rho + sum(z.^2./(delta-orig-tau));

% make a good initial guess
[tau, orig] = initial_guess(i, delta, z, rho);
fprintf(1, 'f(%.12g + %.12g) = %.12g\n', orig, tau, f(tau, orig));

% iterations begin
while 1
    eta = middle_way(i, delta, z, rho, tau, orig);
    tau = tau + eta;
    fprintf(1, 'f(%.12g + %.12g) = %.12g\n', orig, tau, f(tau, orig));
    if i < n && abs(eta) <= 16 * eps * min(abs(delta(i)-orig-tau),...
        abs(delta(i+1)-orig-tau))
        break
    elseif i == n && abs(eta) <= 16 * eps * abs(delta(n)-orig-tau)
        break
    end
end
