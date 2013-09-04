f = @(tau, orig) rho + sum(zeta.^2./(delta-orig-tau));
for k=1:n
    fprintf(1, 'k = %d\n', k);
    [tau, orig] = initial_guess(k, delta, zeta, rho);
    fprintf(1, 'f(%.12g + %.12g) = %.12g\n', orig, tau, f(tau, orig));
end
