init_example
f = @(tau, orig) rho + sum(zeta.^2./(delta-orig-tau));
for k=1:n
    fprintf(1, 'k = %d\n', k);
    % make a good initial guess
    [tau, orig] = initial_guess(k, delta, zeta, rho);
    fprintf(1, 'f(%.12g + %.12g) = %.12g\n', orig, tau, f(tau, orig));
    
    % iterations begin
    while 1
        eta = middle_way(k, delta, zeta, rho, tau, orig);
        tau = tau + eta;
        fprintf(1, 'f(%.12g + %.12g) = %.12g\n', orig, tau, f(tau, orig));
        if k < n && abs(eta) <= 16 * eps * min(abs(delta(k)-orig-tau),...
            abs(delta(k+1)-orig-tau))
            break
        elseif k == n && abs(eta) <= 16 * eps * abs(delta(n)-orig-tau)
            break
        end
    end
end
