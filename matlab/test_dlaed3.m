[lambda, Q] = dlaed3(delta, zeta, rho);
lambda
max(max(abs(diag(delta) + (1/rho) * zeta * zeta' - Q * diag(lambda) * Q')))
