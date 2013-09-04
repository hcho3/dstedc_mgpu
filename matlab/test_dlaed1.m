[Q1, D1] = eig([2 1 0; 1 2 1; 0 1 1])
[Q2, D2] = eig([1 1 0; 1 2 1; 0 1 2])
D = [diag(D1); diag(D2)]                     
Q = [Q1 zeros(3,3); zeros(3,3) Q2]
rho = 1;
[D, Q, perm1] = dlaed1(D, Q, [1:3 1:3], rho, 3)
