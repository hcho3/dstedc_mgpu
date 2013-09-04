[Q1, D1] = eig([2 1 0; 1 2 1; 0 1 1])
[Q2, D2] = eig([1 1 0; 1 2 1; 0 1 2])
D = [diag(D1); diag(D2)]                     
Q = [Q1 zeros(3,3); zeros(3,3) Q2]
z = [Q1(end, :)'; Q2(1, :)']
[K, D, Q, rho, w] = dlaed2(D, Q, 3, 1:6, 1, z)
