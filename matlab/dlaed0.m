function [D, Q] = dlaed0(D, E)
% computes all eigenvalues and corresponding eigenvectors of a symmetric
% tridiagonal matrix using the divide-and-conquer eigenvalue algorithm.
% We will have
%    diag(D(in)) + diag(E, 1) + diag(E, -1) = Q * diag(D(out)) + Q'.

N = length(D);
SMLSIZ = 25;
subpbs = 1;  % number of submatrices
partition = zeros(1, N);  % list of partition points
partition(1) = N;

Q = eye(N, N);
perm1 = zeros(1, N);

while partition(subpbs) > SMLSIZ
    for j=subpbs:-1:1
        partition(2*j) = floor((partition(j) + 1) / 2);
        partition(2*j - 1) = floor(partition(j) / 2);
    end
    subpbs = subpbs * 2;
end
for j=2:subpbs
    partition(j) = partition(j) + partition(j-1);
end

% Divide the matrix into subpbs submatricies of size at most SMLSIZ+1 using
% rank-1 modifications (cuts).

for i=1:subpbs-1
    submat = partition(i) + 1;
    smm1 = submat - 1;
    D(smm1) = D(smm1) - abs( E(smm1) );
    D(submat) = D(submat) - abs( E(smm1) );
end

% Solve each submatrix eigenvalue problem at the bottom of the divide and
% conquer tree.
curr = 0;
for i=0:subpbs-1
    if i == 0
        submat = 1;
        matsiz = partition(1);
    else
        submat = partition(i) + 1;
        matsiz = partition(i+1) - partition(i);
    end
    rlim = partition(i+1);
    [Qtemp, Dtemp] = ...
        eig( diag(D(submat:rlim)) + diag(E(submat:rlim-1), 1) + ...
             diag(E(submat:rlim-1), -1) );
    [D(submat:rlim), I] = sort(diag(Dtemp));
    Q(submat:rlim, submat:rlim) = Qtemp(:, I);
    k = 1;
    for j=submat:partition(i+1)
        perm1(j) = k;
        k = k + 1;
    end
end

% Successively merge eigensystems of adjacent submatrices into eigensystem for
% the corresponding larger matrix.
while subpbs > 1
    for i=0:2:subpbs-2
        if i == 0
            submat = 1;
            matsiz = partition(2);
            msd2 = partition(1);
            curprb = 0;
        else
            submat = partition(i) + 1;
            matsiz = partition(i+2) - partition(i);
            msd2 = floor(matsiz / 2);
            curprb = curprb + 1;
        end

        % Merge lower order eigensystems (of size msd2 and matsiz - msd2) into
        % an eigensystem of size matsiz.
        rlim = partition(i+2);
        old_d = D(submat:rlim);
        old_q = Q(submat:rlim, submat:rlim);
        old_perm1 = perm1(submat:rlim);
        [D(submat:rlim), Q(submat:rlim, submat:rlim), perm1(submat:rlim)] = ...
            dlaed1(D(submat:rlim), Q(submat:rlim, submat:rlim), ...
            perm1(submat:rlim), E(submat+msd2-1), msd2);

        partition(i/2+1) = partition(i+2);
    end
    subpbs = subpbs / 2;
end

% Re-merge the eigenvalues and eigenvectors which were deflated at the final
% merge step.
D = D(perm1);
Q = Q(:, perm1);
