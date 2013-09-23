function [D, Q] = dstedc(D, E)

N = length(D);
SMLSIZ = 25;

if N <= SMLSIZ
    [Q, Dtemp] = eig(diag(D) + diag(E, 1) + diag(E, -1));
    D = diag(Dtemp);
    return;
end

Q = eye(N, N);

start = 1;
while start <= N
    % Let finish be the position of the next subdiagonal entry such that
    % E(finish) <= tiny or finish = N if no such subdiagonal exists. The
    % matrix identified by the elements between start and finish constitutes
    % an independent sub-problem.
    finish = start;
    while finish < N
        tiny = eps * sqrt( abs(D(finish)) ) * sqrt( abs(D(finish+1)) );
        if abs(E(finish)) > tiny
            finish = finish + 1;
        else
            break;
        end
    end

    % (Sub) Problem determined. Compute its size and solve it.
    M = finish - start + 1
    if M == 1
        start = finish + 1;
        continue;
    end
    if M > SMLSIZ
        % Scale.
        orgnrm = max([max(abs(D(start:finish))), max(abs(E(start:finish-1)))])
        D(start:finish) = D(start:finish) / orgnrm;
        E(start:finish-1) = E(start:finish-1) / orgnrm;
        [D(start:finish), Q(start:finish, start:finish)] = ...
            dlaed0(D(start:finish), E(start:finish-1));
        
        % Scale back.
        D(start:finish) = D(start:finish) * orgnrm;
    else
        [Q(start:finish, start:finish), Dtemp] = ...
            eig(diag(D(start:finish)) + diag(E(start:finish-1), 1) + ...
                diag(E(start:finish-1), -1));
        D(start:finish) = diag(Dtemp);
    end
    start = finish + 1;
end

% If the problem split any number of times, then the eigenvalues will not be
% properly ordered. Here we permute the eigenvalues (and the associated
% eigenvectors) into ascending order.
if M ~= N
    % Use Selection Sort to minimize swaps of eigenvectors
    for ii=2:N
        i = ii - 1;
        k = i;
        p = D(i);
        for j=ii:N
            if D(j) < p
                k = j;
                p = D(j);
            end
        end
        if k ~= i
            D(k) = D(i);
            D(i) = p;
            tmp = Q(:, i);
            Q(:, i) = Q(:, k);
            Q(:, k) = tmp;
        end
    end
end
