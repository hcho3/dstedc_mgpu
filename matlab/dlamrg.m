function [perm] = dlamrg(N1, N2, A, DTRD1, DTRD2)
% computes a permutation which merges two sorted lists A(1:N1) and A(N1+1:end)
% into a single sorted list in ascending order.

perm = zeros(1, N1+N2);

if DTRD1 > 0
    i = 1;
else
    i = N1;
end
if DTRD2 > 0
    j = 1 + N1;
else
    j = N1 + N2;
end
idx = 1;
while N1 > 0 && N2 > 0
    if A(i) <= A(j)
        perm(idx) = i;
        idx = idx + 1;
        i = i + DTRD1;
        N1 = N1 - 1;
    else
        perm(idx) = j;
        idx = idx + 1;
        j = j + DTRD2;
        N2 = N2 - 1;
    end
end
if N1 == 0
    for k=1:N2
        perm(idx) = j;
        idx = idx + 1;
        j = j + DTRD2;
    end
else  % N2 == 0
    for k=1:N1
        perm(idx) = i;
        idx = idx + 1;
        i = i + DTRD1;
    end
end
