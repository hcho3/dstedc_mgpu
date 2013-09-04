function [tau] = dlapy2(x, y)
% computes sqrt(x^2 + y^2) without causing overflow.

xabs = abs(x);
yabs = abs(y);
w = max(xabs, yabs);
z = min(xabs, yabs);
if z == 0.0
    tau = w;
else
    tau = w * sqrt(1.0 + (z / w)^2);
end
