#include <math.h>
#include "dstedc.h"

double dlapy2(double x, double y)
// computes sqrt(x^2 + y^2) without causing overflow.
{
    double xabs = fabs(x);
    double yabs = fabs(y);
    double w = fmax(xabs, yabs);
    double z = fmin(xabs, yabs);
    double tau;

    if (z == 0.0)
        tau = w;
    else
        tau = w * sqrt(1.0 + SQ(z / w));

    return tau;
}
