#include "Reductions.h"

#include <algorithm>
#ifndef DO_NOT_USE_MKL
#include <mkl.h>
#endif

float Norm(const float (&x)[XDIM][YDIM][ZDIM])
{
    float result = 0.;

#ifdef DO_NOT_USE_MKL
#pragma omp parallel for reduction(max:result)
    for (int i = 1; i < XDIM-1; i++)
    for (int j = 1; j < YDIM-1; j++)
    for (int k = 1; k < ZDIM-1; k++)
        result = std::max(result, std::abs(x[i][j][k]));

    return result;
#else
    int imax = cblas_isamax(XDIM*YDIM*ZDIM, &x[0][0][0], 1);
    
    int k = imax % ZDIM;
    int j = (imax / ZDIM) % YDIM;
    int i = (imax / (ZDIM*YDIM)) % XDIM;

    return std::abs(x[i][j][k]);
#endif
}

float InnerProduct(const float (&x)[XDIM][YDIM][ZDIM], const float (&y)[XDIM][YDIM][ZDIM])
{
    double result = 0.;

#ifdef DO_NOT_USE_MKL
#pragma omp parallel for reduction(+:result)
    for (int i = 1; i < XDIM-1; i++)
    for (int j = 1; j < YDIM-1; j++)
    for (int k = 1; k < ZDIM-1; k++)
        result += (double) x[i][j][k] * (double) y[i][j][k];

    return (float) result;
#else
    return cblas_sdsdot(
        XDIM*YDIM*ZDIM,
        0.,
        &x[0][0][0],
        1,
        &y[0][0][0],
        1
    );
#endif
}
