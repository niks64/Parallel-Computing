#include "PointwiseOps.h"

// #define DO_NOT_USE_MKL
#ifndef DO_NOT_USE_MKL
#include <mkl.h>
#endif

void Copy(const float (&x)[XDIM][YDIM][ZDIM], float (&y)[XDIM][YDIM][ZDIM])
{
#ifdef DO_NOT_USE_MKL
#pragma omp parallel for    
    for (int i = 1; i < XDIM-1; i++)
    for (int j = 1; j < YDIM-1; j++)
    for (int k = 1; k < ZDIM-1; k++)
        y[i][j][k] = x[i][j][k];
#else
    cblas_scopy(
        XDIM * YDIM * ZDIM,
        &x[0][0][0],
        1,
        &y[0][0][0],
        1
    );
#endif
}

// Version 1
void Saxpy(const float (&x)[XDIM][YDIM][ZDIM], const float (&y)[XDIM][YDIM][ZDIM],
    float (&z)[XDIM][YDIM][ZDIM],
    const float scale)
{
#ifdef DO_NOT_USE_MKL    
#pragma omp parallel for
    for (int i = 0; i < XDIM; i++)
    for (int j = 0; j < YDIM; j++)
    for (int k = 0; k < ZDIM; k++)
        z[i][j][k] = x[i][j][k] * scale + y[i][j][k];
#else
    // z = y
    cblas_scopy(
        XDIM * YDIM * ZDIM,
        &y[0][0][0],
        1,
        &z[0][0][0],
        1
    );
    // z += scale * x
    cblas_saxpy(
        XDIM * YDIM * ZDIM,
        scale,
        &x[0][0][0],
        1,
        &z[0][0][0],
        1
    );
#endif
}

// Version 2
void Saxpy(const float (&x)[XDIM][YDIM][ZDIM], float (&y)[XDIM][YDIM][ZDIM],
    const float scale)
{

#ifdef DO_NOT_USE_MKL    
    // Just for reference -- implementation without MKL
#pragma omp parallel for
    for (int i = 0; i < XDIM; i++)
    for (int j = 0; j < YDIM; j++)
    for (int k = 0; k < ZDIM; k++)
        y[i][j][k] += x[i][j][k] * scale;
#else
    cblas_saxpy(
        XDIM * YDIM * ZDIM, // Length of vectors
        scale,              // Scale factor
        &x[0][0][0],        // Input vector x, in operation y := x * scale + y
        1,                  // Use step 1 for x
        &y[0][0][0],        // Input/output vector y, in operation y := x * scale + y
        1                   // Use step 2 for y
    );
#endif
}

// Version 3
void SaxpyV3(const float (&x)[XDIM][YDIM][ZDIM], float (&y)[XDIM][YDIM][ZDIM],
    const float scale)
{
    // y = scale * y + x
#ifdef DO_NOT_USE_MKL
    Saxpy(y,x,y,scale);
#else
    // y = scale * y
    cblas_sscal(
        XDIM * YDIM * ZDIM,
        scale,
        &y[0][0][0],
        1
    );
    // y += 1.0 * x
    cblas_saxpy(
        XDIM * YDIM * ZDIM,
        1.0f,
        &x[0][0][0],
        1,
        &y[0][0][0],
        1
    );
#endif
}


