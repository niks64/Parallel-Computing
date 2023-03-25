#include "Parameters.h"
#include "Timer.h"

float KernelLine2(
    const float (&x)[XDIM][YDIM][ZDIM], 
    float (&z)[XDIM][YDIM][ZDIM],
    const float (&f)[XDIM][YDIM][ZDIM],
    float (&r)[XDIM][YDIM][ZDIM],
    const float scale) 
{
    float result = 0.;

    // Laplacian, Saxpy and Norm
    #pragma omp parallel for reduction(max:result)
        for (int i = 1; i < XDIM-1; i++)
        for (int j = 1; j < YDIM-1; j++)
        for (int k = 1; k < ZDIM-1; k++) {
            z[i][j][k] = -6 * x[i][j][k]
                + x[i+1][j][k]
                + x[i-1][j][k]
                + x[i][j+1][k]
                + x[i][j-1][k]
                + x[i][j][k+1]
                + x[i][j][k-1];
            r[i][j][k] = z[i][j][k] * scale + f[i][j][k];    
            result = std::max(result, std::abs(r[i][j][k]));}

    return result; 
}

float KernelLine4(
    float (&r)[XDIM][YDIM][ZDIM],
    float (&p)[XDIM][YDIM][ZDIM]) 
{
    
    // Copy and Inner Product    
    double result = 0.;
    #pragma omp parallel for reduction(+:result)
        for (int i = 1; i < XDIM-1; i++)
        for (int j = 1; j < YDIM-1; j++)
        for (int k = 1; k < ZDIM-1; k++) {
            p[i][j][k] = r[i][j][k];
            result += (double) r[i][j][k] * (double) p[i][j][k];
        }
    return (float) result;
}

float KernelLine6(
    float (&p)[XDIM][YDIM][ZDIM],
    float (&z)[XDIM][YDIM][ZDIM])
{
    double result = 0.;

    // Laplacian and Inner Product
    #pragma omp parallel for reduction(+:result)
        for (int i = 1; i < XDIM-1; i++)
        for (int j = 1; j < YDIM-1; j++)
        for (int k = 1; k < ZDIM-1; k++) {
            z[i][j][k] =
            -6 * p[i][j][k]
            + p[i+1][j][k]
            + p[i-1][j][k]
            + p[i][j+1][k]
            + p[i][j-1][k]
            + p[i][j][k+1]
            + p[i][j][k-1];
            result += (double) p[i][j][k] * (double) z[i][j][k];
        }
    
    return (float) result;
}

float KernelLine8(
    const float (&z)[XDIM][YDIM][ZDIM],
    const float scale,
    float (&r)[XDIM][YDIM][ZDIM])
{
    float result = 0.;

    // Saxpy and Norm
    #pragma omp parallel for reduction(max:result)
        for (int i = 1; i < XDIM-1; i++)
        for (int j = 1; j < YDIM-1; j++)
        for (int k = 1; k < ZDIM-1; k++) {
            r[i][j][k] += z[i][j][k] * scale;
            result = std::max(result, std::abs(r[i][j][k]));
        }

    return result;
}

void KernelLine16(
    float (&p)[XDIM][YDIM][ZDIM],
    const float scale1,
    float (&x)[XDIM][YDIM][ZDIM],
    const float (&r)[XDIM][YDIM][ZDIM],
    const float scale2) 
{
    #pragma omp parallel for
        for (int i = 1; i < XDIM-1; i++)
        for (int j = 1; j < YDIM-1; j++)
        for (int k = 1; k < ZDIM-1; k++) {
            x[i][j][k] += p[i][j][k] * scale1;
            p[i][j][k] = p[i][j][k] * scale2 + r[i][j][k];
        }
}