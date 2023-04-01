#include "ConjugateGradients.h"
#include "Laplacian.h"
#include "PointwiseOps.h"
#include "Reductions.h"
#include "Utilities.h"
#include "Timer.h"

#include <iostream>

extern Timer timerLaplacian, timerSaxpy, timerInner, timerNorm, timerCopy, timerSaxpyV1, timerSaxpyV2, timerSaxpyV3;

void ConjugateGradients(
    CSRMatrix& matrix,
    float (&x)[XDIM][YDIM][ZDIM],
    const float (&f)[XDIM][YDIM][ZDIM],
    float (&p)[XDIM][YDIM][ZDIM],
    float (&r)[XDIM][YDIM][ZDIM],
    float (&z)[XDIM][YDIM][ZDIM],
    const bool writeIterations)
{
    // Algorithm : Line 2
    timerLaplacian.Restart(); ComputeLaplacian(matrix, x, z); timerLaplacian.Pause();
    timerSaxpyV1.Restart(); Saxpy(z, f, r, -1); timerSaxpyV1.Pause(); // version 1
    timerNorm.Restart(); float nu = Norm(r); timerNorm.Pause();

    // Algorithm : Line 3
    if (nu < nuMax) return;
        
    // Algorithm : Line 4
    timerCopy.Restart(); Copy(r, p); timerCopy.Pause();
    timerInner.Restart(); float rho=InnerProduct(p, r); timerInner.Pause();
        
    // Beginning of loop from Line 5
    for(int k=0;;k++)
    {
        //std::cout << "Residual norm (nu) after " << k << " iterations = " << nu << std::endl;

        // Algorithm : Line 6
        timerLaplacian.Restart(); ComputeLaplacian(matrix, p, z); timerLaplacian.Pause();
        timerInner.Restart(); float sigma=InnerProduct(p, z); timerInner.Pause();

        // Algorithm : Line 7
        float alpha=rho/sigma;

        // Algorithm : Line 8
        timerSaxpyV2.Restart(); Saxpy(z, r, -alpha); timerSaxpyV2.Pause(); // version 2
        timerNorm.Restart(); nu=Norm(r); timerNorm.Pause();

        // Algorithm : Lines 9-12
        if (nu < nuMax || k == kMax) {
            timerSaxpyV2.Restart(); Saxpy(p, x, alpha); timerSaxpyV2.Pause(); // version 2
            std::cout << "Conjugate Gradients terminated after " << k << " iterations; residual norm (nu) = " << nu << std::endl;
            if (writeIterations) WriteAsImage("x", x, k, 0, 127);
            return;
        }
            
        // Algorithm : Line 13
        timerCopy.Restart(); Copy(r, z); timerCopy.Pause();
        timerInner.Restart(); float rho_new = InnerProduct(z, r); timerInner.Pause();

        // Algorithm : Line 14
        float beta = rho_new/rho;

        // Algorithm : Line 15
        rho=rho_new;

        // Algorithm : Line 16
        timerSaxpyV2.Restart(); Saxpy(p, x, alpha); timerSaxpyV2.Pause(); // version 2
        // Note: this used to be 
        // Saxpy(p, x, x, alpha);
        // The version above uses the fact that the destination vector is the same
        // as the second input vector -- i.e. Saxpy(x, y, c) performs
        // the operation y += c * x
        timerSaxpyV3.Restart(); SaxpyV3(r, p, beta); timerSaxpyV3.Pause();// version 1

        if (writeIterations) WriteAsImage("x", x, k, 0, 127);
    }

}
