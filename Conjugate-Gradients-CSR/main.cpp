#include "ConjugateGradients.h"
#include "Laplacian.h"
#include "Timer.h"
#include "Utilities.h"
// Added includes
#include <iostream>
#include "PointwiseOps.h"
#include "Reductions.h"

Timer timerLaplacian, timerSaxpy, timerInner, timerNorm, timerCopy, timerSaxpyV1, timerSaxpyV2, timerSaxpyV3;


int main(int argc, char *argv[])
{
    using array_t = float (&) [XDIM][YDIM][ZDIM];

    float *xRaw = new float [XDIM*YDIM*ZDIM];
    float *fRaw = new float [XDIM*YDIM*ZDIM];
    float *pRaw = new float [XDIM*YDIM*ZDIM];
    float *rRaw = new float [XDIM*YDIM*ZDIM];
    float *zRaw = new float [XDIM*YDIM*ZDIM];
    
    array_t x = reinterpret_cast<array_t>(*xRaw);
    array_t f = reinterpret_cast<array_t>(*fRaw);
    array_t p = reinterpret_cast<array_t>(*pRaw);
    array_t r = reinterpret_cast<array_t>(*rRaw);
    array_t z = reinterpret_cast<array_t>(*zRaw);
    
    CSRMatrix matrix;

    // Initialization
    {
        Timer timer;
        timer.Start();
        InitializeProblem(x, f);
        matrix = BuildLaplacianMatrix(); // This takes a while ...
        timer.Stop("Initialization : ");
        timer.Restart();
    }

    
    // Call Conjugate Gradients algorithm
    {	
        Timer timer;
        timerLaplacian.Reset(); timerSaxpy.Reset(); timerInner.Reset(); timerNorm.Reset(); timerCopy.Reset(); timerSaxpyV1.Reset(); timerSaxpyV2.Reset(); timerSaxpyV3.Reset();
        timer.Start();
        ConjugateGradients(matrix, x, f, p, r, z, false);
        timer.Pause();
        timerLaplacian.Print("Total Laplacian Time : ");
        timerInner.Print("Total InnerProduct Time : ");
        timerNorm.Print("Total Norm Time : ");
        timerCopy.Print("Total Copy Time : ");
        timerSaxpyV1.Print("Total Saxpy Version 1 Time : ");
        timerSaxpyV2.Print("Total Saxpy Version 2 Time : ");
        timerSaxpyV3.Print("Total Saxpy Version 3 Time : ");
        timer.Print("Total Conjugate Gradients Time : ");
    }
    

    return 0;
}
