#include "ConjugateGradients.h"
#include "Laplacian.h"
#include "Timer.h"
#include "Utilities.h"

Timer timerLaplacian;
Timer timerSaxpy;
Timer timerInner;

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
        timerLaplacian.Reset(); timerSaxpy.Reset(); timerInner.Reset();
        timer.Start();
        ConjugateGradients(matrix, x, f, p, r, z, false);
        timer.Pause();
        timerLaplacian.Print("Total Laplacian Time : ");
        timerSaxpy.Print("Total Saxpy Time : ");
        timerInner.Print("Total InnerProduct Time : ");
        timer.Print("Total Conjugate Gradients Time : ");
    }

    return 0;
}
