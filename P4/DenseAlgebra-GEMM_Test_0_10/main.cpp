#include "MatMatMultiply.h"
#include "Timer.h"
#include "Utilities.h"

#include <iostream>
#include <iomanip>
#include <algorithm>

int main(int argc, char *argv[])
{
    float *Araw = static_cast<float*>( AlignedAllocate( rMATRIX_SIZE_ONE * cMATRIX_SIZE_ONE * sizeof(float), 64 ) );
    float *Braw = static_cast<float*>( AlignedAllocate( rMATRIX_SIZE_TWO * cMATRIX_SIZE_TWO * sizeof(float), 64 ) );
    float *Craw = static_cast<float*>( AlignedAllocate( rMATRIX_SIZE_ONE * cMATRIX_SIZE_TWO * sizeof(float), 64 ) );
    float *referenceCraw = static_cast<float*>( AlignedAllocate( rMATRIX_SIZE_ONE * cMATRIX_SIZE_TWO * sizeof(float), 64 ) );

    using matrix_t_A = float (&) [rMATRIX_SIZE_ONE][cMATRIX_SIZE_ONE];
    using matrix_t_B = float (&) [rMATRIX_SIZE_TWO][cMATRIX_SIZE_TWO];
    using matrix_t_C = float (&) [rMATRIX_SIZE_ONE][cMATRIX_SIZE_TWO];

    matrix_t_A A = reinterpret_cast<matrix_t_A>(*Araw);
    matrix_t_B B = reinterpret_cast<matrix_t_B>(*Braw);
    matrix_t_C C = reinterpret_cast<matrix_t_C>(*Craw);
    matrix_t_C referenceC = reinterpret_cast<matrix_t_C>(*referenceCraw);

    InitializeMatrices(A, B);
    Timer timer;

    // Correctness test
    std::cout << "Running candidate kernel for correctness test ... " << std::flush;
    timer.Start();
    MatMatMultiply(A, B, C);
    timer.Stop("Elapsed time : ");
    
    std::cout << "Running reference kernel for correctness test ... " << std::flush;
    timer.Start();
    MatMatMultiplyReference(A, B, referenceC);
    timer.Stop("Elapsed time : ");

    float discrepancy = MatrixMaxDifference(C, referenceC);
    std::cout << "Discrepancy between two methods : " << discrepancy << std::endl;
    
    double minTime = 100000;
    for(int test = 1; test <= 15; test++)
    {
        std::cout << "Running kernel for performance run #" << std::setw(2) << test << " ... ";
        timer.Start();
        MatMatMultiply(A, B, C);
        double curr = timer.Stop("Elapsed time : ");
        minTime = std::min(minTime, curr);
    }

    std::cout << "Minimum time out of the runs: " << minTime << "ms" << std::endl;
    
    return 0;
}
