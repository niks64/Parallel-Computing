#include "Laplacian.h"
#include "Parameters.h"
#include "PointwiseOps.h"
#include "Reductions.h"
#include "Utilities.h"
#include "Timer.h"
#include "CombinedKernels.h"
#include <iostream>
#include <iomanip>
#include <string>

void printTimes(std::string fn, int line, double duration, int itr) {
    if (itr == 1) {std::cout << std::left << std::setw(32) << "[" + fn + " on line " + std::to_string(line) + ": " << duration << " ms]" << std::endl; return;}
    std::cout << std::left << std::setw(32) << "[" + fn + " on line " + std::to_string(line) + ": " <<
    duration << " ms] [Per Iteration: " << duration / itr << " ms]" << std::endl;
}

void ConjugateGradients(
    float (&x)[XDIM][YDIM][ZDIM],
    const float (&f)[XDIM][YDIM][ZDIM],
    float (&p)[XDIM][YDIM][ZDIM],
    float (&r)[XDIM][YDIM][ZDIM],
    float (&z)[XDIM][YDIM][ZDIM],
    const bool writeIterations)
{
    // Individual Kernel Timers
    Timer Laplacian2, Saxpy2, Norm2, Copy4, Inner4, Laplacian6, Inner6, Saxpy8, Norm8, Saxpy10, Copy13, Inner13, Saxpy16;
    
    // Combined Kernel Timers
    Timer Kernel2, Kernel4, Kernel6, Kernel8, Kernel13, Kernel16;

    Timer TotalTimer;
    TotalTimer.Start();

    // Algorithm : Line 2
    Kernel2.Start(); float nu = KernelLine2(x,z,f,r,-1); Kernel2.Pause();
    double Kernel2Duration = Kernel2.getElapsedTime();
    printTimes("Combined Kernel", 2, Kernel2Duration, 1);
    
    // Code for the individual kernels on line 2
    // Laplacian2.Start(); ComputeLaplacian(x, z); Laplacian2.Pause();
    // Saxpy2.Start(); Saxpy(z, f, r, -1); Saxpy2.Pause();
    // Norm2.Start(); float nu = Norm(r); Norm2.Pause();

    // Algorithm : Line 3
    if (nu < nuMax) return;
        
    // Algorithm : Line 4
    Kernel4.Start(); float rho = KernelLine4(r, p); Kernel4.Pause();
    double Kernel4Duration = Kernel4.getElapsedTime();
    printTimes("Combined Kernel", 4, Kernel4Duration, 1);

    
    // Code for individual kernels on line 4
    // Copy4.Start(); Copy(r, p); Copy4.Pause();
    // Inner4.Start(); float rho=InnerProduct(p, r); Inner4.Pause();
    
    int k = 0;    
    
    // Beginning of loop from Line 5
    for(k=0;;k++)
    {
        //std::cout << "Residual norm (nu) after " << k << " iterations = " << nu << std::endl;

        // Algorithm : Line 6
        if (k == 0) Kernel6.Start(); else Kernel6.Restart();
        float sigma = KernelLine6(p, z); Kernel6.Pause();

        // Individual Kernel code for line 6
        // if (k == 0) Laplacian6.Start(); else Laplacian6.Restart(); 
        // ComputeLaplacian(p, z); Laplacian6.Pause();
        // if (k == 0) Inner6.Start(); else Inner6.Restart(); 
        // float sigma=InnerProduct(p, z); Inner6.Pause();

        // Algorithm : Line 7
        float alpha=rho/sigma;

        // Algorithm : Line 8
        if (k == 0) Kernel8.Start(); else Kernel8.Restart(); 
        nu = KernelLine8(z, -alpha, r); Kernel8.Pause();
        
        // Individual kernel code for line 8
        // if (k == 0) Saxpy8.Start(); else Saxpy8.Restart(); 
        // Saxpy(z, r, r, -alpha); Saxpy8.Pause();
        // if (k == 0) Norm8.Start(); else Norm8.Restart(); 
        // nu=Norm(r); Norm8.Pause();

        // Algorithm : Lines 9-12
        if (nu < nuMax || k == kMax) {
            Saxpy10.Start(); Saxpy(p, x, x, alpha); Saxpy10.Pause();
            TotalTimer.Pause();
            //if (writeIterations) WriteAsImage("x", x, k, 0, 127);
            break;
        }
            
        // Algorithm : Line 13
        if (k == 0) Kernel13.Start(); else Kernel13.Restart(); 
        float rho_new = KernelLine4(r,z); Kernel13.Pause();

        // Individual kernel code for line 13
        // if (k == 0) Copy13.Start(); else Copy13.Restart(); 
        // Copy(r, z); Copy13.Pause();
        // if (k == 0) Inner13.Start(); else Inner13.Restart(); 
        // float rho_new = InnerProduct(z, r); Inner13.Pause();

        // Algorithm : Line 14
        float beta = rho_new/rho;

        // Algorithm : Line 15
        rho=rho_new;

        // Algorithm : Line 16
        if (k == 0) Kernel16.Start(); else Kernel16.Restart();
        KernelLine16(p, alpha, x, r, beta); Kernel16.Pause();
        
        // Individual kernel code for line 16
        // if (k == 0) Saxpy16.Start(); else Saxpy16.Restart(); 
        // Saxpy(p, x, x, alpha);
        // Saxpy(p, r, p, beta); Saxpy16.Pause();

        //if (writeIterations) WriteAsImage("x", x, k, 0, 127);
    }

    // Individual Kernel Durations
    
    // double Laplacian2Duration = Laplacian2.getElapsedTime();
    // double Saxpy2Duration = Saxpy2.getElapsedTime();
    // double Norm2Duration = Norm2.getElapsedTime();
    // double Copy4Duration = Copy4.getElapsedTime();
    // double Inner4Duration = Inner4.getElapsedTime();
    // double Laplacian6Duration = Laplacian6.getElapsedTime();
    // double Inner6Duration = Inner6.getElapsedTime();
    // double Saxpy8Duration = Saxpy8.getElapsedTime();
    // double Norm8Duration = Norm8.getElapsedTime();
    // double Copy13Duration = Copy13.getElapsedTime();
    // double Inner13Duration = Inner13.getElapsedTime();
    // double Saxpy16Duration = Saxpy16.getElapsedTime();

    // Combined Kernel Durations
    double Saxpy10Duration = Saxpy10.getElapsedTime();  
    double Kernel6Duration = Kernel6.getElapsedTime();
    double Kernel8Duration = Kernel8.getElapsedTime();
    double Kernel13Duration = Kernel13.getElapsedTime();
    double Kernel16Duration = Kernel16.getElapsedTime();

    // Combined Kernel prints
    printTimes("Combined Kernel", 6, Kernel6Duration, k);
    printTimes("Combined Kernel", 8, Kernel8Duration, k);
    printTimes("Saxpy()", 10, Saxpy10Duration, k);
    printTimes("Combined Kernel", 13, Kernel13Duration, k);
    printTimes("Combined Kernel", 16, Kernel16Duration, k);

    // Individual Kernel Prints
    // printTimes("ComputeLaplacian()", 2, Laplacian2Duration, 1);
    // printTimes("Saxpy()", 2, Saxpy2Duration, 1);
    // printTimes("Norm()", 2, Norm2Duration, 1);
    // printTimes("Copy()", 4, Copy4Duration, 1);
    // printTimes("InnerProduct()", 4, Inner4Duration, 1);
    // printTimes("ComputeLaplacian()", 6, Laplacian6Duration, k);
    // printTimes("InnerProduct()", 6, Inner6Duration, k);
    // printTimes("Saxpy()", 8, Saxpy8Duration, k);
    // printTimes("Norm()", 8, Norm8Duration, k);
    // printTimes("Copy()", 13, Copy13Duration, k);
    // printTimes("InnerProduct()", 13, Inner13Duration, k);
    // printTimes("2 Saxpy()", 16, Saxpy16Duration, k);
    
    std::cout << "Conjugate Gradients terminated after " << k << " iterations; residual norm (nu) = " << nu << std::endl;
    
    // Total Duration comparisons
    double TotalDuration = TotalTimer.getElapsedTime();
    double TotalKernelDuration = 0.0;
    TotalKernelDuration = Saxpy10Duration/* + Laplacian2Duration + Saxpy2Duration + Norm2Duration + Copy4Duration + Inner4Duration + Laplacian6Duration + Inner6Duration + Saxpy8Duration + Norm8Duration + Copy13Duration + Inner13Duration + Saxpy16Duration*/;
    TotalKernelDuration += Kernel2Duration + Kernel4Duration + Kernel6Duration + Kernel8Duration + Kernel13Duration + Kernel16Duration;

    std::cout << std::endl;
    std::cout << std::left << std::setw(32) << "[Total Duration: " << TotalDuration << " ms]" << std::endl;
    std::cout << std::left << std::setw(32) << "[Sum of Kernel Durations: " << TotalKernelDuration << " ms]" << std::endl;

}
