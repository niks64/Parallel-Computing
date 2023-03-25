#include "Timer.h"
#include <iomanip>
#include <iostream>
#define XDIM 512
#define YDIM 512
#define ZDIM 512

// void Copy(const float (&x)[XDIM][YDIM][ZDIM], float (&y)[XDIM][YDIM][ZDIM])
// {
// #pragma omp parallel for    
//     for (int i = 1; i < XDIM-1; i++)
//     for (int j = 1; j < YDIM-1; j++)
//     for (int k = 1; k < ZDIM-1; k++)
//         y[i][j][k] = x[i][j][k] + y[i][j][k];
// }

int main() {
    using array_t = float (&) [XDIM][YDIM][ZDIM];
    
    float *uRaw = new float [XDIM*YDIM*ZDIM];
    float *LuRaw = new float [XDIM*YDIM*ZDIM];
    array_t u = reinterpret_cast<array_t>(*uRaw);
    array_t Lu = reinterpret_cast<array_t>(*LuRaw);

    std::cout << (sizeof(u) + sizeof(Lu)) / 1e9 << std::endl;

    // Timer timer;

    // for(int test = 1; test <= 10; test++)
    // {
    //     std::cout << "Running test iteration " << std::setw(2) << test << " ";
    //     timer.Start();
    //     Copy(u, Lu);
    //     timer.Stop("Elapsed time : ");
    // }
}