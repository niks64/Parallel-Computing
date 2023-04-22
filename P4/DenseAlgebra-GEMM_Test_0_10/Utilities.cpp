#include "Utilities.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <new>
#include <random>

void* AlignedAllocate(const std::size_t size, const std::size_t alignment)
{
    std::size_t capacity = size + alignment - 1;
    void *ptr = new char[capacity];
    auto result = std::align(alignment, size, ptr, capacity);
    if (result == nullptr) throw std::bad_alloc();
    if (capacity < size) throw std::bad_alloc();
    return ptr;
}

void InitializeMatrices(float (&A)[rMATRIX_SIZE_ONE][cMATRIX_SIZE_ONE],float (&B)[rMATRIX_SIZE_TWO][cMATRIX_SIZE_TWO])
{
    std::random_device rd; std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform_dist(-1., 1.);

    for (int i = 0; i < rMATRIX_SIZE_ONE; i++)
        for (int j = 0; j < cMATRIX_SIZE_ONE; j++) {
            A[i][j] = uniform_dist(gen);
        }

    for (int i = 0; i < rMATRIX_SIZE_TWO; i++)
        for (int j = 0; j < cMATRIX_SIZE_TWO; j++) {
            B[i][j] = uniform_dist(gen);
        }
}

float MatrixMaxDifference(const float (&A)[rMATRIX_SIZE_ONE][cMATRIX_SIZE_TWO],const float (&B)[rMATRIX_SIZE_ONE][cMATRIX_SIZE_TWO])
{
    float result = 0.;
    for (int i = 0; i < rMATRIX_SIZE_ONE; i++)
    for (int j = 0; j < cMATRIX_SIZE_TWO; j++)
        result = std::max( result, std::abs( A[i][j] - B[i][j] ) );
    return result;
}
