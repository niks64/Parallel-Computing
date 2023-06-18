#pragma once

#include "Parameters.h"

#include <cstdlib>

void* AlignedAllocate(const std::size_t size, const std::size_t alignment);
void InitializeMatrices(float (&A)[rMATRIX_SIZE_ONE][cMATRIX_SIZE_ONE],float (&B)[rMATRIX_SIZE_TWO][cMATRIX_SIZE_TWO]);
float MatrixMaxDifference(const float (&A)[rMATRIX_SIZE_ONE][cMATRIX_SIZE_TWO],const float (&B)[rMATRIX_SIZE_ONE][cMATRIX_SIZE_TWO]);
