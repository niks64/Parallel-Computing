#pragma once
#include "CSRMatrix.h"
#include "Parameters.h"

void MatVecMultiply(CSRMatrix& mat, const float *x, float *y);
