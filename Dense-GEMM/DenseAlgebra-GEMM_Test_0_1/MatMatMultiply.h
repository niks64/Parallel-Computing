#pragma once

#include "Parameters.h"

void MatMatMultiply(const float (&A)[rMATRIX_SIZE_ONE][cMATRIX_SIZE_ONE],
    const float (&B)[rMATRIX_SIZE_TWO][cMATRIX_SIZE_TWO], float (&C)[rMATRIX_SIZE_ONE][cMATRIX_SIZE_TWO]);
