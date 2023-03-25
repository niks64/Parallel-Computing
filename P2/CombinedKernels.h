#pragma once

#include "Parameters.h"

float KernelLine2(
    const float (&x)[XDIM][YDIM][ZDIM], 
    float (&z)[XDIM][YDIM][ZDIM],
    const float (&f)[XDIM][YDIM][ZDIM],
    float (&r)[XDIM][YDIM][ZDIM],
    const float scale
    );

float KernelLine4(
    float (&r)[XDIM][YDIM][ZDIM],
    float (&p)[XDIM][YDIM][ZDIM]
    );

float KernelLine6(
    float (&p)[XDIM][YDIM][ZDIM],
    float (&z)[XDIM][YDIM][ZDIM]);

float KernelLine8(
    const float (&z)[XDIM][YDIM][ZDIM],
    const float scale,
    float (&r)[XDIM][YDIM][ZDIM] 
);

void KernelLine16(
    float (&p)[XDIM][YDIM][ZDIM],
    const float scale1,
    float (&x)[XDIM][YDIM][ZDIM],
    const float (&r)[XDIM][YDIM][ZDIM],
    const float scale2
);