#include "MatMatMultiply.h"
#include "mkl.h"

void MatMatMultiply(const float (&A)[rMATRIX_SIZE_ONE][cMATRIX_SIZE_ONE],
    const float (&B)[rMATRIX_SIZE_TWO][cMATRIX_SIZE_TWO], float (&C)[rMATRIX_SIZE_ONE][cMATRIX_SIZE_TWO])
{
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        rMATRIX_SIZE_ONE,
        cMATRIX_SIZE_TWO,
        cMATRIX_SIZE_ONE,
        1.,
        &A[0][0],
        cMATRIX_SIZE_ONE,
        &B[0][0],
        cMATRIX_SIZE_TWO,
        0.,
        &C[0][0],
        cMATRIX_SIZE_TWO
    );
}
