#include "MatMatMultiply.h"
#include "mkl.h"

alignas(64) float localA[BLOCK_SIZE][BLOCK_SIZE];
alignas(64) float localB[BLOCK_SIZE][BLOCK_SIZE];
alignas(64) float localC[BLOCK_SIZE][BLOCK_SIZE];

#pragma omp threadprivate(localA, localB, localC)

void MatMatMultiply(const float (&A)[rMATRIX_SIZE_ONE][cMATRIX_SIZE_ONE],
    const float (&B)[rMATRIX_SIZE_TWO][cMATRIX_SIZE_TWO], float (&C)[rMATRIX_SIZE_ONE][cMATRIX_SIZE_TWO])
{
    static constexpr int rNBLOCKS_A = rMATRIX_SIZE_ONE / BLOCK_SIZE;
    static constexpr int cNBLOCKS_A = cMATRIX_SIZE_ONE / BLOCK_SIZE;
    static constexpr int rNBLOCKS_B = rMATRIX_SIZE_TWO / BLOCK_SIZE;
    static constexpr int cNBLOCKS_B = cMATRIX_SIZE_TWO / BLOCK_SIZE;
    static constexpr int rNBLOCKS_C = rMATRIX_SIZE_ONE / BLOCK_SIZE;
    static constexpr int cNBLOCKS_C = cMATRIX_SIZE_TWO / BLOCK_SIZE;

    using blocked_matrix_t_C = float (&) [rNBLOCKS_C][BLOCK_SIZE][cNBLOCKS_C][BLOCK_SIZE];
    using const_blocked_matrix_t_A = const float (&) [rNBLOCKS_A][BLOCK_SIZE][cNBLOCKS_A][BLOCK_SIZE];
    using const_blocked_matrix_t_B = const float (&) [rNBLOCKS_B][BLOCK_SIZE][cNBLOCKS_B][BLOCK_SIZE];

    auto blockA = reinterpret_cast<const_blocked_matrix_t_A>(A[0][0]);
    auto blockB = reinterpret_cast<const_blocked_matrix_t_B>(B[0][0]);
    auto blockC = reinterpret_cast<blocked_matrix_t_C>(C[0][0]);

#pragma omp parallel for
    for (int bi = 0; bi < rNBLOCKS_C; bi++)
    for (int bj = 0; bj < cNBLOCKS_C; bj++) {
        
        for (int ii = 0; ii < BLOCK_SIZE; ii++)
            for (int jj = 0; jj < BLOCK_SIZE; jj++) {
                localC[ii][jj] = 0.;
            }

        for (int bk = 0; bk < cNBLOCKS_A; bk++) { 

            for (int ii = 0; ii < BLOCK_SIZE; ii++)
            for (int jj = 0; jj < BLOCK_SIZE; jj++) {
                localA[ii][jj] = blockA[bi][ii][bk][jj];
                localB[ii][jj] = blockB[bk][ii][bj][jj];
            }

            for (int ii = 0; ii < BLOCK_SIZE; ii++)
                for (int kk = 0; kk < BLOCK_SIZE; kk++)
#pragma omp simd
                    for (int jj = 0; jj < BLOCK_SIZE; jj++)
                    localC[ii][jj] += localA[ii][kk] * localB[kk][jj];
        }

        for (int ii = 0; ii < BLOCK_SIZE; ii++)
        for (int jj = 0; jj < BLOCK_SIZE; jj++)                
            blockC[bi][ii][bj][jj] = localC[ii][jj];
    }
}

void MatMatMultiplyReference(const float (&A)[rMATRIX_SIZE_ONE][cMATRIX_SIZE_ONE],
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
