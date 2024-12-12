// matrix_mult.h

#ifndef MATRIX_MULT_H
#define MATRIX_MULT_H

#include <cuda_runtime.h>

// CUDA kernel declaration for matrix multiplication
__global__ void matMulKernel(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB);

// Matrix multiplication wrapper declaration
void matMul(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB);

#endif // MATRIX_MULT_H
