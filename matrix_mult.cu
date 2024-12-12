#include "matrix_mult.h"
#include <iostream>

// CUDA kernel definition
__global__ void matMulKernel(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        float value = 0;
        for (int k = 0; k < colsA; ++k) {
            value += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = value;
    }
}

// Matrix multiplication wrapper definition
void matMul(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB) {
    float *d_A, *d_B, *d_C;
    size_t sizeA = rowsA * colsA * sizeof(float);
    size_t sizeB = colsA * colsB * sizeof(float);
    size_t sizeC = rowsA * colsB * sizeof(float);

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((colsB + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rowsA + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, colsB);

    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}