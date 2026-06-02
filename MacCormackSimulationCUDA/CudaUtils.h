#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// Helper macro for checking CUDA errors
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

inline void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " (" << cudaGetErrorString(result) << ") "
            << "at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset(); // Reset device on error
        throw std::runtime_error("CUDA error");
    }
}