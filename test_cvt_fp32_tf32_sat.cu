#include <iostream>
#include <cmath>
#include <limits>
#include <cstring>
#include <random>
#include <cfenv>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <float.h>



__global__ void test_fp32_tf32_cvt(const float  *input, float *output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        unsigned int storage = reinterpret_cast<unsigned const &>(input[tid]);
        asm volatile ("cvt.rna.satfinite.tf32.f32  %0, %1 ;" : "=r"(storage) : "r"(storage));
        output[tid] = storage;
    }
}

int main() {
    //const int N = 1024;
    const int N = 1;
    using Element = float;
    thrust::host_vector<Element> h_S(N);
    thrust::host_vector<Element> h_D(N);


   // minsubnormaltf32 = ldexp(1., -136), // smallest subnormal tf32
   // minsubnormal32 = ldexp(1., -149), // smallest subnormal binary32
   // belowone = nextafterf(1., 0.) ,   // largest float smaller than 1.0
   // gapbelowone = 1. - belowone,
   // aboveone = nextafterf(1., 2.),    // smallest float larger than 1.0
   // belowtwo = 2. - ldexp(1., -23);   // largest float smaller than 2.0


    for (size_t i = 0; i < h_S.size(); ++i) {
        h_S[i] = FLT_MAX;
        h_D[i] = Element{};
        std::cout <<  std::hexfloat;
        std::cout << "init value: "<< h_S[i] << std::endl;
    }

    thrust::device_vector<Element> d_S = h_S;
    thrust::device_vector<Element> d_D = h_D;

    //test_exp2f_precision<<<(N + 255) / 256, 256>>>( thrust::raw_pointer_cast(d_S.data()), thrust::raw_pointer_cast(d_D.data()), N);
    test_fp32_tf32_cvt<<<1,1>>>(thrust::raw_pointer_cast(d_S.data()), thrust::raw_pointer_cast(d_D.data()), N);

    cudaDeviceSynchronize();
    h_D = d_D;

    std::cout <<  std::hexfloat;
    for (size_t i = 0; i < h_S.size(); ++i) {
        std::cout << "init value: "<< h_D[i] << std::endl;
    }

    return 0;
}
