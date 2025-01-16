#include <iostream>
#include <cmath>
#include <limits>
#include <cstring>
#include <random>
#include <cfenv>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template<typename T>
int ulp_difference(T computed, T expected) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "Only float and double are supported.");

    using int_type = typename std::conditional<sizeof(T) == 4, int32_t, int64_t>::type;
    int_type computed_bits, expected_bits;
    std::memcpy(&computed_bits, &computed, sizeof(T));
    std::memcpy(&expected_bits, &expected, sizeof(T));

    return std::abs(computed_bits - expected_bits);
}


__global__ void test_exp2f_precision(const float *input, float *output_exp2f, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float x = input[idx];
        output_exp2f[idx] = exp2f(x);  // 计算单精度
    }
}

template<typename T>
T generateRandomFloatInRange(T min,  T max) {
    // 生成 [min, max] 之间的随机浮点数
    T random = static_cast<T>(std::rand()) / static_cast<T>(RAND_MAX);
    return min + random * (max - min);
}

int main() {
    //const int N = 1024;
    const int N = 1024 * 1024;
    using Element = float;
    thrust::host_vector<Element> h_S(N);
    thrust::host_vector<Element> h_D(N);
    thrust::host_vector<Element> h_REF(N);

    std::srand(static_cast<unsigned>(std::time(0)));

    //std::fesetround(FE_UPWARD);  // 默认模式也是 FE_TONEAREST


    int currentRoundMode = fegetround();  
    
    // 根据不同的舍入模式输出相应的值  
    switch (currentRoundMode) {  
        case FE_TONEAREST:  
            std::cout << "当前舍入模式: FE_TONEAREST" << std::endl;  
            break;  
        case FE_DOWNWARD:  
            std::cout << "当前舍入模式: FE_DOWNWARD" << std::endl;  
            break;  
        case FE_UPWARD:  
            std::cout << "当前舍入模式: FE_UPWARD" << std::endl;  
            break;  
        case FE_TOWARDZERO:  
            std::cout << "当前舍入模式: FE_TOWARDZERO" << std::endl;  
            break;  
        default:  
            std::cout << "未知舍入模式" << std::endl;  
            break;  
    }  


    for (size_t i = 0; i < h_S.size(); ++i) {
        h_S[i] = generateRandomFloatInRange(-64.0f, 64.0f);
        h_D[i] = Element{};
        h_REF[i] = exp2f(h_S[i]);
    }

    thrust::device_vector<Element> d_S = h_S;
    thrust::device_vector<Element> d_D = h_D;
    
    test_exp2f_precision<<<(N + 255) / 256, 256>>>( thrust::raw_pointer_cast(d_S.data()), thrust::raw_pointer_cast(d_D.data()), N);
    cudaDeviceSynchronize();
    h_D = d_D;

    std::cout <<  std::hexfloat;
    for (size_t i = 0; i < N; ++i) {
        int ulp = ulp_difference(h_D[i], h_REF[i]);
        if (ulp > 2) {
            std::cout << "Error: " << h_S[i] << " -> " << h_D[i] << " vs. " << h_REF[i] << " (" << ulp << " ulp)\n";
        }
    }  

    return 0;
}

