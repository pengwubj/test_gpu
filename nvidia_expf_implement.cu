#include <cmath>
#include <iostream>
#include <limits>
#include <cfenv>

int main() {
    // Initialize variables
    float f1 = 3.0f; // This value should be set based on your actual computation
    uint32_t tmp_f2  = 0x3F000000; //0.5 
    uint32_t tmp_f3  = 0x3BBB989D; //0.00572498
    uint32_t tmp_f6  = 0x4B400001; //1.255829e+07
    uint32_t tmp_f7  = 0x437C0000; //252
    uint32_t tmp_f11 = 0x3FB8AA3B; //1.4427
    uint32_t tmp_f13 = 0x32A57060; //1.92596e-08
    uint32_t tmp_fx  = 0xCB40007F;  //-1.2583e+07

    float f2  = *reinterpret_cast<float*>(&tmp_f2);
    float f3  = *reinterpret_cast<float*>(&tmp_f3);
    float f6  = *reinterpret_cast<float*>(&tmp_f6);
    float f7  = *reinterpret_cast<float*>(&tmp_f7);
    float f11 = *reinterpret_cast<float*>(&tmp_f11);
    float f13 = *reinterpret_cast<float*>(&tmp_f13);
    float fx  = *reinterpret_cast<float*>(&tmp_fx);


    //std::cout << std::hexfloat;
    std::cout << "f2: " << f2 << std::endl;
    std::cout << "f3: " << f3 << std::endl;
    std::cout << "f6: " << f6 << std::endl;
    std::cout << "f7: " << f7 << std::endl; 
    std::cout << "f11: " << f11 << std::endl;
    std::cout << "f13: " << f13 << std::endl;
    std::cout << "fx: " << fx << std::endl;


    // fma.rn.f32 %f4, %f1, %f3, %f2;
    float f4 = f1 * f3 + f2;
    std::cout << "f4: " << f4 << std::endl;

    // cvt.sat.f32.f32 %f5, %f4;
    // Saturate the result (clamp it between 0 and 1)
    float f5 = std::fmax(0.0f, std::fmin(1.0f, f4));
    std::cout << "f5: " << f5 << std::endl;

    // fma.rm.f32 %f8, %f5, %f7, %f6;

    std::fesetround(FE_DOWNWARD);  // 默认模式也是 FE_TONEAREST

    float f8 = f5 * f7 + f6;
    std::cout << "f8: " << f8 << std::endl;

    std::fesetround(FE_TONEAREST);  // 默认模式也是 FE_TONEAREST

    // add.f32 %f9, %f8, 0fCB40007F;
    float f9 = f8 + fx; // 0fCB40007F in hex

    // neg.f32 %f10, %f9;
    float f10 = -f9;
    std::cout << "f10: " << f10 << std::endl;

    // fma.rn.f32 %f12, %f1, %f11, %f10;
    float f12 = f1 * f11 + f10;

    // fma.rn.f32 %f14, %f1, %f13, %f12;
    float f14 = f1 * f13 + f12;
    std::cout << "f14: " << f14 << std::endl;

    // Now we deal with bit-level operations and exponentiation

    // mov.b32 %r6, %f8;
    // Move f8 to an integer for manipulation (reinterpreting the bits)
    int r6 = *reinterpret_cast<int*>(&f8);

    // shl.b32 %r7, %r6, 23;
    // Shift the bits left by 23 (equivalent to moving the exponent part)
    int r7 = r6 << 23;

    // mov.b32 %f15, %r7;
    // Reinterpret the shifted integer back to a float
    float f15 = *reinterpret_cast<float*>(&r7);

    // ex2.approx.ftz.f32 %f16, %f14;
    // Approximate 2^f14 using exp2
    float f16 = exp2f(f14);
    std::cout << "f16: " << f16 << std::endl;

    // mul.f32 %f17, %f16, %f15;
    float f17 = f16 * f15;

    // Output the final result
    std::cout << "Result: " << f17 << std::endl;
    
    std::cout << "Result: " << expf(f1) << std::endl;

    return 0;
}
