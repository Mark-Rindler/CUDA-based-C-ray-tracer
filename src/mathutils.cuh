#ifndef MATH_UTILS_CUH
#define MATH_UTILS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

// Constants
#define PI 3.14159265358979323846f
#define TWO_PI 6.28318530717958647692f
#define PI_OVER_TWO 1.57079632679489661923f
#define PI_OVER_FOUR 0.78539816339744830961f
#define INV_PI 0.31830988618379067154f
#define INV_TWO_PI 0.15915494309189533577f
#define EPSILON 1e-6f

// Utility functions
__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * PI / 180.0f;
}

__host__ __device__ inline float radians_to_degrees(float radians) {
    return radians * 180.0f / PI;
}

__host__ __device__ inline float clamp(float x, float min, float max) {
    return fminf(fmaxf(x, min), max);
}

__host__ __device__ inline int clamp(int x, int min, int max) {
    return x < min ? min : (x > max ? max : x);
}

__host__ __device__ inline float lerp(float a, float b, float t) {
    return (1.0f - t) * a + t * b;
}

__host__ __device__ inline float smoothstep(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

__device__ inline float random_float(curandState* state) {
    return curand_uniform(state);
}

__device__ inline float random_float(curandState* state, float min, float max) {
    return min + (max - min) * curand_uniform(state);
}

__device__ inline int random_int(curandState* state, int min, int max) {
    return min + static_cast<int>((max - min + 1) * curand_uniform(state));
}

__device__ inline void init_rand_state(curandState* state, int tid, int seed = 1984) {
    curand_init(seed, tid, 0, state);
}

__host__ __device__ inline float fast_pow(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __powf(a, b);                         
#else
    return static_cast<float>(std::pow(a, b));   
#endif
}

__host__ __device__ inline float fast_exp(float x) {
#if defined(__CUDA_ARCH__)
    return __expf(x);
#else
    return static_cast<float>(std::exp(x));
#endif
}

__host__ __device__ inline float fast_log(float x) {
#if defined(__CUDA_ARCH__)
    return __logf(x);
#else
    return static_cast<float>(std::log(x));
#endif
}

__host__ __device__ inline float distance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

__host__ __device__ inline float distance_squared(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return dx * dx + dy * dy;
}

__host__ __device__ inline float linear_to_gamma(float linear, float gamma = 2.2f) {
    return powf(linear, 1.0f / gamma);
}

__host__ __device__ inline float gamma_to_linear(float gamma_val, float gamma = 2.2f) {
    return powf(gamma_val, gamma);
}

__host__ __device__ inline unsigned int pack_float4(float x, float y, float z, float w) {
    x = clamp(x, 0.0f, 1.0f);
    y = clamp(y, 0.0f, 1.0f);
    z = clamp(z, 0.0f, 1.0f);
    w = clamp(w, 0.0f, 1.0f);
    
    unsigned int r = (unsigned int)(x * 255.0f);
    unsigned int g = (unsigned int)(y * 255.0f);
    unsigned int b = (unsigned int)(z * 255.0f);
    unsigned int a = (unsigned int)(w * 255.0f);
    
    return (a << 24) | (b << 16) | (g << 8) | r;
}

__host__ __device__ inline void unpack_float4(unsigned int packed, float& x, float& y, float& z, float& w) {
    x = (packed & 0xFF) / 255.0f;
    y = ((packed >> 8) & 0xFF) / 255.0f;
    z = ((packed >> 16) & 0xFF) / 255.0f;
    w = ((packed >> 24) & 0xFF) / 255.0f;
}

__host__ __device__ inline float fract(float x) {
    return x - floorf(x);
}

__host__ __device__ inline float mod(float x, float y) {
    return x - y * floorf(x / y);
}

__host__ __device__ inline float sign(float x) {
    return (x > 0.0f) - (x < 0.0f);
}

__host__ __device__ inline float safe_divide(float a, float b, float default_val = 0.0f) {
    return (fabsf(b) > EPSILON) ? (a / b) : default_val;
}

__host__ __device__ inline bool is_finite(float x) {
    return isfinite(x);
}

__host__ __device__ inline bool solve_quadratic(float a, float b, float c, float& x0, float& x1) {
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return false;
    
    float sqrt_disc = sqrtf(discriminant);
    x0 = (-b - sqrt_disc) / (2 * a);
    x1 = (-b + sqrt_disc) / (2 * a);
    
    if (x0 > x1) {
        float temp = x0;
        x0 = x1;
        x1 = temp;
    }
    
    return true;
}

#endif 