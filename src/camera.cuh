#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "ray.cuh"
#include "vec3.cuh"
#include <cuda_runtime.h>

// old random unit in disk
/*__host__ __device__ inline vec3 random_in_unit_disk(uint32_t x, uint32_t y,
                                                    uint32_t frame) {
    // Combine pixel coords with frame number for temporal variation
    uint32_t seed = (x * 1973u) ^ (y * 9277u) ^ (frame * 26699u) ^ 0x9e3779b9u;
    seed ^= seed >> 17;
    seed *= 0xed5ad4bbu;
    seed ^= seed >> 11;
    seed *= 0xac4c1b51u;
    seed ^= seed >> 15;
    seed *= 0x31848babu;
    seed ^= seed >> 14;

    float r1 = ((seed & 0xFFFFu) + 0.5f) / 65536.0f;
    float r2 = (((seed * 0x343fdu + 0xc0f5u) & 0xFFFFu) + 0.5f) /
               65536.0f; // Different hash for r2

    float r = sqrtf(r1), phi = 6.2831853f * r2;
    return vec3(r * cosf(phi), r * sinf(phi), 0.0f);
} */

// First, add the blue noise texture to constant memory
__constant__ float d_blue_noise[64][64][2]; // 64x64 blue noise with 2 channels

inline void initBlueNoise() {
    // Generate or load blue noise data
    float blueNoise[64][64][2];

    // Simple stratified generation
    for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 64; ++x) {
            // Van der Corput sequence for better distribution
            float u = 0.0f, v = 0.0f;
            float p = 0.5f;
            int n = y * 64 + x + 1;
            int m = n;
            while (n > 0) {
                if (n & 1)
                    u += p;
                p *= 0.5f;
                n >>= 1;
            }
            // Sobol for second dimension
            p = 0.5f;
            while (m > 0) {
                if (m & 1)
                    v += p;
                p *= 0.5f;
                m >>= 1;
            }
            blueNoise[y][x][0] = u;
            blueNoise[y][x][1] = v;
        }
    }

    cudaMemcpyToSymbol(d_blue_noise, blueNoise, sizeof(blueNoise));
}

class Camera {
  private:
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w; // Camera basis vectors
    float lens_radius;

    // Blue noise sampling for DOF
    __device__ inline vec3 random_in_unit_disk_blue(int pixel_x, int pixel_y,
                                                    int sample_index) const {
        // Wrap coordinates with bit masking (fast for power of 2)
        int tx = (pixel_x + sample_index * 17) & 63;
        int ty = (pixel_y + sample_index * 29) & 63;

        // Get blue noise values
        float r1 = d_blue_noise[ty][tx][0];
        float r2 = d_blue_noise[ty][tx][1];

        // Apply temporal jitter using golden ratio
        float phi_offset = sample_index * 0.618033988749895f;
        r1 = fmodf(r1 + phi_offset, 1.0f);
        r2 = fmodf(r2 + phi_offset * 0.381966011250105f, 1.0f);

        // Convert to disk coordinates
        float r = sqrtf(r1);
        float theta = 2.0f * 3.14159265f * r2;

        return vec3(r * cosf(theta), r * sinf(theta), 0.0f);
    }

    // Fallback hash-based version for host code
    __host__ inline vec3 random_in_unit_disk_hash(uint32_t x,
                                                  uint32_t y) const {
        uint32_t seed = (x * 1973u) ^ (y * 9277u) ^ 0x9e3779b9u;
        seed ^= seed >> 17;
        seed *= 0xed5ad4bbu;
        seed ^= seed >> 11;
        seed *= 0xac4c1b51u;
        seed ^= seed >> 15;
        seed *= 0x31848babu;
        seed ^= seed >> 14;

        float r1 = ((seed & 0xFFFFu) + 0.5f) / 65536.0f;
        float r2 = (((seed * 0x343fdu + 0xc0f5u) & 0xFFFFu) + 0.5f) / 65536.0f;

        float r = sqrtf(r1);
        float phi = 6.2831853f * r2;
        return vec3(r * cosf(phi), r * sinf(phi), 0.0f);
    }

  public:
    // Constructors remain the same
    __host__ __device__ Camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov,
                               float aspect_ratio, float aperture = 0.0f,
                               float focus_dist = 1.0f) {
        float theta = vfov * 3.14159265358979323846f / 180.0f;
        float h = tanf(theta / 2.0f);
        float viewport_height = 2.0f * h;
        float viewport_width = aspect_ratio * viewport_height;

        w = (lookfrom - lookat).normalized();
        u = cross(vup, w).normalized();
        v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner =
            origin - horizontal * 0.5f - vertical * 0.5f - focus_dist * w;

        lens_radius = aperture / 2.0f;
    }

    __host__ __device__ Camera(float aspect_ratio, float viewport_height = 2.0f,
                               float focal_length = 1.0f) {
        origin = vec3(0.0f, 0.0f, 0.0f);

        float viewport_width = viewport_height * aspect_ratio;

        horizontal = vec3(viewport_width, 0.0f, 0.0f);
        vertical = vec3(0.0f, -viewport_height, 0.0f);
        lower_left_corner = origin - horizontal * 0.5f - vertical * 0.5f -
                            vec3(0.0f, 0.0f, focal_length);

        lens_radius = 0.0f;

        u = vec3(1, 0, 0);
        v = vec3(0, 1, 0);
        w = vec3(0, 0, 1);
    }

    __device__ Ray get_ray(float s, float t, int pixel_x, int pixel_y,
                           int sample_index = 0) const {
        if (lens_radius <= 0)
            return get_ray_simple(s, t);

        vec3 rd = lens_radius *
                  random_in_unit_disk_blue(pixel_x, pixel_y, sample_index);
        vec3 offset = u * rd.x + v * rd.y;
        vec3 ray_dir =
            lower_left_corner + s * horizontal + t * vertical - origin - offset;

        return Ray(origin + offset, ray_dir.normalized());
    }

    // Keep backward compatibility - old signature for host code
    __host__ __device__ Ray get_ray(float s, float t) const {
        if (lens_radius <= 0)
            return get_ray_simple(s, t);

#ifdef __CUDA_ARCH__
        // On device, still use pixel coordinates but with default values
        // This shouldn't be called on device - the new signature should be used
        return get_ray_simple(s, t);
#else
        // On host, use the hash-based version
        vec3 rd = lens_radius * random_in_unit_disk_hash((uint32_t)(s * 1e6f),
                                                         (uint32_t)(t * 1e6f));
        vec3 offset = u * rd.x + v * rd.y;
        vec3 ray_dir =
            lower_left_corner + s * horizontal + t * vertical - origin - offset;
        return Ray(origin + offset, ray_dir.normalized());
#endif
    }

    __host__ __device__ Ray get_ray_simple(float s, float t) const {
        vec3 ray_direction =
            lower_left_corner + s * horizontal + t * vertical - origin;
        return Ray(origin, ray_direction.normalized());
    }

    __host__ __device__ vec3 get_origin() const { return origin; }
    __host__ __device__ vec3 get_lower_left_corner() const {
        return lower_left_corner;
    }
    __host__ __device__ vec3 get_horizontal() const { return horizontal; }
    __host__ __device__ vec3 get_vertical() const { return vertical; }

    __host__ __device__ void set_position(const vec3 &pos) {
        vec3 delta = pos - origin;
        origin = pos;
        lower_left_corner += delta;
    }

    __host__ __device__ void look_at(const vec3 &target,
                                     const vec3 &vup = vec3(0, 1, 0)) {
        w = (origin - target).normalized();
        u = cross(vup, w).normalized();
        v = cross(w, u);

        float viewport_height = vertical.length();
        float viewport_width = horizontal.length();
        float focus_dist = (origin - target).length();

        horizontal = viewport_width * u;
        vertical = viewport_height * v;
        lower_left_corner =
            origin - horizontal * 0.5f - vertical * 0.5f - focus_dist * w;
    }
};

#endif