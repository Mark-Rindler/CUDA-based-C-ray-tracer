#ifndef RAY_CUH
#define RAY_CUH

#include "vec3.cuh"

class Ray {
  public:
    point3 orig;
    vec3 dir;

    // Constructors
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const point3 &origin, const vec3 &direction)
        : orig(origin), dir(direction) {}

    // Accessors
    __host__ __device__ const point3 &origin() const { return orig; }
    __host__ __device__ const vec3 &direction() const { return dir; }

    // Ray equation: P(t) = origin + t * direction
    __host__ __device__ point3 at(float t) const { return orig + t * dir; }

    // Utility functions
    __host__ __device__ Ray transformed(const vec3 &offset) const {
        return Ray(orig + offset, dir);
    }

    __host__ __device__ void normalize_direction() { dir.normalize(); }
};

// Utility functions for rays
__host__ __device__ inline float distance_to_point(const Ray &r,
                                                   const point3 &p) {
    vec3 v = p - r.orig;
    float t = dot(v, r.dir) / r.dir.length_squared();
    point3 closest = r.at(t);
    return (p - closest).length();
}

// Check if ray intersects sphere
__host__ __device__ inline bool ray_sphere_intersect(const Ray &r,
                                                     const point3 &center,
                                                     float radius, float &t_min,
                                                     float &t_max) {
    vec3 oc = r.orig - center;
    float a = r.dir.length_squared();
    float half_b = dot(oc, r.dir);
    float c = oc.length_squared() - radius * radius;

    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0)
        return false;

    float sqrtd = sqrtf(discriminant);
    t_min = (-half_b - sqrtd) / a;
    t_max = (-half_b + sqrtd) / a;

    return true;
}

// Stream output (host only)
inline std::ostream &operator<<(std::ostream &os, const Ray &r) {
    return os << "Ray(origin: " << r.orig << ", direction: " << r.dir << ")";
}

#endif