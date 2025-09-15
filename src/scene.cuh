// scene.cuh ──────────────────────────────────────────────────────────────
#ifndef SCENE_CUH
#define SCENE_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "camera.cuh"
#include "mathutils.cuh"
#include "mesh.cuh"
#include "ray.cuh"
#include "triangle.cuh"
#include "vec3.cuh"

// ── Material properties ─────────────────────────────────────────────────────
struct Material {
    // Base properties
    vec3 albedo;     // Base color / diffuse color
    vec3 specular;   // Specular color (F0 for dielectrics)
    float metallic;  // Metalness (0 = dielectric, 1 = metal)
    float roughness; // Surface roughness (0 = smooth, 1 = rough)
    vec3 emission;   // Emissive color

    // Advanced properties
    float ior; // Index of refraction (1.5 for glass, 1.33 for water)
    float
        transmission; // Transmission/transparency (0 = opaque, 1 = transparent)
    float transmissionRoughness; // Roughness of transmitted rays

    // Clearcoat layer (for car paint, lacquered wood, etc.)
    float clearcoat;          // Clearcoat amount (0-1)
    float clearcoatRoughness; // Clearcoat roughness

    // Subsurface scattering (simplified)
    vec3 subsurfaceColor;   // SSS color tint
    float subsurfaceRadius; // SSS radius/strength

    // Artistic controls
    float anisotropy; // Anisotropic highlight (-1 to 1)
    float sheen;      // Sheen amount for fabrics
    vec3 sheenTint;   // Sheen color tint

    // Special effects
    float iridescence;          // Iridescence strength (soap bubbles, oil)
    float iridescenceThickness; // Thin-film thickness

    __host__ __device__ Material()
        : albedo(vec3(0.8f)), specular(vec3(0.04f)), metallic(0.0f),
          roughness(0.5f), emission(vec3(0.0f)), ior(1.5f), transmission(0.0f),
          transmissionRoughness(0.0f), clearcoat(0.0f),
          clearcoatRoughness(0.03f), subsurfaceColor(vec3(1.0f)),
          subsurfaceRadius(0.0f), anisotropy(0.0f), sheen(0.0f),
          sheenTint(vec3(0.5f)), iridescence(0.0f),
          iridescenceThickness(550.0f) {}

    __host__ __device__ Material(const vec3 &alb, float rough = 0.5f,
                                 float met = 0.0f)
        : Material() {
        albedo = alb;
        roughness = rough;
        metallic = met;
        // Set F0 based on metallic
        specular = lerp(vec3(0.04f), albedo, metallic);
    }
};

// ── Helper device functions ─────────────────────────────────────────────────
__device__ inline float attenuate(float distance, float range) {
    float att = range / (range + distance);
    return att * att;
}
// Schlick's Fresnel approximation
__device__ inline vec3 fresnelSchlick(float cosTheta, const vec3 &F0) {
    float f = powf(1.0f - cosTheta, 5.0f);
    return F0 + (vec3(1.0f) - F0) * f;
}

// Fresnel with roughness (for IBL, but useful for direct lighting too)
__device__ inline vec3 fresnelSchlickRoughness(float cosTheta, const vec3 &F0,
                                               float roughness) {
    float f = powf(fmaxf(1.0f - cosTheta, 0.0f), 5.0f);
    vec3 maxRefl =
        vec3(fmaxf(1.0f - roughness, F0.x), fmaxf(1.0f - roughness, F0.y),
             fmaxf(1.0f - roughness, F0.z));
    return F0 + (maxRefl - F0) * f;
}

// GGX/Trowbridge-Reitz normal distribution
__device__ inline float distributionGGX(const vec3 &N, const vec3 &H,
                                        float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = PI * denom * denom;

    return num / fmaxf(denom, 0.001f);
}

// Geometry function (Smith's method with GGX)
__device__ inline float geometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0f);
    float k = (r * r) / 8.0f;

    float num = NdotV;
    float denom = NdotV * (1.0f - k) + k;

    return num / fmaxf(denom, 0.001f);
}

__device__ inline float geometrySmith(const vec3 &N, const vec3 &V,
                                      const vec3 &L, float roughness) {
    float NdotV = fmaxf(dot(N, V), 0.0f);
    float NdotL = fmaxf(dot(N, L), 0.0f);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// Iridescence (thin-film interference)
__device__ inline vec3 calculateIridescence(float thickness, float cosTheta) {
    // Simplified thin-film interference
    float phase = 2.0f * thickness * cosTheta;
    vec3 color;
    color.x = 0.5f + 0.5f * cosf(phase * 0.005f);
    color.y = 0.5f + 0.5f * cosf(phase * 0.005f + TWO_PI / 3.0f);
    color.z = 0.5f + 0.5f * cosf(phase * 0.005f + 2.0f * TWO_PI / 3.0f);
    return color;
}

__host__ __device__ inline float clamp01(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

__host__ __device__ inline vec3 reflectVec(const vec3 &I, const vec3 &N) {
    return I - 2.0f * dot(I, N) * N;
}

__host__ __device__ inline bool refractVec(const vec3 &I, const vec3 &N,
                                           float eta, vec3 &T) {
    // I: incident dir (from hitpoint into scene), N: outward normal
    float NdotI = dot(N, I);
    float k = 1.0f - eta * eta * (1.0f - NdotI * NdotI);
    if (k < 0.0f)
        return false;
    T = eta * I - (eta * NdotI + sqrtf(k)) * N;
    return true;
}

__host__ __device__ inline vec3 faceForward(const vec3 &N, const vec3 &I) {
    // Ensure N faces against incident direction I
    return (dot(N, I) < 0.0f) ? N : (-N);
}

__device__ inline vec3 beerLambert(const vec3 &transRGBPerUnit, float dist) {
    // Treat 'transRGBPerUnit' in [0..1] as per-unit transmittance
    vec3 t = clamp(transRGBPerUnit, 0.0f, 1.0f);
    return vec3(powf(t.x, dist), powf(t.y, dist), powf(t.z, dist));
}

__device__ inline vec3 sampleSky(const Ray &r, const vec3 &top,
                                 const vec3 &bottom, bool useSky) {
    if (!useSky)
        return vec3(0.0f);
    float t = 0.5f * (r.direction().y + 1.0f);
    return lerp(bottom, top, t);
}

// Map Blinn-Phong shininess (n) to GGX roughness in [0..1].
// Good approximation: alpha ≈ sqrt(2/(n+2)); perceptual roughness = alpha.
__host__ __device__ inline float phongShininessToRoughness(float n) {
    float alpha = sqrtf(2.0f / (fmaxf(n, 1.0f) + 2.0f));
    // Keep a small floor to avoid singularities
    return clamp01(fmaxf(alpha, 0.02f));
}

// For dielectrics, derive F0 from IOR
__host__ __device__ inline float iorToF0(float ior) {
    float a = (ior - 1.0f) / (ior + 1.0f);
    return a * a;
}

// ── Light types ─────────────────────────────────────────────────────────────
enum LightType { LIGHT_POINT = 0, LIGHT_DIRECTIONAL = 1, LIGHT_SPOT = 2 };

struct Light {
    LightType type;
    vec3 position;   // For point and spot lights
    vec3 direction;  // For directional and spot lights
    vec3 color;      // Light color and intensity
    float intensity; // Light intensity multiplier
    float range;     // For point lights (attenuation)
    float innerCone; // For spot lights (inner cone angle in radians)
    float outerCone; // For spot lights (outer cone angle in radians)

    __host__ __device__ Light()
        : type(LIGHT_POINT), position(vec3(0, 10, 0)),
          direction(vec3(0, -1, 0)), color(vec3(1.0f)), intensity(1.0f),
          range(100.0f), innerCone(0.5f), outerCone(0.7f) {}
};

struct DeviceMesh {
    vec3 *verts;
    Tri *faces;
    int faceCount;
    Material material;
    DeviceBVHNode *bvhNodes;
    int nodeCount;
    int *primIndices;
};

// ── Hit information structure ───────────────────────────────────────────────
struct HitInfo {
    bool hit;
    float t;
    vec3 point;
    vec3 normal;
    Material material;

    __device__ HitInfo() : hit(false), t(1e30f) {}
};

// ── Forward declaration of render kernel ────────────────────────────────────
__global__ void render_kernel(unsigned char *out, int W, int H, Camera cam,
                              DeviceMesh *meshes, int nMeshes, Light *lights,
                              int nLights, vec3 ambientLight, vec3 skyColorTop,
                              vec3 skyColorBottom, bool useSky);

__device__ inline bool bvh_any_hit(const Ray &ray, const DeviceMesh &M,
                                   float tMax) {
    if (!M.bvhNodes || M.nodeCount == 0)
        return false;
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;
    while (sp) {
        const int ni = stack[--sp];
        const DeviceBVHNode &N = M.bvhNodes[ni];
        if (!N.bbox.hit(ray, tMax))
            continue;
        if (N.count > 0) {
            // leaf
            for (int i = 0; i < N.count; ++i) {
                int fidx = M.primIndices[N.start + i];
                const Tri idx = M.faces[fidx];
                const vec3 v0 = M.verts[idx.v0];
                const vec3 v1 = M.verts[idx.v1];
                const vec3 v2 = M.verts[idx.v2];
                Triangle tri(v0, v1, v2);
                float t, u, v;
                if (tri.intersect(ray, t, u, v)) {
                    if (t > 0.001f && t < tMax)
                        return true;
                }
            }
        } else {
            // internal
            if (N.left >= 0)
                stack[sp++] = N.left;
            if (N.right >= 0)
                stack[sp++] = N.right;
        }
    }
    return false;
}

__device__ inline HitInfo bvh_trace(const Ray &ray, const DeviceMesh &M) {
    HitInfo out;
    out.hit = false;
    out.t = 1e30f;

    if (!M.bvhNodes || M.nodeCount == 0 /* ⭐ */ || !M.primIndices) {
        return out; // nothing to traverse
    }

    constexpr int MaxStack = 256;
    int stack[MaxStack];
    int sp = 0;
    int ni = 0; // root

    while (true) {
        const DeviceBVHNode &N = M.bvhNodes[ni];

        // miss -> pop
        if (!N.bbox.hit(ray, out.t)) {
            if (sp == 0)
                break;
            ni = stack[--sp];
            continue;
        }

        if (N.count > 0) {
            for (int i = 0; i < N.count; ++i) {
                const int fidx = M.primIndices[N.start + i];
                const Tri triIdx = M.faces[fidx];
                const vec3 v0 = M.verts[triIdx.v0];
                const vec3 v1 = M.verts[triIdx.v1];
                const vec3 v2 = M.verts[triIdx.v2];

                float tHit, uHit, vHit;
                Triangle tri(v0, v1, v2);
                if (tri.intersect(ray, tHit, uHit, vHit) && tHit > 1e-3f &&
                    tHit < out.t) {
                    out.hit = true;
                    out.t = tHit;
                    out.point = ray.at(tHit);
                    // geometric normal
                    vec3 e1 = v1 - v0, e2 = v2 - v0;
                    out.normal = normalize(cross(e1, e2));
                    out.material = M.material;
                }
            }

            if (sp == 0)
                break;
            ni = stack[--sp];
            continue;
        }

        // internal: order children by entry t and prune by current best t
        const int L = N.left, R = N.right;
        float tL = 0.f, tR = 0.f;
        bool hL = (L >= 0) && M.bvhNodes[L].bbox.hit_t(ray, out.t, tL);
        bool hR = (R >= 0) && M.bvhNodes[R].bbox.hit_t(ray, out.t, tR);

        // ⭐ explicit prune by best-so-far
        hL = hL && (tL < out.t);
        hR = hR && (tR < out.t);

        if (!hL && !hR) {
            if (sp == 0)
                break;
            ni = stack[--sp];
            continue;
        }

        int nearIdx, farIdx;
        float tNear, tFar;
        if (hL && (!hR || tL <= tR)) {
            nearIdx = L;
            farIdx = R;
            tNear = tL;
            tFar = tR;
        } else {
            nearIdx = R;
            farIdx = L;
            tNear = tR;
            tFar = tL;
        }

        // only push far if it actually hits & is worth visiting
        if (farIdx >= 0 && (farIdx == R ? hR : hL) && tFar < out.t &&
            sp < MaxStack) {
            stack[sp++] = farIdx;
        }

        // only descend if near actually hits & is worth visiting
        if (nearIdx >= 0 && (nearIdx == L ? hL : hR) && tNear < out.t) {
            ni = nearIdx;
        } else {
            if (sp == 0)
                break;
            ni = stack[--sp];
        }
    }
    return out;
}

// FORWARD DECL
__device__ inline vec3 shadeOneBounce(const Ray &r, DeviceMesh *meshes,
                                      int nMeshes, Light *lights, int nLights,
                                      const vec3 &ambientLight,
                                      const vec3 &skyTop, const vec3 &skyBottom,
                                      bool useSky);

__device__ inline HitInfo traceRay(const Ray &ray, DeviceMesh *meshes,
                                   int nMeshes) {
    HitInfo best;
    best.hit = false;
    best.t = 1e30f;
    for (int m = 0; m < nMeshes; ++m) {
        HitInfo h = bvh_trace(ray, meshes[m]);
        if (h.hit && h.t < best.t)
            best = h;
    }
    return best;
}

// core that can disable secondary rays to avoid infinite recursion --
__device__ inline vec3 calculatePBRLightingCore(
    const HitInfo &hit, const Ray &ray, DeviceMesh *meshes, int nMeshes,
    Light *lights, int nLights, const vec3 &ambientLight, const vec3 &skyTop,
    const vec3 &skyBottom, bool useSky, bool allowSpecTransmission) {
    vec3 color = vec3(0.0f);
    vec3 V = -ray.direction();   // view from hit toward camera
    const vec3 &Ng = hit.normal; // geometric normal
    const Material &mat = hit.material;

    // Clamp + classify
    const float rough = fminf(fmaxf(mat.roughness, 0.02f), 1.0f);
    const float metal = fminf(fmaxf(mat.metallic, 0.0f), 1.0f);
    const bool isGlass = (mat.transmission > 0.0f) && (metal < 0.1f);

    // Base F0
    vec3 F0 = mat.specular;
    F0 = lerp(F0, mat.albedo, metal);

    // Emission
    color = color + mat.emission;

    // Ambient
    float NdotV = fmaxf(dot(Ng, V), 0.0f);
    vec3 F_ambient = fresnelSchlickRoughness(NdotV, F0, rough);
    vec3 kS_ambient = F_ambient;
    vec3 kD_ambient = (vec3(1.0f) - kS_ambient) * (1.0f - metal);
    if (isGlass)
        kD_ambient = vec3(0.0f);
    color = color + kD_ambient * mat.albedo * ambientLight;

    // Direct lights
    for (int i = 0; i < nLights; ++i) {
        const Light &light = lights[i];
        vec3 L;
        float attenuation = 1.0f;

        if (light.type == LIGHT_DIRECTIONAL) {
            L = -light.direction;
        } else {
            vec3 toLight = light.position - hit.point;
            float distance = toLight.length();
            L = toLight / fmaxf(distance, 1e-6f);
            float att = attenuate(distance, light.range);
            if (light.type == LIGHT_SPOT) {
                float theta = dot(L, -light.direction);
                float epsilon = light.innerCone - light.outerCone;
                float spotIntensity =
                    clamp((theta - light.outerCone) / epsilon, 0.0f, 1.0f);
                att *= spotIntensity;
            }
            attenuation = att;
        }

        // Shadows: skip thin-glass blockers
        bool inShadow = false;
        const float eps = 1e-3f * fmaxf(1.0f, hit.t);
        Ray shadowRay(hit.point + Ng * eps, L);
        float lightDistance = (light.type == LIGHT_DIRECTIONAL)
                                  ? 1e30f
                                  : (light.position - hit.point).length();
        for (int m = 0; m < nMeshes && !inShadow; ++m) {
            if (meshes[m].material.transmission > 0.0f)
                continue; // thin glass doesn't occlude
            if (bvh_any_hit(shadowRay, meshes[m], lightDistance))
                inShadow = true;
        }
        if (inShadow)
            continue;

        // Microfacet
        vec3 H = (L + V).normalized();
        float NdotL = fmaxf(dot(Ng, L), 0.0f);
        float VdotH = fmaxf(dot(V, H), 0.0f);

        float D = distributionGGX(Ng, H, rough);
        float G = geometrySmith(Ng, V, L, rough);
        vec3 F = fresnelSchlick(VdotH, F0);

        if (mat.iridescence > 0.0f) {
            vec3 iridColor =
                calculateIridescence(mat.iridescenceThickness, VdotH);
            F = lerp(F, F * iridColor, mat.iridescence);
        }

        vec3 specular =
            (D * G * F) / (4.0f * fmaxf(dot(Ng, V), 0.0f) * NdotL + 0.001f);

        vec3 kS = F;
        vec3 kD = (vec3(1.0f) - kS) * (1.0f - metal);
        vec3 diffuse = mat.albedo / PI;

        if (mat.sheen > 0.0f) {
            float FH = powf(1.0f - VdotH, 5.0f);
            vec3 sheenColor = lerp(vec3(1.0f), mat.sheenTint, FH);
            kD = kD + sheenColor * mat.sheen * (1.0f - metal);
        }

        if (mat.subsurfaceRadius > 0.0f) {
            float sss =
                powf(fmaxf(dot(V, -L), 0.0f), 2.0f) * mat.subsurfaceRadius;
            diffuse = lerp(diffuse, mat.subsurfaceColor / PI, sss);
        }

        // Your old "thinTrans" lobe — only keep for glass if we are NOT doing
        // secondary rays
        vec3 thinTrans = vec3(0.0f);
        if (isGlass && !allowSpecTransmission) {
            kD = vec3(0.0f);
            thinTrans = (vec3(1.0f) - F) * mat.transmission;
        }

        vec3 Lo = (kD * diffuse + specular + thinTrans) * light.color *
                  light.intensity * 20.0f * NdotL * attenuation;

        // Clearcoat
        if (mat.clearcoat > 0.0f) {
            float ccD = distributionGGX(Ng, H, mat.clearcoatRoughness);
            float ccG = geometrySmith(Ng, V, L, mat.clearcoatRoughness);
            vec3 ccF = fresnelSchlick(VdotH, vec3(0.04f));
            vec3 ccBRDF = (ccD * ccG * ccF) /
                          (4.0f * fmaxf(dot(Ng, V), 0.0f) * NdotL + 0.001f);
            Lo = Lo * (vec3(1.0f) - mat.clearcoat * ccF) +
                 ccBRDF * light.color * light.intensity * 20.0f * NdotL *
                     attenuation * mat.clearcoat;
        }

        color = color + Lo;
    }

    // single-bounce reflect/refract for glass
    if (isGlass && allowSpecTransmission) {
        // Relative IOR handling (entering or exiting)
        vec3 I = ray.direction();
        vec3 Nf = faceForward(Ng, I); // outward normal w.r.t. I
        float n1 = 1.0f, n2 = mat.ior;
        if (dot(Ng, I) > 0.0f) { // exiting
            float tmp = n1;
            n1 = n2;
            n2 = tmp;
            Nf = faceForward(Ng, I); // recompute
        }
        float eta = n1 / n2;

        // Fresnel with relative IOR
        float F0s = (n2 - n1) / (n2 + n1);
        F0s = F0s * F0s;
        float cosTheta = fmaxf(dot(-I, Nf), 0.0f);
        vec3 F = fresnelSchlick(cosTheta, vec3(F0s));

        // Offsets
        const float eps = 1e-3f * fmaxf(1.0f, hit.t);

        // Reflection
        vec3 Rdir = reflectVec(I, Nf).normalized();
        vec3 Rcol = shadeOneBounce(Ray(hit.point + Nf * eps, Rdir), meshes,
                                   nMeshes, lights, nLights, ambientLight,
                                   skyTop, skyBottom, useSky);

        // Refraction
        vec3 Tdir;
        vec3 Tcol = vec3(0.0f);
        bool refrOk = refractVec(I, Nf, eta, Tdir);
        if (refrOk) {
            Tdir = Tdir.normalized();
            // Trace refracted ray once
            HitInfo h2 =
                traceRay(Ray(hit.point - Nf * eps, Tdir), meshes, nMeshes);
            float thickness = 1.0f;
            if (h2.hit)
                thickness = h2.t;

            vec3 behind = h2.hit ? calculatePBRLightingCore(
                                       h2, Ray(hit.point - Nf * eps, Tdir),
                                       meshes, nMeshes, lights, nLights,
                                       ambientLight, skyTop, skyBottom, useSky,
                                       /*allowSpecTransmission=*/false)
                                 : sampleSky(Ray(hit.point - Nf * eps, Tdir),
                                             skyTop, skyBottom, useSky);

            // Beer–Lambert using albedo as per-unit transmittance tint
            vec3 absorb = beerLambert(clamp(mat.albedo, 0.0f, 1.0f), thickness);
            Tcol = absorb * behind;
        } else {
            // Total internal reflection
            F = vec3(1.0f);
        }

        color = color + F * Rcol + (vec3(1.0f) - F) * (mat.transmission) * Tcol;
    }

    return color;
}

__device__ inline vec3
calculatePBRLighting(const HitInfo &hit, const Ray &ray, DeviceMesh *meshes,
                     int nMeshes, Light *lights, int nLights,
                     const vec3 &ambientLight, const vec3 &skyTop,
                     const vec3 &skyBottom, bool useSky) {
    return calculatePBRLightingCore(hit, ray, meshes, nMeshes, lights, nLights,
                                    ambientLight, skyTop, skyBottom, useSky,
                                    /*allowSpecTransmission=*/true);
}

__device__ inline vec3 shadeOneBounce(const Ray &r, DeviceMesh *meshes,
                                      int nMeshes, Light *lights, int nLights,
                                      const vec3 &ambientLight,
                                      const vec3 &skyTop, const vec3 &skyBottom,
                                      bool useSky) {
    HitInfo h = traceRay(r, meshes, nMeshes);
    if (!h.hit)
        return sampleSky(r, skyTop, skyBottom, useSky);
    // No further bounces from here (prevents recursion explosion)
    return calculatePBRLightingCore(h, r, meshes, nMeshes, lights, nLights,
                                    ambientLight, skyTop, skyBottom, useSky,
                                    /*allowSpecTransmission=*/false);
}

// ── Scene class ─────────────────────────────────────────────────────────────
class Scene {
  private:
    // Image settings
    int width;
    int height;

    int bvhLeafTarget_ = 12;
    int bvhLeafTol_ = 5;

    // Scene components
    std::vector<std::unique_ptr<Mesh>> meshes;
    std::vector<Material> mesh_materials;
    std::vector<Light> lights;
    Camera camera;

    // GPU resources
    DeviceMesh *d_mesh_descriptors = nullptr;
    Light *d_lights = nullptr;
    unsigned char *d_pixels = nullptr;

    // Lighting settings
    vec3 ambient_light = vec3(0.1f);

    // Background settings
    bool use_sky = true;
    vec3 sky_color_top = vec3(0.6f, 0.7f, 1.0f);
    vec3 sky_color_bottom = vec3(1.0f, 1.0f, 1.0f);

  public:
    // Constructor
    Scene(int w, int h)
        : width(w), height(h), camera(static_cast<float>(w) / h, 2.0f, 1.0f) {
        // Allocate pixel buffer on GPU
        size_t nBytes = static_cast<size_t>(width) * height * 3;
        cudaError_t err = cudaMalloc(&d_pixels, nBytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU pixel buffer");
        }
    }

    // Destructor
    ~Scene() {
        if (d_mesh_descriptors) {
            cudaFree(d_mesh_descriptors);
        }
        if (d_lights) {
            cudaFree(d_lights);
        }
        if (d_pixels) {
            cudaFree(d_pixels);
        }
    }

    // Delete copy operations
    Scene(const Scene &) = delete;
    Scene &operator=(const Scene &) = delete;

    void setBVHLeafTarget(int target, int tol = 5) {
        bvhLeafTarget_ = (target < 1 ? 1 : target);
        bvhLeafTol_ = (tol < 0 ? 0 : tol);
        // mark meshes dirty
        for (auto &m : meshes)
            m->bvhDirty = true;
    }

    // Camera setup methods
    void setCamera(const vec3 &lookfrom, const vec3 &lookat, const vec3 &vup,
                   float vfov, float aperture = 0.0f, float focus_dist = 1.0f) {
        float aspect = static_cast<float>(width) / height;
        camera =
            Camera(lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist);
    }

    void setCameraSimple(float viewport_height = 2.0f,
                         float focal_length = 1.0f) {
        float aspect = static_cast<float>(width) / height;
        camera = Camera(aspect, viewport_height, focal_length);
    }

    // --- Camera helpers for runtime control ---
    vec3 cameraOrigin() const { return camera.get_origin(); }
    vec3 cameraForward() const {
        // Direction of ray through the center pixel
        vec3 dir = camera.get_lower_left_corner() +
                   camera.get_horizontal() * 0.5f +
                   camera.get_vertical() * 0.5f - camera.get_origin();
        return dir.normalized();
    }
    void moveCamera(const vec3 &pos) { camera.set_position(pos); }
    void lookCameraAt(const vec3 &target, const vec3 &vup = vec3(0, 1, 0)) {
        camera.look_at(target, vup);
    }

    // Mesh management with materials
    Mesh *addMesh(const std::string &obj_path,
                  const Material &mat = Material()) {
        meshes.push_back(std::make_unique<Mesh>(obj_path));
        mesh_materials.push_back(mat);
        return meshes.back().get();
    }

    inline Mesh *addTriangles(const std::vector<Triangle> &tris,
                              const Material &mat = Material()) {
        // Create a new mesh and store material
        meshes.push_back(std::make_unique<Mesh>());
        mesh_materials.push_back(mat);

        Mesh *m = meshes.back().get();

        // If Mesh() builds a unit cube by default, nuke it.
        m->vertices.clear();
        m->faces.clear();

        m->vertices.reserve(tris.size() * 3);
        m->faces.reserve(tris.size());

        for (const Triangle &t : tris) {
            const int base = static_cast<int>(m->vertices.size());
            m->vertices.push_back(t.v0);
            m->vertices.push_back(t.v1);
            m->vertices.push_back(t.v2);

            // CCW face (Tri indices) assumed by your pipeline
            m->faces.push_back(Tri{base + 0, base + 1, base + 2});
        }
        return m;
    }

    inline Mesh *addPlaneXZ(float planeY, float halfSize,
                            const Material &mat = Material(vec3(0.8f))) {
        // Square in XZ at y=planeY
        const vec3 A(-halfSize, planeY, -halfSize);
        const vec3 B(halfSize, planeY, -halfSize);
        const vec3 C(halfSize, planeY, halfSize);
        const vec3 D(-halfSize, planeY, halfSize);

        std::vector<Triangle> tris;
        tris.reserve(2);

        tris.emplace_back(A, C, B);
        tris.emplace_back(A, D, C);

        return addTriangles(tris, mat);
    }

    inline void addCheckerboardPlaneXZ(float planeY, int tilesPerSide,
                                       float tileSize, const Material &whiteMat,
                                       const Material &blackMat) {
        std::vector<Triangle> whiteTris, blackTris;
        whiteTris.reserve(tilesPerSide * tilesPerSide * 2);
        blackTris.reserve(tilesPerSide * tilesPerSide * 2);

        const int N = tilesPerSide;
        const float start = -N * tileSize;

        for (int iz = 0; iz < 2 * N; ++iz) {
            for (int ix = 0; ix < 2 * N; ++ix) {
                const float x0 = start + ix * tileSize;
                const float x1 = x0 + tileSize;
                const float z0 = start + iz * tileSize;
                const float z1 = z0 + tileSize;

                const vec3 A(x0, planeY, z0);
                const vec3 B(x1, planeY, z0);
                const vec3 C(x1, planeY, z1);
                const vec3 D(x0, planeY, z1);

                const bool white = ((ix + iz) & 1) == 0;
                auto &bucket = white ? whiteTris : blackTris;

                bucket.emplace_back(A, C, B);
                bucket.emplace_back(A, D, C);
            }
        }

        if (!whiteTris.empty())
            addTriangles(whiteTris, whiteMat);
        if (!blackTris.empty())
            addTriangles(blackTris, blackMat);
    }

    Mesh *addCube(const Material &mat = Material(vec3(1.0f, 0.0f, 0.0f))) {
        meshes.push_back(std::make_unique<Mesh>());
        mesh_materials.push_back(mat);
        return meshes.back().get();
    }

    // Light management
    void addPointLight(const vec3 &position, const vec3 &color,
                       float intensity = 1.0f, float range = 100.0f) {
        Light light;
        light.type = LIGHT_POINT;
        light.position = position;
        light.color = color;
        light.intensity = intensity;
        light.range = range;
        lights.push_back(light);
    }

    void addDirectionalLight(const vec3 &direction, const vec3 &color,
                             float intensity = 1.0f) {
        Light light;
        light.type = LIGHT_DIRECTIONAL;
        light.direction = direction.normalized();
        light.color = color;
        light.intensity = intensity;
        lights.push_back(light);
    }

    void addSpotLight(const vec3 &position, const vec3 &direction,
                      const vec3 &color, float intensity = 1.0f,
                      float innerCone = 0.5f, float outerCone = 0.7f,
                      float range = 100.0f) {
        Light light;
        light.type = LIGHT_SPOT;
        light.position = position;
        light.direction = direction.normalized();
        light.color = color;
        light.intensity = intensity;
        light.innerCone = cosf(innerCone);
        light.outerCone = cosf(outerCone);
        light.range = range;
        lights.push_back(light);
    }

    void setAmbientLight(const vec3 &ambient) { ambient_light = ambient; }

    // Background settings
    void setSkyGradient(const vec3 &top, const vec3 &bottom) {
        sky_color_top = top;
        sky_color_bottom = bottom;
        use_sky = true;
    }

    void disableSky() { use_sky = false; }

    // Upload scene to GPU
    void uploadToGPU() {
        if (meshes.empty()) {
            std::cerr << "Warning: No meshes in scene\n";
            return;
        }

        // Upload each mesh
        for (auto &mesh : meshes) {
            mesh->upload();
        }

        // Create mesh descriptors
        if (d_mesh_descriptors) {
            cudaFree(d_mesh_descriptors);
        }

        cudaError_t err = cudaMallocManaged(&d_mesh_descriptors,
                                            meshes.size() * sizeof(DeviceMesh));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate mesh descriptors");
        }

        // Fill descriptors
        for (size_t i = 0; i < meshes.size(); ++i) {
            DeviceMesh &desc = d_mesh_descriptors[i];
            desc.verts = meshes[i]->d_vertices;
            desc.faces = meshes[i]->d_faces;
            desc.faceCount = static_cast<int>(meshes[i]->faces.size());
            desc.material = mesh_materials[i];

            desc.bvhNodes = nullptr;
            desc.nodeCount = 0;
            desc.primIndices = nullptr;
        }

        // Upload lights
        if (!lights.empty()) {
            if (d_lights) {
                cudaFree(d_lights);
            }

            err = cudaMallocManaged(&d_lights, lights.size() * sizeof(Light));
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to allocate lights");
            }

            for (size_t i = 0; i < lights.size(); ++i) {
                d_lights[i] = lights[i];
            }
        }

        cudaDeviceSynchronize();
    }

    // Render the scene
    void render(unsigned char *output_pixels) {
        if (!d_mesh_descriptors || meshes.empty()) {
            std::cerr << "Error: Scene not uploaded to GPU\n";
            return;
        }

        // Configure kernel launch
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);

        // Launch render kernel
        render_kernel<<<grid, block>>>(
            d_pixels, width, height, camera, d_mesh_descriptors,
            static_cast<int>(meshes.size()), d_lights,
            static_cast<int>(lights.size()), ambient_light, sky_color_top,
            sky_color_bottom, use_sky);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Kernel launch failed: ") +
                                     cudaGetErrorString(err));
        }

        cudaDeviceSynchronize();

        // Copy result to host
        size_t nBytes = static_cast<size_t>(width) * height * 3;
        err =
            cudaMemcpy(output_pixels, d_pixels, nBytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy pixels from GPU");
        }
    }
    // Render directly into an external device pointer (e.g., CUDA-mapped GL
    // PBO). device_pixels must be a device pointer with at least
    // width*height*3 bytes (RGB8).
    void render_to_device(unsigned char *device_pixels) {
        if (meshes.empty()) {
            std::cerr << "Error: no meshes in scene\n";
            return;
        }

        // --- 1) Build/Upload per-mesh data + BVH and rebuild descriptors ---
        std::vector<DeviceMesh> h_mesh_desc(meshes.size());
        for (size_t i = 0; i < meshes.size(); ++i) {
            Mesh *m = meshes[i].get();

            // Make sure vertex/face arrays are on the device
            m->upload();

            // Build BVH on CPU + upload to GPU if needed
            if (m->bvhDirty || m->d_bvhNodes == nullptr) {
                // Use the Scene's BVH parameters for the mesh
                m->setBVHLeafParams(bvhLeafTarget_, bvhLeafTol_);
                m->buildBVH();
                m->uploadBVH();
            }

            // Fill device descriptor
            h_mesh_desc[i].verts = m->d_vertices;
            h_mesh_desc[i].faces = m->d_faces;
            h_mesh_desc[i].faceCount = static_cast<int>(m->faces.size());
            h_mesh_desc[i].material = mesh_materials[i];

            // BVH pointers
            h_mesh_desc[i].bvhNodes = m->d_bvhNodes;
            h_mesh_desc[i].nodeCount = static_cast<int>(m->bvhNodes.size());
            h_mesh_desc[i].primIndices = m->d_bvhPrim;
        }

        if (d_mesh_descriptors) {
            cudaFree(d_mesh_descriptors);
            d_mesh_descriptors = nullptr;
        }
        cudaMalloc(&d_mesh_descriptors,
                   sizeof(DeviceMesh) * h_mesh_desc.size());
        cudaMemcpy(d_mesh_descriptors, h_mesh_desc.data(),
                   sizeof(DeviceMesh) * h_mesh_desc.size(),
                   cudaMemcpyHostToDevice);

        // --- 2) Configure kernel ---
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);

        // --- 3) Launch ---
        render_kernel<<<grid, block>>>(
            device_pixels, width, height, camera, d_mesh_descriptors,
            static_cast<int>(meshes.size()), d_lights,
            static_cast<int>(lights.size()), ambient_light, sky_color_top,
            sky_color_bottom, use_sky);

        // --- 4) Check + sync ---
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Kernel launch failed: ") +
                                     cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
    }

    // Save to PPM file
    void saveAsPPM(const std::string &filename, unsigned char *pixels) const {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        ofs << "P3\n" << width << ' ' << height << "\n255\n";

        size_t idx = 0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                ofs << int(pixels[idx]) << ' ' << int(pixels[idx + 1]) << ' '
                    << int(pixels[idx + 2]) << '\n';
                idx += 3;
            }
        }
        ofs.close();
    }

    // Getters
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    size_t getPixelBufferSize() const {
        return static_cast<size_t>(width) * height * 3;
    }
    Camera &getCamera() { return camera; }
};

// ── CUDA kernel implementation
__global__ void render_kernel(unsigned char *out, int W, int H, Camera cam,
                              DeviceMesh *meshes, int nMeshes, Light *lights,
                              int nLights, vec3 ambientLight, vec3 skyColorTop,
                              vec3 skyColorBottom, bool useSky) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;

    const float u = (x + 0.5f) / W;
    const float v = 1.0f - (y + 0.5f) / H;

    Ray ray = cam.get_ray(u, v);
    HitInfo hit = traceRay(ray, meshes, nMeshes);

    vec3 color;
    if (hit.hit) {
        color = calculatePBRLighting(hit, ray, meshes, nMeshes, lights, nLights,
                                     ambientLight, skyColorTop, skyColorBottom,
                                     useSky);
    } else {
        color = useSky ? lerp(skyColorBottom, skyColorTop,
                              0.5f * (ray.direction().y + 1.f))
                       : vec3(0.0f);
    }

    color = color / (color + vec3(1.0f)); // Reinhard
    color = vec3(powf(color.x, 1.0f / 2.2f), powf(color.y, 1.0f / 2.2f),
                 powf(color.z, 1.0f / 2.2f));

    const vec3 rgb = clamp(color, 0.f, 1.f) * 255.99f;
    const int y_out = H - 1 - y;
    const size_t idx = (static_cast<size_t>(y_out) * W + x) * 3;
    out[idx + 0] = static_cast<unsigned char>(rgb.x);
    out[idx + 1] = static_cast<unsigned char>(rgb.y);
    out[idx + 2] = static_cast<unsigned char>(rgb.z);
}

// Predefined Scene Builders
namespace Scenes {

inline std::unique_ptr<Scene> createLitTestScene(int width = 800,
                                                 int height = 600) {
    auto scene = std::make_unique<Scene>(width, height);

    // Materials
    Material redMat(vec3(0.8f, 0.2f, 0.2f), 0.2f);
    redMat.specular = vec3(0.5f);

    Material blueMat(vec3(0.2f, 0.2f, 0.8f), 0.3f);
    blueMat.specular = vec3(0.3f);

    Material goldMat(vec3(0.9f, 0.7f, 0.3f), 0.15f, 1.0f); // metallic gold
    goldMat.specular = vec3(0.8f, 0.6f, 0.2f);

    // Add objects
    Mesh *cube = scene->addCube(redMat);
    cube->moveTo(vec3(-2, 0, -5));
    cube->scale(0.8f);

    Mesh *cube2 = scene->addCube(blueMat);
    cube2->moveTo(vec3(2, 0, -5));
    cube2->scale(0.8f);

    Mesh *cube3 = scene->addCube(goldMat);
    cube3->moveTo(vec3(0, 2, -5));
    cube3->scale(0.8f);

    // Add lights
    scene->addPointLight(vec3(5, 5, 0), vec3(1.0f, 0.9f, 0.8f), 2.0f, 50.0f);
    scene->addDirectionalLight(vec3(-0.3f, -0.8f, -0.5f),
                               vec3(0.9f, 0.9f, 1.0f), 0.5f);
    scene->addSpotLight(vec3(0, 4, -2), vec3(0, -1, -0.3f),
                        vec3(1.0f, 0.8f, 0.6f), 3.0f, 0.3f, 0.5f, 20.0f);

    // Set ambient light
    scene->setAmbientLight(vec3(0.05f, 0.05f, 0.08f));

    // Setup camera
    scene->setCamera(vec3(0, 2, 3),  // lookfrom
                     vec3(0, 0, -5), // lookat
                     vec3(0, 1, 0),  // up
                     60.0f           // fov
    );

    Material floorMat(vec3(0.8f));
    floorMat.specular = vec3(0.1f);

    scene->addPlaneXZ(/*planeY=*/-1.0f, /*halfSize=*/50.0f, floorMat);

    return scene;
}

inline std::unique_ptr<Scene> createOrbitScene(int width = 800,
                                               int height = 600) {
    auto scene = std::make_unique<Scene>(width, height);

    // Add multiple cubes in orbit with different materials
    float radius = 5.0f;
    int num_cubes = 8;
    for (int i = 0; i < num_cubes; ++i) {
        float angle = (TWO_PI * i) / num_cubes;
        float hue = static_cast<float>(i) / num_cubes;
        vec3 color(0.5f + 0.5f * cosf(TWO_PI * hue),
                   0.5f + 0.5f * cosf(TWO_PI * hue + TWO_PI / 3),
                   0.5f + 0.5f * cosf(TWO_PI * hue + 2 * TWO_PI / 3));

        Material mat(color, 32.0f + i * 8.0f);
        mat.metallic = static_cast<float>(i) / num_cubes;
        mat.specular = vec3(0.5f);

        Mesh *cube = scene->addCube(mat);
        cube->scale(0.5f);
        cube->moveTo(vec3(radius * cosf(angle), 0, -10 + radius * sinf(angle)));
        cube->rotateSelfEulerXYZ(vec3(angle, angle * 0.5f, 0));
    }

    // Add emissive cube in center
    Material emissiveMat(vec3(0.1f));
    emissiveMat.emission = vec3(2.0f, 1.5f, 1.0f);
    Mesh *centerCube = scene->addCube(emissiveMat);
    centerCube->moveTo(vec3(0, 0, -10));
    centerCube->scale(0.3f);

    // Add multiple colored lights
    scene->addPointLight(vec3(0, 8, -10), vec3(1.0f, 1.0f, 1.0f), 2.0f, 30.0f);
    scene->addPointLight(vec3(-8, 2, -5), vec3(1.0f, 0.3f, 0.3f), 1.5f, 20.0f);
    scene->addPointLight(vec3(8, 2, -15), vec3(0.3f, 0.3f, 1.0f), 1.5f, 20.0f);

    scene->setAmbientLight(vec3(0.02f));

    // Look at the center
    scene->setCamera(vec3(0, 5, 0),   // lookfrom
                     vec3(0, 0, -10), // lookat
                     vec3(0, 1, 0),   // up
                     60.0f            // fov
    );

    Material floorMat(vec3(0.8f));
    floorMat.specular = vec3(0.1f);

    scene->addPlaneXZ(/*planeY=*/-1.0f, /*halfSize=*/50.0f, floorMat);

    return scene;
}
} // namespace Scenes

namespace Materials {
// Metals
inline Material Gold() {
    Material m(vec3(1.0f, 0.766f, 0.336f), 0.1f, 1.0f);
    m.specular = vec3(1.0f, 0.782f, 0.344f);
    return m;
}

inline Material Silver() {
    Material m(vec3(0.972f, 0.960f, 0.915f), 0.05f, 1.0f);
    m.specular = vec3(0.972f, 0.960f, 0.915f);
    return m;
}

inline Material Copper() {
    Material m(vec3(0.955f, 0.637f, 0.538f), 0.15f, 1.0f);
    m.specular = vec3(0.955f, 0.637f, 0.538f);
    return m;
}

inline Material BrushedAluminum() {
    Material m(vec3(0.913f, 0.921f, 0.925f), 0.3f, 1.0f);
    m.anisotropy = 0.8f;
    return m;
}

// Dielectrics
inline Material Glass() {
    Material m(vec3(1.0f), 0.02f, 0.0f); // tiny roughness floor
    m.transmission = 0.98f;
    m.ior = 1.5f;
    m.specular = vec3(0.04f);
    return m;
}

inline Material FrostedGlass() {
    Material m = Glass();
    m.roughness = 0.3f; // surface microfacet roughness (reflections)
    m.transmissionRoughness =
        0.5f; // keep for future use if you add rough refraction
    return m;
}

inline Material Diamond() {
    Material m(vec3(1.0f), 0.0f, 0.0f);
    m.transmission = 0.95f;
    m.ior = 2.42f;
    m.specular = vec3(0.17f); // Higher F0 due to high IOR
    return m;
}

inline Material Water() {
    Material m(vec3(0.8f, 0.95f, 1.0f), 0.01f, 0.0f);
    m.transmission = 0.9f;
    m.ior = 1.33f;
    m.specular = vec3(0.02f);
    return m;
}

// Plastics
inline Material PlasticRed() {
    Material m(vec3(0.8f, 0.1f, 0.1f), 0.2f, 0.0f);
    m.specular = vec3(0.04f);
    return m;
}

inline Material RubberBlack() {
    Material m(vec3(0.05f), 0.8f, 0.0f);
    m.specular = vec3(0.03f);
    return m;
}

// Car Paint
inline Material CarPaint(const vec3 &baseColor) {
    Material m(baseColor, 0.2f, 0.3f);
    m.clearcoat = 1.0f;
    m.clearcoatRoughness = 0.03f;
    m.specular = vec3(0.05f);
    return m;
}

inline Material PearlescentPaint(const vec3 &baseColor) {
    Material m = CarPaint(baseColor);
    m.iridescence = 0.8f;
    m.iridescenceThickness = 400.0f;
    return m;
}

// Organic
inline Material Skin() {
    Material m(vec3(0.95f, 0.75f, 0.67f), 0.4f, 0.0f);
    m.subsurfaceColor = vec3(1.0f, 0.4f, 0.3f);
    m.subsurfaceRadius = 0.5f;
    m.specular = vec3(0.028f);
    return m;
}

inline Material Wax() {
    Material m(vec3(0.95f, 0.93f, 0.88f), 0.3f, 0.0f);
    m.subsurfaceColor = vec3(1.0f, 0.9f, 0.7f);
    m.subsurfaceRadius = 0.8f;
    m.specular = vec3(0.03f);
    return m;
}

inline Material Jade() {
    Material m(vec3(0.2f, 0.6f, 0.4f), 0.1f, 0.0f);
    m.subsurfaceColor = vec3(0.3f, 0.8f, 0.5f);
    m.subsurfaceRadius = 0.3f;
    m.specular = vec3(0.05f);
    return m;
}

// Fabrics
inline Material Velvet(const vec3 &color) {
    Material m(color, 0.8f, 0.0f);
    m.sheen = 1.0f;
    m.sheenTint = color * 1.2f;
    m.specular = vec3(0.02f);
    return m;
}

inline Material Silk(const vec3 &color) {
    Material m(color, 0.2f, 0.0f);
    m.sheen = 0.6f;
    m.sheenTint = vec3(1.0f);
    m.anisotropy = 0.5f;
    m.specular = vec3(0.04f);
    return m;
}

// Special Effects
inline Material SoapBubble() {
    Material m(vec3(1.0f), 0.0f, 0.0f);
    m.transmission = 0.95f;
    m.ior = 1.33f;
    m.iridescence = 1.0f;
    m.iridescenceThickness = 380.0f;
    m.specular = vec3(0.04f);
    return m;
}

inline Material OilSlick() {
    Material m(vec3(0.01f), 0.0f, 0.95f);
    m.iridescence = 1.0f;
    m.iridescenceThickness = 450.0f;
    return m;
}

inline Material EmissiveLamp(const vec3 &color, float intensity = 5.0f) {
    Material m(vec3(1.0f), 0.0f, 0.0f);
    m.emission = color * intensity;
    return m;
}

inline Material NeonLight(const vec3 &color) {
    Material m(color * 0.1f, 0.0f, 0.0f);
    m.emission = color * 10.0f;
    return m;
}

__host__ __device__ inline Material MarbleCarrara(bool polished = true) {
    const float baseRough = polished ? 0.15f : 0.35f; // surface micro-roughness
    const float coatAmt =
        polished ? 0.70f : 0.15f; // lacquer feel when polished
    const float coatRough =
        polished ? 0.05f : 0.20f; // sharper highlight when polished

    Material m(vec3(0.93f, 0.94f, 0.96f), baseRough, /*metallic=*/0.0f);
    m.ior = 1.49f; // calcite-ish
    m.clearcoat = coatAmt;
    m.clearcoatRoughness = coatRough;

    // very subtle warm SSS for depth
    m.subsurfaceColor = vec3(0.98f, 0.98f, 0.96f);
    m.subsurfaceRadius = 1.0f;

    // marble is opaque; use SSS instead of transmission
    m.transmission = 0.0f;
    m.transmissionRoughness = 0.0f;

    // keep the rest off for stone
    m.anisotropy = 0.0f;
    m.sheen = 0.0f;
    m.sheenTint = vec3(0.5f);
    m.iridescence = 0.0f;
    m.iridescenceThickness = 550.0f;

    return m;
}

// Nero Marquina-style black marble
__host__ __device__ inline Material MarbleNero(bool polished = true) {
    const float baseRough = polished ? 0.12f : 0.28f;
    const float coatAmt = polished ? 0.85f : 0.20f;
    const float coatRough = polished ? 0.04f : 0.18f;

    Material m(vec3(0.04f, 0.045f, 0.05f), baseRough, 0.0f);
    m.ior = 1.49f;
    m.clearcoat = coatAmt;
    m.clearcoatRoughness = coatRough;

    m.subsurfaceColor = vec3(0.15f, 0.15f, 0.16f);
    m.subsurfaceRadius = 0.6f;

    m.transmission = 0.0f;
    m.transmissionRoughness = 0.0f;

    m.anisotropy = 0.0f;
    m.sheen = 0.0f;
    m.iridescence = 0.0f;

    return m;
}

// Verde Alpi-style green marble
__host__ __device__ inline Material MarbleVerde(bool polished = true) {
    const float baseRough = polished ? 0.14f : 0.30f;
    const float coatAmt = polished ? 0.75f : 0.18f;
    const float coatRough = polished ? 0.05f : 0.19f;

    Material m(vec3(0.10f, 0.18f, 0.14f), baseRough, 0.0f);
    m.ior = 1.49f;
    m.clearcoat = coatAmt;
    m.clearcoatRoughness = coatRough;

    m.subsurfaceColor = vec3(0.12f, 0.20f, 0.16f);
    m.subsurfaceRadius = 0.8f;

    m.transmission = 0.0f;
    m.transmissionRoughness = 0.0f;

    m.anisotropy = 0.0f;
    m.sheen = 0.0f;
    m.iridescence = 0.0f;

    return m;
}

} // namespace Materials

#endif // SCENE_CUH