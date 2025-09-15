#ifndef MESH_CUH
#define MESH_CUH

#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ray.cuh"
#include "vec3.cuh"

struct AABB {
    vec3 bmin;
    vec3 bmax;

    // Standard slab test against current best tMax
    __host__ __device__ inline bool hit(const Ray &r, float tMax) const {
        float tmin = 1e-3f,
              tmax = tMax; // keep epsilon consistent with triangles
#pragma unroll
        for (int a = 0; a < 3; ++a) {
            const float invD = 1.0f / r.direction()[a]; // inf ok if dir[a]==0
            float t0 = (bmin[a] - r.origin()[a]) * invD;
            float t1 = (bmax[a] - r.origin()[a]) * invD;
            if (invD < 0.0f) {
                float tmp = t0;
                t0 = t1;
                t1 = tmp;
            }
            tmin = fmaxf(tmin, t0);
            tmax = fminf(tmax, t1);
            if (tmax <= tmin)
                return false;
        }
        return true;
    }

    // Same math, but return entry time for near/far ordering
    __host__ __device__ inline bool hit_t(const Ray &r, float tMax,
                                          float &tEnter) const {
        float tmin = 1e-3f, tmax = tMax;
#pragma unroll
        for (int a = 0; a < 3; ++a) {
            const float invD = 1.0f / r.direction()[a];
            float t0 = (bmin[a] - r.origin()[a]) * invD;
            float t1 = (bmax[a] - r.origin()[a]) * invD;
            if (invD < 0.0f) {
                float tmp = t0;
                t0 = t1;
                t1 = tmp;
            }
            tmin = fmaxf(tmin, t0);
            tmax = fminf(tmax, t1);
            if (tmax <= tmin)
                return false;
        }
        tEnter = tmin;
        return true;
    }

    __host__ __device__ inline vec3 extent() const { return bmax - bmin; }
    __host__ __device__ inline vec3 center() const {
        return (bmin + bmax) * 0.5f;
    }
    __host__ __device__ inline float radius() const {
        return 0.5f * extent().length();
    }

    __host__ __device__ static inline AABB make_invalid() {
        // Use huge/sentinel bounds; fine on device
        return {vec3(1e30f), vec3(-1e30f)};
    }

    __host__ __device__ inline void expand(const AABB &b) {
        bmin.x = fminf(bmin.x, b.bmin.x);
        bmin.y = fminf(bmin.y, b.bmin.y);
        bmin.z = fminf(bmin.z, b.bmin.z);
        bmax.x = fmaxf(bmax.x, b.bmax.x);
        bmax.y = fmaxf(bmax.y, b.bmax.y);
        bmax.z = fmaxf(bmax.z, b.bmax.z);
    }

    __host__ __device__ inline void expand(const vec3 &p) {
        bmin.x = fminf(bmin.x, p.x);
        bmin.y = fminf(bmin.y, p.y);
        bmin.z = fminf(bmin.z, p.z);
        bmax.x = fmaxf(bmax.x, p.x);
        bmax.y = fmaxf(bmax.y, p.y);
        bmax.z = fmaxf(bmax.z, p.z);
    }
};

struct DeviceBVHNode {
    AABB bbox; // bounds
    int left;  // child index (internal) or -1 for leaf
    int right; // child index (internal) or -1 for leaf
    int start; // start index into primIndices (leaf only)
    int count; // number of triangles in leaf (0 for internal)
};

struct Tri {
    int v0, v1, v2;
};

class Mesh {
  public:
    // host‑side data
    std::vector<vec3> vertices;
    std::vector<Tri> faces;

    // device copies (nullptr until upload())
    vec3 *d_vertices = nullptr;
    Tri *d_faces = nullptr;

    // --- BVH data (host) ---
    std::vector<DeviceBVHNode> bvhNodes;
    std::vector<int> bvhPrimIndices;

    // --- BVH data (device) ---
    DeviceBVHNode *d_bvhNodes = nullptr;
    int *d_bvhPrim = nullptr;

    // --- BVH config/dirty flags ---
    bool bvhDirty = true;
    int bvhLeafTarget = 12; // default target tris/leaf
    int bvhLeafTol = 5;     // default tolerance

    // BVH API
    void setBVHLeafParams(int target, int tol = 5) {
        bvhLeafTarget = target < 1 ? 1 : target;
        bvhLeafTol = tol < 0 ? 0 : tol;
        bvhDirty = true;
    }
    void buildBVH();      // CPU build -> bvhNodes + bvhPrimIndices
    void uploadBVH();     // alloc+copy to device
    void freeBVHDevice(); // free device arrays

    // ------- ctors / dtor -------
    Mesh();                                 // unit cube
    explicit Mesh(const std::string &path); // load OBJ
    ~Mesh();

    // prevents accidental shallow copies
    Mesh(const Mesh &) = delete;
    Mesh &operator=(const Mesh &) = delete;

    // move‑support (for convenience)
    Mesh(Mesh &&other) noexcept { *this = std::move(other); }

    Mesh &operator=(Mesh &&other) noexcept {
        vertices = std::move(other.vertices);
        faces = std::move(other.faces);
        d_vertices = other.d_vertices;
        other.d_vertices = nullptr;
        d_faces = other.d_faces;
        other.d_faces = nullptr;

        // Move BVH data
        bvhNodes = std::move(other.bvhNodes);
        bvhPrimIndices = std::move(other.bvhPrimIndices);
        d_bvhNodes = other.d_bvhNodes;
        other.d_bvhNodes = nullptr;
        d_bvhPrim = other.d_bvhPrim;
        other.d_bvhPrim = nullptr;
        bvhDirty = other.bvhDirty;
        bvhLeafTarget = other.bvhLeafTarget;
        bvhLeafTol = other.bvhLeafTol;

        return *this;
    }

    // GPU helpers
    void upload();     // alloc + memcpy to device
    void freeDevice(); // free device buffers (safe if already freed)

    AABB boundingBox() const;

    void scale(float s); // uniform scale about the origin
    void scale(vec3 s);
    void translate(const vec3 &d); // add offset to every vertex
    void moveTo(const vec3 &p);    // place bbox-center at p
    void rotateSelfEulerXYZ(const vec3 &rad);

    __host__ __device__ size_t faceCount() const { return faces.size(); }
    __host__ __device__ size_t vertexCount() const { return vertices.size(); }
};

// ---------------- implementation ----------------

inline Mesh::Mesh() {
    // create a unit cube centered at (0,0,-3)
    vertices = {
        {-0.5f, -0.5f, -3.5f}, {0.5f, -0.5f, -3.5f},
        {0.5f, 0.5f, -3.5f},   {-0.5f, 0.5f, -3.5f}, // back
        {-0.5f, -0.5f, -2.5f}, {0.5f, -0.5f, -2.5f},
        {0.5f, 0.5f, -2.5f},   {-0.5f, 0.5f, -2.5f} // front
    };

    faces = {// back face  (looking toward -Z)
             {0, 2, 1},
             {0, 3, 2},

             // front face (looking toward +Z)
             {4, 5, 6},
             {4, 6, 7},

             // bottom face (looking toward -Y)
             {0, 1, 5},
             {0, 5, 4},

             // top face (looking toward +Y)
             {3, 7, 6},
             {3, 6, 2},

             // left face (looking toward -X)
             {0, 4, 7},
             {0, 7, 3},

             // right face (looking toward +X)
             {1, 2, 6},
             {1, 6, 5}};
}

inline Mesh::Mesh(const std::string &path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Mesh: cannot open " + path);

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if (key == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            vertices.emplace_back(x, y, z);
        } else if (key == "f") {
            std::vector<int> idx;
            std::string vert;
            while (ss >> vert) {
                size_t slash = vert.find('/');
                int id = std::stoi(
                    slash == std::string::npos ? vert : vert.substr(0, slash));
                idx.push_back(id - 1); // OBJ is 1‑based
            }
            if (idx.size() < 3)
                continue; // ignore degenerate
            // triangulate fan if quad/ngon
            for (size_t i = 1; i + 1 < idx.size(); ++i) {
                faces.push_back({idx[0], idx[i], idx[i + 1]});
            }
        }
    }
    if (vertices.empty() || faces.empty())
        throw std::runtime_error("Mesh: no geometry in " + path);
}

inline void Mesh::upload() {
    freeDevice(); // in case called twice
    if (vertices.empty() || faces.empty())
        return;
    cudaMalloc(&d_vertices, sizeof(vec3) * vertices.size());
    cudaMalloc(&d_faces, sizeof(Tri) * faces.size());
    cudaMemcpy(d_vertices, vertices.data(), sizeof(vec3) * vertices.size(),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_faces, faces.data(), sizeof(Tri) * faces.size(),
               cudaMemcpyHostToDevice);
}

inline void Mesh::freeDevice() {
    if (d_vertices) {
        cudaFree(d_vertices);
        d_vertices = nullptr;
    }
    if (d_faces) {
        cudaFree(d_faces);
        d_faces = nullptr;
    }
}

inline void Mesh::freeBVHDevice() {
    if (d_bvhNodes) {
        cudaFree(d_bvhNodes);
        d_bvhNodes = nullptr;
    }
    if (d_bvhPrim) {
        cudaFree(d_bvhPrim);
        d_bvhPrim = nullptr;
    }
}

inline static AABB tri_bounds(const vec3 &a, const vec3 &b, const vec3 &c) {
    AABB box = {a, a};
    box.expand(b);
    box.expand(c);
    return box;
}

struct _BuildRef {
    int f;  // face index
    vec3 c; // centroid
    AABB b; // bounds
};

// In mesh.cuh, replace the buildBVH method (around line 280) with this fixed
// version:

inline void Mesh::buildBVH() {
    bvhNodes.clear();
    bvhPrimIndices.clear();

    if (faces.empty()) {
        bvhDirty = false;
        return;
    }

    // 1) build refs
    std::vector<_BuildRef> refs;
    refs.reserve(faces.size());
    for (int i = 0; i < (int)faces.size(); ++i) {
        const Tri t = faces[i];
        const vec3 &v0 = vertices[t.v0];
        const vec3 &v1 = vertices[t.v1];
        const vec3 &v2 = vertices[t.v2];
        _BuildRef r;
        r.f = i;
        r.b = tri_bounds(v0, v1, v2);
        r.c = (v0 + v1 + v2) * (1.0f / 3.0f);
        refs.push_back(r);
    }

    const int leafMax = bvhLeafTarget + bvhLeafTol;

    // 2) recursive lambda
    struct _Builder {
        std::vector<DeviceBVHNode> &nodes;
        std::vector<int> &prims;
        std::vector<_BuildRef> &R;
        int leafMax;

        int build(int begin, int end) {
            // compute bounds + centroid bounds
            AABB bb = AABB::make_invalid();
            AABB cb = AABB::make_invalid();
            for (int i = begin; i < end; ++i) {
                bb.expand(R[i].b);
                cb.expand(R[i].c);
            }
            int n = end - begin;

            int me = (int)nodes.size();
            nodes.emplace_back(); // placeholder

            nodes[me].bbox = bb;
            nodes[me].left = -1;
            nodes[me].right = -1;
            nodes[me].start = -1;
            nodes[me].count = 0;

            if (n <= leafMax) {
                nodes[me].start = (int)prims.size();
                nodes[me].count = n;
                prims.reserve(prims.size() + n);
                for (int i = begin; i < end; ++i)
                    prims.push_back(R[i].f);
                return me;
            }

            // choose split axis = longest centroid axis
            vec3 e = cb.extent();
            int axis = (e.x > e.y && e.x > e.z) ? 0 : ((e.y > e.z) ? 1 : 2);

            int mid = (begin + end) / 2;
            std::nth_element(R.begin() + begin, R.begin() + mid,
                             R.begin() + end,
                             [axis](const _BuildRef &A, const _BuildRef &B) {
                                 return A.c[axis] < B.c[axis];
                             });

            int L = build(begin, mid);
            int Rn = build(mid, end);
            nodes[me].left = L;
            nodes[me].right = Rn;
            return me;
        }
    };

    _Builder B{bvhNodes, bvhPrimIndices, refs, leafMax};
    B.build(0, (int)refs.size());

    bvhDirty = false;
}

inline void Mesh::uploadBVH() {
    freeBVHDevice();
    if (bvhNodes.empty())
        return;

    cudaMalloc(&d_bvhNodes, sizeof(DeviceBVHNode) * bvhNodes.size());
    cudaMemcpy(d_bvhNodes, bvhNodes.data(),
               sizeof(DeviceBVHNode) * bvhNodes.size(), cudaMemcpyHostToDevice);

    // Add safety check for empty primIndices
    if (!bvhPrimIndices.empty()) {
        cudaMalloc(&d_bvhPrim, sizeof(int) * bvhPrimIndices.size());
        cudaMemcpy(d_bvhPrim, bvhPrimIndices.data(),
                   sizeof(int) * bvhPrimIndices.size(), cudaMemcpyHostToDevice);
    } else {
        d_bvhPrim = nullptr;
    }
}

inline Mesh::~Mesh() {
    freeDevice();
    freeBVHDevice();
}

inline AABB Mesh::boundingBox() const {
    if (vertices.empty())
        return {vec3(0.0f), vec3(0.0f)};

    vec3 vmin = vertices[0];
    vec3 vmax = vertices[0];

    for (size_t i = 1; i < vertices.size(); ++i) {
        const vec3 &v = vertices[i];
        vmin.x = fminf(vmin.x, v.x);
        vmax.x = fmaxf(vmax.x, v.x);
        vmin.y = fminf(vmin.y, v.y);
        vmax.y = fmaxf(vmax.y, v.y);
        vmin.z = fminf(vmin.z, v.z);
        vmax.z = fmaxf(vmax.z, v.z);
    }
    return {vmin, vmax};
}

inline void Mesh::scale(float s) {
    for (auto &v : vertices) {
        v.x *= s;
        v.y *= s;
        v.z *= s;
    }

    bvhDirty = true; // scaling changes the bounding box
}
inline void Mesh::scale(vec3 s) {
    for (auto &v : vertices) {
        v.x *= s.x;
        v.y *= s.y;
        v.z *= s.z;
    }

    bvhDirty = true;
}
inline void Mesh::translate(const vec3 &d) {
    for (auto &v : vertices) {
        v.x += d.x;
        v.y += d.y;
        v.z += d.z;
    }

    bvhDirty = true;
}

inline void Mesh::moveTo(const vec3 &p) {
    AABB box = boundingBox();
    vec3 center = box.center();
    translate(p - center); // reuse translate() with the computed offset

    bvhDirty = true;
}

// ── helpers
inline static vec3 rX(const vec3 &v, float c, float s) {
    return {v.x, c * v.y - s * v.z, s * v.y + c * v.z};
}
inline static vec3 rY(const vec3 &v, float c, float s) {
    return {c * v.x + s * v.z, v.y, -s * v.x + c * v.z};
}
inline static vec3 rZ(const vec3 &v, float c, float s) {
    return {c * v.x - s * v.y, s * v.x + c * v.y, v.z};
}

inline void Mesh::rotateSelfEulerXYZ(const vec3 &rad) {
    // 1. compute center once
    vec3 center = boundingBox().center();

    // 2. pre-compute sin/cos
    const float cx = cosf(rad.x), sx = sinf(rad.x);
    const float cy = cosf(rad.y), sy = sinf(rad.y);
    const float cz = cosf(rad.z), sz = sinf(rad.z);

    // 3. translate rotate translate-back in one pass
    for (auto &v : vertices) {
        vec3 p = v - center; // to origin
        p = rX(p, cx, sx);   // X
        p = rY(p, cy, sy);   // Y
        p = rZ(p, cz, sz);   // Z
        v = p + center;      // back to place
    }

    bvhDirty = true;
}

#endif