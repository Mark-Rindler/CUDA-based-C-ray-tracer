#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "scene.cuh"

#define GLFW_INCLUDE_NONE
#include "glfw_view_interop.hpp"

#define CUDA_CHECK(stmt)                                                       \
    do {                                                                       \
        cudaError_t err = (stmt);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Scene Creation Functions
namespace DemoScenes {

std::unique_ptr<Scene> createCornellBox(int width = 800, int height = 800) {
    auto scene = std::make_unique<Scene>(width, height);

    // Walls/floor/ceiling: matte-ish dielectrics
    Material whiteMat(vec3(0.73f, 0.73f, 0.73f), 0.6f, 0.0f);
    whiteMat.specular = vec3(0.04f);

    Material redMat(vec3(0.65f, 0.05f, 0.05f), 0.6f, 0.0f);
    redMat.specular = vec3(0.04f);

    Material greenMat(vec3(0.12f, 0.45f, 0.15f), 0.6f, 0.0f);
    greenMat.specular = vec3(0.04f);

    Material lightMat(vec3(0.0f), 0.0f, 0.0f);
    lightMat.emission = vec3(15.0f); // area light cube

    // Add walls
    Mesh *back = scene->addCube(whiteMat);
    back->scale(vec3(10.0f, 10.0f, 0.1f));
    back->moveTo(vec3(0, 0, -10));

    Mesh *left = scene->addCube(redMat);
    left->scale(vec3(0.1f, 10.0f, 10.0f));
    left->moveTo(vec3(-5, 0, -5));

    Mesh *right = scene->addCube(greenMat);
    right->scale(vec3(0.1f, 10.0f, 10.0f));
    right->moveTo(vec3(5, 0, -5));

    Mesh *floor = scene->addCube(whiteMat);
    floor->scale(vec3(10.0f, 0.1f, 10.0f));
    floor->moveTo(vec3(0, -5, -5));

    Mesh *ceiling = scene->addCube(whiteMat);
    ceiling->scale(vec3(10.0f, 0.1f, 10.0f));
    ceiling->moveTo(vec3(0, 5, -5));

    // Emissive “ceiling panel”
    Mesh *light = scene->addCube(lightMat);
    light->scale(vec3(2.0f, 0.1f, 2.0f));
    light->moveTo(vec3(0, 4.9f, -5));

    // Boxes
    Material boxMat(vec3(0.9f), 0.2f, 0.0f); // a bit glossier
    boxMat.specular = vec3(0.04f);

    Mesh *box1 = scene->addCube(boxMat);
    box1->scale(vec3(1.5f, 3.0f, 1.5f));
    box1->moveTo(vec3(-1.5f, -3.5f, -6));
    box1->rotateSelfEulerXYZ(vec3(0, 0.3f, 0));

    Mesh *box2 = scene->addCube(boxMat);
    box2->scale(vec3(1.5f, 1.5f, 1.5f));
    box2->moveTo(vec3(1.5f, -4.25f, -4));
    box2->rotateSelfEulerXYZ(vec3(0, -0.4f, 0));

    // Single point light to supplement emissive
    scene->addPointLight(vec3(0, 4.5f, -5), vec3(1.0f, 0.9f, 0.8f), 3.0f,
                         20.0f);
    scene->setAmbientLight(vec3(0.02f));

    // Camera
    scene->setCamera(vec3(0, 0, 5), vec3(0, 0, -5), vec3(0, 1, 0), 40.0f);

    scene->disableSky();
    return scene;
}

// Showcase different material properties (grid)
std::unique_ptr<Scene> createMaterialShowcase1(int width = 1200,
                                               int height = 800) {
    auto scene = std::make_unique<Scene>(width, height);

    const int rows = 3;
    const int cols = 5;
    const float spacing = 2.5f;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float metallic = static_cast<float>(j) / (cols - 1);  // 0..1
            float roughness = static_cast<float>(i) / (rows - 1); // 0..1

            Material mat(vec3(0.8f, 0.3f, 0.2f), roughness, metallic);
            mat.specular = vec3(
                0.04f); // dielectric F0; metallic path will colorize via shader

            Mesh *sphere = scene->addCube(mat);
            sphere->scale(0.8f);
            float x = (j - cols / 2.0f) * spacing;
            float y = (i - rows / 2.0f) * spacing;
            sphere->moveTo(vec3(x, y, -10));
        }
    }

    // Three-point lighting
    scene->addPointLight(vec3(10, 10, 0), vec3(1.0f, 0.95f, 0.9f), 3.0f,
                         50.0f); // Key
    scene->addPointLight(vec3(-10, 5, 5), vec3(0.4f, 0.4f, 0.5f), 2.0f,
                         40.0f); // Fill
    scene->addPointLight(vec3(0, 15, -15), vec3(0.8f, 0.8f, 1.0f), 1.5f,
                         40.0f); // Rim

    scene->setAmbientLight(vec3(0.03f));
    scene->setCamera(vec3(0, 0, 5), vec3(0, 0, -10), vec3(0, 1, 0), 45.0f);

    Material floorMat(vec3(0.8f), 0.4f, 0.0f);
    floorMat.specular = vec3(0.04f);
    scene->addPlaneXZ(-10.0f, 50.0f, floorMat);

    return scene;
}

std::unique_ptr<Scene> createLightShow(int width = 1024, int height = 768) {
    auto scene = std::make_unique<Scene>(width, height);

    // Chrome-like metal (low roughness, high metallic)
    Mesh *centerSphere = scene->addCube(Materials::Water());
    centerSphere->scale(2.0f);
    centerSphere->moveTo(vec3(0, 0, -10));

    // Surrounding objects
    const int numObjects = 12;
    const float radius = 6.0f;

    for (int i = 0; i < numObjects; ++i) {
        float angle = (TWO_PI * i) / numObjects;
        float hue = static_cast<float>(i) / numObjects;

        vec3 color(0.5f + 0.5f * cosf(TWO_PI * hue),
                   0.5f + 0.5f * cosf(TWO_PI * hue + TWO_PI / 3),
                   0.5f + 0.5f * cosf(TWO_PI * hue + 2 * TWO_PI / 3));

        Material mat(color, 0.25f, (i % 2) ? 0.8f : 0.2f);
        mat.specular = vec3(0.04f);

        Mesh *obj = scene->addCube(mat);
        obj->scale(0.7f);
        obj->moveTo(vec3(radius * cosf(angle), 2.0f * sinf(angle * 2),
                         -10 + radius * sinf(angle)));
        obj->rotateSelfEulerXYZ(vec3(angle, angle * 0.5f, 0));
    }

    // Lights
    scene->addPointLight(vec3(5, 3, -5), vec3(1.0f, 0.2f, 0.2f), 3.0f, 30.0f);
    scene->addPointLight(vec3(-5, 3, -5), vec3(0.2f, 1.0f, 0.2f), 3.0f, 30.0f);
    scene->addPointLight(vec3(0, -3, -5), vec3(0.2f, 0.2f, 1.0f), 3.0f, 30.0f);
    scene->addPointLight(vec3(0, 8, -10), vec3(1.0f, 1.0f, 1.0f), 2.0f, 40.0f);
    scene->addSpotLight(vec3(0, 10, 0), vec3(0, -1, -0.5f),
                        vec3(1.0f, 0.9f, 0.7f), 4.0f, 0.2f, 0.4f, 30.0f);

    scene->setAmbientLight(vec3(0.01f));
    scene->setCamera(vec3(8, 5, 8), vec3(0, 0, -10), vec3(0, 1, 0), 50.0f);

    Material floorMat(vec3(0.8f), 0.4f, 0.0f);
    floorMat.specular = vec3(0.04f);
    scene->addPlaneXZ(-5.0f, 50.0f, floorMat);

    return scene;
}

std::unique_ptr<Scene> createArchitectural(int width = 1280, int height = 720) {
    auto scene = std::make_unique<Scene>(width, height);

    Material concrete(vec3(0.7f, 0.7f, 0.65f), 0.6f, 0.0f);
    concrete.specular = vec3(0.04f);

    // Thin glass panels (dielectric with transmission)
    Material glass(vec3(0.98f), 0.02f, 0.0f);
    glass.specular = vec3(0.04f);
    glass.transmission = 0.98f;
    glass.ior = 1.5f;

    Material wood(vec3(0.55f, 0.35f, 0.2f), 0.45f, 0.0f);
    wood.specular = vec3(0.04f);

    // Pillars
    for (int i = 0; i < 5; ++i) {
        Mesh *pillar = scene->addCube(concrete);
        pillar->scale(vec3(0.5f, 8.0f, 0.5f));
        pillar->moveTo(vec3(-8.0f + i * 4.0f, 0.0f, -15.0f));
    }

    // Glass panels
    for (int i = 0; i < 4; ++i) {
        Mesh *panel = scene->addCube(glass);
        panel->scale(vec3(3.8f, 6.0f, 0.1f));
        panel->moveTo(vec3(-6.0f + i * 4.0f, 0.0f, -14.5f));
    }

    // Floor / ceiling
    Mesh *floor = scene->addCube(wood);
    floor->scale(vec3(20.0f, 0.2f, 20.0f));
    floor->moveTo(vec3(0, -4, -15));

    Mesh *ceiling = scene->addCube(concrete);
    ceiling->scale(vec3(20.0f, 0.5f, 20.0f));
    ceiling->moveTo(vec3(0, 4, -15));

    // Lighting
    scene->addDirectionalLight(vec3(-0.3f, -0.6f, -0.5f),
                               vec3(1.0f, 0.95f, 0.8f), 1.5f);
    for (int i = 0; i < 3; ++i) {
        scene->addPointLight(vec3(-4.0f + i * 4.0f, 3, -12.0f),
                             vec3(1.0f, 0.9f, 0.7f), 0.8f, 15.0f);
    }

    scene->setAmbientLight(vec3(0.15f, 0.15f, 0.2f));
    scene->setCamera(vec3(10, 2, 0), vec3(0, 0, -15), vec3(0, 1, 0), 60.0f);

    Material ground(vec3(0.8f), 0.4f, 0.0f);
    ground.specular = vec3(0.04f);
    scene->addPlaneXZ(-10.0f, 50.0f, ground);

    return scene;
}

// Uses Materials::* presets
inline std::unique_ptr<Scene> createMaterialShowcase(int width = 1024,
                                                     int height = 768) {
    auto scene = std::make_unique<Scene>(width, height);

    const int gridSize = 5;
    const float spacing = 2.5f;
    const float startX = -(gridSize - 1) * spacing / 2.0f;
    const float startZ = -10.0f;

    // Row 1: Metals
    Mesh *m1 = scene->addCube(Materials::Gold());
    m1->moveTo(vec3(startX + 0 * spacing, 0, startZ));
    m1->scale(0.8f);
    Mesh *m2 = scene->addCube(Materials::Silver());
    m2->moveTo(vec3(startX + 1 * spacing, 0, startZ));
    m2->scale(0.8f);
    Mesh *m3 = scene->addCube(Materials::Copper());
    m3->moveTo(vec3(startX + 2 * spacing, 0, startZ));
    m3->scale(0.8f);
    Mesh *m4 = scene->addCube(Materials::BrushedAluminum());
    m4->moveTo(vec3(startX + 3 * spacing, 0, startZ));
    m4->scale(0.8f);
    Mesh *m5 = scene->addCube(Materials::OilSlick());
    m5->moveTo(vec3(startX + 4 * spacing, 0, startZ));
    m5->scale(0.8f);

    // Row 2: Dielectrics / Glass
    Mesh *m6 = scene->addCube(Materials::Glass());
    m6->moveTo(vec3(startX + 0 * spacing, 0, startZ - spacing));
    m6->scale(0.8f);
    Mesh *m7 = scene->addCube(Materials::FrostedGlass());
    m7->moveTo(vec3(startX + 1 * spacing, 0, startZ - spacing));
    m7->scale(0.8f);
    Mesh *m8 = scene->addCube(Materials::Diamond());
    m8->moveTo(vec3(startX + 2 * spacing, 0, startZ - spacing));
    m8->scale(0.8f);
    Mesh *m9 = scene->addCube(Materials::SoapBubble());
    m9->moveTo(vec3(startX + 3 * spacing, 0, startZ - spacing));
    m9->scale(0.8f);
    Mesh *m10 = scene->addCube(Materials::Water());
    m10->moveTo(vec3(startX + 4 * spacing, 0, startZ - spacing));
    m10->scale(0.8f);

    // Row 3: Car paints / organics
    Mesh *m11 = scene->addCube(Materials::CarPaint(vec3(0.8f, 0.1f, 0.1f)));
    m11->moveTo(vec3(startX + 0 * spacing, 0, startZ - 2 * spacing));
    m11->scale(0.8f);

    Mesh *m12 =
        scene->addCube(Materials::PearlescentPaint(vec3(0.9f, 0.9f, 1.0f)));
    m12->moveTo(vec3(startX + 1 * spacing, 0, startZ - 2 * spacing));
    m12->scale(0.8f);

    Mesh *m13 = scene->addCube(Materials::Skin());
    m13->moveTo(vec3(startX + 2 * spacing, 0, startZ - 2 * spacing));
    m13->scale(0.8f);

    Mesh *m14 = scene->addCube(Materials::Jade());
    m14->moveTo(vec3(startX + 3 * spacing, 0, startZ - 2 * spacing));
    m14->scale(0.8f);

    Mesh *m15 = scene->addCube(Materials::Wax());
    m15->moveTo(vec3(startX + 4 * spacing, 0, startZ - 2 * spacing));
    m15->scale(0.8f);

    // Row 4: Fabrics + emissive
    Mesh *m16 = scene->addCube(Materials::Velvet(vec3(0.5f, 0.1f, 0.6f)));
    m16->moveTo(vec3(startX + 0 * spacing, 0, startZ - 3 * spacing));
    m16->scale(0.8f);

    Mesh *m17 = scene->addCube(Materials::Silk(vec3(0.1f, 0.3f, 0.8f)));
    m17->moveTo(vec3(startX + 1 * spacing, 0, startZ - 3 * spacing));
    m17->scale(0.8f);

    Mesh *m18 = scene->addCube(Materials::PlasticRed());
    m18->moveTo(vec3(startX + 2 * spacing, 0, startZ - 3 * spacing));
    m18->scale(0.8f);

    Mesh *m19 = scene->addCube(Materials::RubberBlack());
    m19->moveTo(vec3(startX + 3 * spacing, 0, startZ - 3 * spacing));
    m19->scale(0.8f);

    Mesh *m20 = scene->addCube(Materials::NeonLight(vec3(0.3f, 0.8f, 1.0f)));
    m20->moveTo(vec3(startX + 4 * spacing, 0, startZ - 3 * spacing));
    m20->scale(0.8f);

    // Lighting
    scene->addPointLight(vec3(0, 8, -8), vec3(1.0f), 3.0f, 50.0f);
    scene->addPointLight(vec3(-8, 4, -4), vec3(1.0f, 0.9f, 0.8f), 2.0f, 30.0f);
    scene->addPointLight(vec3(8, 4, -12), vec3(0.8f, 0.9f, 1.0f), 2.0f, 30.0f);

    scene->setAmbientLight(vec3(0.03f));

    // Floor (slightly glossy, with clearcoat sheen)
    Material floorMat(vec3(0.9f), 0.05f, 0.0f);
    floorMat.specular = vec3(0.04f);
    floorMat.clearcoat = 0.5f;
    floorMat.clearcoatRoughness = 0.1f;
    scene->addPlaneXZ(-1.5f, 50.0f, floorMat);

    // Camera & sky
    scene->setCamera(vec3(0, 6, 5), vec3(0, -0.5f, -10), vec3(0, 1, 0), 45.0f);
    scene->setSkyGradient(vec3(0.05f, 0.05f, 0.08f), vec3(0.02f, 0.02f, 0.03f));

    return scene;
}

} // namespace DemoScenes

// ── Utility Functions ────────────────────────────────────────────────────────
void printUsage(const char *programName) {
    std::cout << "\nUsage: " << programName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -s, --scene <id>    Select scene (0-" << 6 << ")\n";
    std::cout << "  -w, --width <size>  Set image width (default: 800)\n";
    std::cout << "  -h, --height <size> Set image height (default: 600)\n";
    std::cout << "  -o, --output <name> Output filename (without extension)\n";
    std::cout << "  --help              Show this help message\n\n";
    std::cout << "Scenes:\n";
    std::cout << "  0: Lit Test Scene (basic lighting demo)\n";
    std::cout << "  1: Orbit Scene (rotating objects)\n";
    std::cout << "  2: Cornell Box (classic rendering test)\n";
    std::cout << "  3: Material Showcase (metallic/roughness grid)\n";
    std::cout << "  4: Light Show (multiple colored lights)\n";
    std::cout << "  5: Architectural (minimalist interior)\n";
    std::cout << "  6: Custom Scene (user-defined)\n\n";
}

struct RenderConfig {
    int sceneId = 0;
    int width = 800;
    int height = 600;
    std::string outputName = "output";
    bool showHelp = false;
    bool headlessOnce = false;

    int bvhLeafTarget = 12;
    int bvhLeafTol = 5;
};

RenderConfig parseArguments(int argc, char *argv[]) {
    RenderConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            config.showHelp = true;
            return config;
        } else if ((arg == "-s" || arg == "--scene") && i + 1 < argc) {
            config.sceneId = std::atoi(argv[++i]);
        } else if ((arg == "-w" || arg == "--width") && i + 1 < argc) {
            config.width = std::atoi(argv[++i]);
        } else if ((arg == "-h" || arg == "--height") && i + 1 < argc) {
            config.height = std::atoi(argv[++i]);
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            config.outputName = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            config.showHelp = true;
            return config;
        }
    }

    return config;
}

void printRenderInfo(const std::string &sceneName, int width, int height) {
    std::cout << "Scene:      " << std::left << std::setw(26) << sceneName
              << "\n";
    std::cout << "Resolution: " << std::setw(26)
              << (std::to_string(width) + " x " + std::to_string(height))
              << "\n";
}

struct CameraController {
    // State
    vec3 pos{0, 0, 3};
    float yaw = -90.0f;        // degrees, -Z forward by default
    float pitch = 0.0f;        // degrees
    float speed = 5.0f;        // units per second
    float sensitivity = 0.12f; // deg per pixel
    bool captureMouse = true;

    // mouse book-keeping
    double lastX = 0.0, lastY = 0.0;
    bool firstMouse = true;

    void initFromScene(Scene &s, int W, int H) {
        pos = s.cameraOrigin();
        vec3 f = s.cameraForward();
        // yaw: atan2(z,x), pitch: asin(y)
        yaw = std::atan2(f.z, f.x) * 180.0f / 3.14159265f;
        pitch = std::asin(std::clamp(f.y, -1.0f, 1.0f)) * 180.0f / 3.14159265f;
        lastX = W * 0.5;
        lastY = H * 0.5;
        firstMouse = true;
    }

    static vec3 rightFromForward(const vec3 &f) {
        vec3 up(0, 1, 0);
        return cross(f, up).normalized();
    }
    static vec3 forwardFromYawPitch(float yawDeg, float pitchDeg) {
        float cy = std::cos(yawDeg * 3.14159265f / 180.0f);
        float sy = std::sin(yawDeg * 3.14159265f / 180.0f);
        float cp = std::cos(pitchDeg * 3.14159265f / 180.0f);
        float sp = std::sin(pitchDeg * 3.14159265f / 180.0f);
        // OpenGL-ish right-handed: X right, Y up, -Z forward for yaw=-90
        return vec3(cy * cp, sp, sy * cp).normalized();
    }

    void applyMouse(GLFWwindow *win, float /*dt*/) {
        if (!captureMouse)
            return;
        double x, y;
        glfwGetCursorPos(win, &x, &y);
        if (firstMouse) {
            lastX = x;
            lastY = y;
            firstMouse = false;
        }
        double dx = x - lastX;
        double dy = lastY - y; // invert
        lastX = x;
        lastY = y;

        yaw += float(dx) * sensitivity;
        pitch += float(dy) * sensitivity;
        pitch = std::clamp(pitch, -89.9f, 89.9f);
    }

    void applyKeyboard(GLFWwindow *win, float dt) {
        float boost =
            (glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) ? 2.5f : 1.0f;
        float v = speed * boost * dt;

        vec3 fwd = forwardFromYawPitch(yaw, pitch);
        vec3 right = rightFromForward(fwd);
        vec3 up(0, 1, 0);

        if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS)
            pos = pos + fwd * v;
        if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS)
            pos = pos - fwd * v;
        if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS)
            pos = pos - right * v;
        if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS)
            pos = pos + right * v;
        if (glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_PRESS)
            pos = pos + up * v;
        if (glfwGetKey(win, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
            pos = pos - up * v;

        // Press C to toggle mouse capture
        static bool cPrev = false;
        bool cNow = (glfwGetKey(win, GLFW_KEY_C) == GLFW_PRESS);
        if (cNow && !cPrev) {
            captureMouse = !captureMouse;
            glfwSetInputMode(win, GLFW_CURSOR,
                             captureMouse ? GLFW_CURSOR_DISABLED
                                          : GLFW_CURSOR_NORMAL);
            firstMouse = true;
        }
        cPrev = cNow;
    }

    void update(Scene &s, GLFWwindow *win, float dt) {
        applyMouse(win, dt);
        applyKeyboard(win, dt);
        vec3 fwd = forwardFromYawPitch(yaw, pitch);
        s.moveCamera(pos);
        s.lookCameraAt(pos + fwd, vec3(0, 1, 0));
    }
};

static std::pair<std::unique_ptr<Scene>, std::string>
buildSceneById(RenderConfig configs) {
    std::unique_ptr<Scene> scene;
    std::string sceneName;

    switch (configs.sceneId) {
    case 0:
        sceneName = "Lit Test Scene";
        scene = Scenes::createLitTestScene(configs.width, configs.height);
        break;

    case 1:
        sceneName = "Orbit Scene";
        scene = Scenes::createOrbitScene(configs.width, configs.height);
        break;

    case 2:
        sceneName = "Cornell Box";
        scene = DemoScenes::createCornellBox(configs.width, configs.height);
        break;

    case 3:
        sceneName = "Material Showcase";
        scene =
            DemoScenes::createMaterialShowcase1(configs.width, configs.height);
        break;

    case 4:
        sceneName = "Light Show";
        scene = DemoScenes::createLightShow(configs.width, configs.height);
        break;

    case 5:
        sceneName = "Architectural";
        scene = DemoScenes::createArchitectural(configs.width, configs.height);
        break;

    case 6:
        sceneName = "Material Showcase new";
        scene =
            DemoScenes::createMaterialShowcase(configs.width, configs.height);
        break;
    case 7: {
        sceneName = "Custom Scene";
        scene = std::make_unique<Scene>(configs.width, configs.height);

        Mesh *fancyguy = scene->addMesh("models/subhumanchoppedahhdude.obj",
                                        Materials::MarbleCarrara());
        fancyguy->scale(0.01f);
        fancyguy->moveTo(vec3(0, 3.5, 5));
        fancyguy->rotateSelfEulerXYZ(vec3(-PI_OVER_TWO, 0.0f, 0.0f));
        fancyguy->rotateSelfEulerXYZ(vec3(0.0f, PI, 0.0f));

        Mesh *gem1 = scene->addMesh("models/ugly.obj", Materials::Glass());
        gem1->scale(10.5f);
        gem1->moveTo(vec3(-3, 0, -10));

        Mesh *gem2 =
            scene->addMesh("models/halfway.obj", Materials::MarbleNero());
        gem2->scale(10.5f);
        gem2->moveTo(vec3(0, 0, -10));

        Mesh *gem3 =
            scene->addMesh("models/full.obj", Materials::MarbleVerde());
        gem3->scale(10.5f);
        gem3->moveTo(vec3(3, 0, -10));

        // Dramatic lighting
        scene->addSpotLight(vec3(0, 4, 2), vec3(0, -1, -0.5f),
                            vec3(1.0f, 1.0f, 1.0f), 5.0f, 0.1f, 0.3f, 1.75f);
        scene->addPointLight(vec3(0, 4.5, 2), vec3(0.5f, 0.5f, 1.0f), 1.0f,
                             1.0f);

        scene->setAmbientLight(vec3(0.08f));

        scene->setCamera(vec3(0, 3, 0), vec3(0, 3.5, 5), vec3(0, 1, 0), 60.0f);
        Material floorMat(vec3(0.8f));
        floorMat.specular = vec3(0.1f);

        scene->addPlaneXZ(/*planeY=*/-3.0f, /*halfSize=*/50.0f, floorMat);

        break;
    }

    default:
        std::cerr << "Invalid scene ID. Using default.\n";
        sceneName = "Lit Test Scene";
        scene = Scenes::createLitTestScene(configs.width, configs.height);
        break;
    }

    scene->setBVHLeafTarget(configs.bvhLeafTarget, configs.bvhLeafTol);

    return {std::move(scene), sceneName};
}

int main(int argc, char *argv[]) {

    try {

        RenderConfig config = parseArguments(argc, argv);
        config.bvhLeafTarget = 4;
        config.bvhLeafTol = 8;

        config.width = 1920;
        config.height = 1080;

        if (config.showHelp) {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        }

        auto result = buildSceneById(config);
        std::unique_ptr<Scene> scene = std::move(result.first);
        std::string sceneName = result.second;

        // Print render information
        printRenderInfo(sceneName, config.width, config.height);

        // Upload scene to GPU
        std::cout << "Uploading scene to GPU...\n";
        scene->uploadToGPU();

        // Create window + CUDA-GL interop viewer
        rtgl::InteropViewer V{};
        rtgl::init_interop_viewer(V, config.width, config.height,
                                  "Ray Tracer (CUDA-GL Interop)",
                                  /*cudaDevice=*/0);

        // If headlessOnce: render once to device and exit (no window loop)
        if (config.headlessOnce) {
            size_t nbytes = 0;
            uint8_t *dptr = rtgl::map_pbo_device_ptr(V, &nbytes);
            const size_t want =
                static_cast<size_t>(config.width) * config.height * 3;
            if (nbytes < want) {
                std::cerr << "PBO too small: have " << nbytes << " need "
                          << want << std::endl;
            } else {
                scene->render_to_device(dptr);
            }
            rtgl::unmap_pbo(V);
            rtgl::blit_pbo_to_texture(V);
            rtgl::draw_interop(V);
            rtgl::destroy_interop_viewer(V);
            return EXIT_SUCCESS;
        }

        while (!glfwWindowShouldClose(V.window)) {
            CameraController camCtl;
            camCtl.initFromScene(*scene, config.width, config.height);
            glfwSetInputMode(V.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

            int currentSceneId = config.sceneId;
            bool mouseWasDown = false;
            double prevTime = glfwGetTime();

            while (!glfwWindowShouldClose(V.window)) {
                double now = glfwGetTime();
                float dt = float(now - prevTime);
                prevTime = now;

                // 1) Update camera from input (WASD + mouse)
                camCtl.update(*scene, V.window, dt);

                // 2) Scene switching (keys or “button”)
                int wantScene = currentSceneId;

                // Number keys 0..6 jump directly
                for (int n = 0; n <= 7; ++n)
                    if (glfwGetKey(V.window, GLFW_KEY_0 + n) == GLFW_PRESS)
                        wantScene = n;

                // Press N or ] to go to next scene (edge-triggered)
                bool nextNow = (glfwGetKey(V.window, GLFW_KEY_RIGHT_BRACKET) ==
                                GLFW_PRESS) ||
                               (glfwGetKey(V.window, GLFW_KEY_N) == GLFW_PRESS);
                static bool nextPrev = false;
                if (nextNow && !nextPrev)
                    wantScene = (currentSceneId + 1) % 7;
                nextPrev = nextNow;

                // Click “button” in top-right rectangle to go next scene
                double mx, my;
                glfwGetCursorPos(V.window, &mx, &my);
                bool mouseDown =
                    glfwGetMouseButton(V.window, GLFW_MOUSE_BUTTON_LEFT) ==
                    GLFW_PRESS;
                const int buttonW = 150, buttonH = 36; // invisible hitbox
                if (mouseDown && !mouseWasDown) {
                    if (mx >= config.width - buttonW && my <= buttonH) {
                        wantScene = (currentSceneId + 1) % 7;
                    }
                }
                mouseWasDown = mouseDown;

                if (wantScene != currentSceneId) {
                    config.sceneId = wantScene;
                    auto [newScene, name] = buildSceneById(config);
                    scene = std::move(newScene);
                    scene->uploadToGPU();
                    currentSceneId = wantScene;
                    camCtl.initFromScene(*scene, config.width, config.height);

                    std::ostringstream tt;
                    tt << name << " — press N or click top-right to switch";
                    glfwSetWindowTitle(V.window, tt.str().c_str());
                }

                // 3) Render
                auto start = std::chrono::high_resolution_clock::now();
                size_t nbytes = 0;
                uint8_t *dptr = rtgl::map_pbo_device_ptr(V, &nbytes);
                const size_t want =
                    static_cast<size_t>(config.width) * config.height * 3;
                if (nbytes < want) {
                    std::cerr << "PBO too small: have " << nbytes << " need "
                              << want << std::endl;
                } else {
                    scene->render_to_device(dptr);
                }
                rtgl::unmap_pbo(V);

                rtgl::blit_pbo_to_texture(V);
                rtgl::draw_interop(V);

                if (glfwGetKey(V.window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                    glfwSetWindowShouldClose(V.window, 1);
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              end - start)
                              .count();
                std::ostringstream os;
                os << "Scene " << currentSceneId
                   << " | WASD+mouse | N/Click to switch | " << ms << " ms";
                glfwSetWindowTitle(V.window, os.str().c_str());
            }
        }

        rtgl::destroy_interop_viewer(V);
        return EXIT_SUCCESS;
    } catch (const std::exception &e) {
        std::cerr << "\nError: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}