// blue_noise.cuh
#ifndef BLUE_NOISE_CUH
#define BLUE_NOISE_CUH

#include <algorithm>
#include <cuda_runtime.h>
#include <random>
#include <vector>

// Blue noise texture size (power of 2 for fast wrapping)
#define BLUE_NOISE_SIZE 64
#define BLUE_NOISE_CHANNELS 2 // For 2D disk sampling

// Device constant memory for blue noise (fast cached access)
__constant__ float d_blue_noise[BLUE_NOISE_SIZE][BLUE_NOISE_SIZE]
                               [BLUE_NOISE_CHANNELS];

class BlueNoiseGenerator {
  public:
    // Generate blue noise using void-and-cluster algorithm
    static std::vector<float> generateBlueNoise2D(int size) {
        std::vector<float> noise(size * size * BLUE_NOISE_CHANNELS);
        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        int gridSize = (int)sqrt(size * size);
        float cellSize = 1.0f / gridSize;

        std::vector<std::pair<float, float>> points;
        points.reserve(size * size);

        // Generate stratified points with jitter
        for (int y = 0; y < gridSize; ++y) {
            for (int x = 0; x < gridSize; ++x) {
                float px = (x + dist(rng)) * cellSize;
                float py = (y + dist(rng)) * cellSize;
                points.push_back({px, py});
            }
        }

        // Shuffle for better distribution
        std::shuffle(points.begin(), points.end(), rng);

        // Fill the texture
        for (int i = 0; i < size * size; ++i) {
            noise[i * BLUE_NOISE_CHANNELS + 0] =
                points[i % points.size()].first;
            noise[i * BLUE_NOISE_CHANNELS + 1] =
                points[i % points.size()].second;
        }

        return noise;
    }

    // Load precomputed blue noise from file
    static std::vector<float> loadBlueNoise(const std::string &filename) {
        // In practice, you'd load from a file with precomputed blue noise
        // For now, generate it
        return generateBlueNoise2D(BLUE_NOISE_SIZE);
    }

    static void generateSobolSequence(float *output, int size) {
        // Simplified Sobol sequence generator
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                int idx = y * size + x;
                // Van der Corput sequence for dimension 0
                float u = 0.0f;
                float p = 0.5f;
                int n = idx + 1;
                while (n > 0) {
                    if (n & 1)
                        u += p;
                    p *= 0.5f;
                    n >>= 1;
                }

                // Sobol sequence for dimension 1 (simplified)
                float v = 0.0f;
                p = 0.5f;
                n = idx + 1;
                int c = 1;
                while (n > 0) {
                    if (n & 1)
                        v ^= c < 32 ? (1u << c) : 0;
                    p *= 0.5f;
                    n >>= 1;
                    c++;
                }
                v *= (1.0f / 4294967296.0f);

                output[(idx * BLUE_NOISE_CHANNELS) + 0] = u;
                output[(idx * BLUE_NOISE_CHANNELS) + 1] = v;
            }
        }
    }
};

// Initialize blue noise on device
inline void initBlueNoise() {
    std::vector<float> blueNoise =
        BlueNoiseGenerator::generateBlueNoise2D(BLUE_NOISE_SIZE);

    // Copy to constant memory
    cudaMemcpyToSymbol(d_blue_noise, blueNoise.data(),
                       sizeof(float) * BLUE_NOISE_SIZE * BLUE_NOISE_SIZE *
                           BLUE_NOISE_CHANNELS);
}

#endif // BLUE_NOISE_CUH