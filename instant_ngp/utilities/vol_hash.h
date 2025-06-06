//
// Created by khkim on 6/6/25.
//

#pragma once
#include <cstdint>
#include <vector>
#include "../externals/half-2.2.1/include/half.hpp"

typedef half_float::half float16;

typedef struct {
    uint8_t levels;
    uint8_t num_feat_per_entry;
    uint8_t coarsest_res;
    uint8_t spatial_dim;
    uint32_t finest_res;
    uint32_t max_num_entries;
} hash_params;

typedef struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} vec3u32;

typedef struct vec3f{
    float x;
    float y;
    float z;

    vec3f operator*(const float k) const {
        return vec3f{x * k, y * k, z * k};
    }

    vec3f operator-(const vec3u32& p) const {
        return vec3f{x - p.x, y - p.y, z - p.z};
    }

    void operator*=(const float k) {
        x *= k;
        y *= k;
        z *= k;
    }
} vec3f;

/*
Taken from Instant-NGP:
encoding parameters Theta. These are arranged into 𝐿 levels, each containing up to𝑇 feature vectors with dim 𝐹.
Typical values for these hyperparameters are shown in Table 1. Figure 3 illustrates
the steps performed in our multiresolution hash encoding.
Each level (two of which are shown as red and blue in the figure) is independent and conceptually stores
feature vectors at the vertices of a grid, the resolution of which is chosen to be a geometric progression
between the coarsest and finest resolutions
*/
class VolHash {
public:
    std::vector<float> pos_features_at(const vec3f& pos);

private:
    /* Helper Functions */
    std::vector<float16> level_hash(const vec3f& pos, uint8_t level);

    void init_b();
    void init_level(uint8_t level);

    vec3f scaled_by_level_res(const vec3f& pos, uint8_t level);
    vec3u32 level_scaled_rounded_up(const vec3f& pos, uint8_t level);
    vec3u32 level_scaled_rounded_down(const vec3f& pos, uint8_t level);

    hash_params params;
    float b;
    std::vector<std::vector<float16>> levels_features;
    std::vector<uint32_t> levels_res;
};
