//
// Created by khkim on 6/6/25.
//

#include "vol_hash.h"
#include <cassert>
#include <cmath>
#include <algorithm>

std::vector<float>
VolHash::pos_features_at(const vec3f& pos){
    std::vector<float> vol_features(params.levels * params.num_feat_per_entry);
    std::vector<float16> level_features(params.num_feat_per_entry);



    assert(vol_features.size() == params.levels * params.num_feat_per_entry);
    return vol_features;
}

std::vector<float16>
VolHash::level_hash(const vec3f& pos, uint8_t level) {
    std::vector<float16> level_features(params.num_feat_per_entry);

    /* naive mapping from 3D volume to 1D array
     * could be upgraded using morton ordering
     */
    if (std::pow((levels_res[level] + 1.f),3) < params.max_num_entries):
        return level_features;

    else:
    return std::vector<float16>();
}

void
VolHash::init_b() {
    assert(params.levels > 1);
    b = (log(params.finest_res) - log(params.coarsest_res)) / (params.levels - 1);
}

void
VolHash::init_level(uint8_t level) {
    assert(b > 0);
    assert(levels_features.size() > level);
    assert(levels_res.size() > level);

    float base_res_vertices = std::floor(params.coarsest_res * std::pow(b, level)) + 1.0f;
    levels_res[level] = static_cast<uint32_t>(base_res_vertices - 1);
    levels_features[level].resize(
        std::min(
            static_cast<uint32_t>(std::pow(base_res_vertices, params.spatial_dim)),
            params.max_num_entries)
        );
}

vec3f
VolHash::scaled_by_level_res(const vec3f& pos, uint8_t level) {
    return pos * levels_res[level];
}

vec3u32
VolHash::level_scaled_rounded_up(const vec3f& pos, uint8_t level) {
    vec3f scaled_pos = scaled_by_level_res(pos, level);
    return vec3u32{
        static_cast<uint32_t>(std::ceil(scaled_pos.x)),
        static_cast<uint32_t>(std::ceil(scaled_pos.y)),
        static_cast<uint32_t>(std::ceil(scaled_pos.z))
    };
}

vec3u32
VolHash::level_scaled_rounded_down(const vec3f& pos, uint8_t level) {
    vec3f scaled_pos = scaled_by_level_res(pos, level);
    return {
        static_cast<uint32_t>(std::floor(scaled_pos.x)),
        static_cast<uint32_t>(std::floor(scaled_pos.y)),
        static_cast<uint32_t>(std::floor(scaled_pos.z))
    };
}
