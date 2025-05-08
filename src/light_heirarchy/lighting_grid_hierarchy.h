#ifndef LIGHTING_GRID_HIERARCHY_H
#define LIGHTING_GRID_HIERARCHY_H

#include "kdtree3d.h"
#include "pbrt/pbrt.h"
#include "pbrt/util/transform.h"
#include <nanovdb/NanoVDB.h>
#include <pbrt/samplers.h>
#include <array>
#include <fstream>
#include <sstream>
#include <iomanip>

class LGH
{
public:
    // TODO: separate into constructor and initialization function
    // TODO: fix inputs
    // LGH(pbrt::SampledGrid<float> temperature_grid, int depth, float base_voxel_size, float transmission);
    LGH(const nanovdb::FloatGrid* temperature_grid, const nanovdb::FloatGrid* density_grid, int depth, float base_voxel_size, float transmission, pbrt::Transform transform);

    pbrt::SampledSpectrum get_intensity(int L,
    LGH(const nanovdb::FloatGrid* temperature_grid, int depth, float base_voxel_size, float transmission, pbrt::Transform transform);
    

void save_cube_map_face_as_pgm(const std::vector<std::vector<float>>& face, const std::string& filename) {
    int res = face.size();
    std::ofstream ofs(filename, std::ios::binary);
    ofs << "P5\n" << res << " " << res << "\n255\n";
    // Find min/max for normalization
    float min_val = face[0][0], max_val = face[0][0];
    for (const auto& row : face)
        for (float v : row) {
            min_val = std::min(min_val, v);
            max_val = std::max(max_val, v);
        }
    float scale = (max_val > min_val) ? 255.0f / (max_val - min_val) : 1.0f;
    for (const auto& row : face)
        for (float v : row) {
            unsigned char pixel = static_cast<unsigned char>(std::clamp((v - min_val) * scale, 0.0f, 255.0f));
            ofs.write(reinterpret_cast<char*>(&pixel), 1);
        }
    ofs.close();
}

    // pbrt::SampledSpectrum
    float get_intensity(int L,
                                        Vector3f targetPos,
                                        KDNode* light,
                                        float radius,
                                        pbrt::SampledWavelengths lambda,
                                        pbrt::Sampler sampler,
                                        pbrt::Medium medium);

    pbrt::SampledSpectrum get_total_illum(pbrt::Point3f pos,
                                          pbrt::SampledWavelengths lambda,
                                          pbrt::Sampler sampler,
                                          pbrt::Medium medium,
                                          pbrt::RNG rng,
                                          float tMax,
                                          pbrt::Ray ray);

    const float TEMP_THRESHOLD = 1.0f;

    const pbrt::Transform medium_transform;


    // static void extract_lights(pbrt::SampledGrid<float> temperature_grid);

    const float alpha = 1.0f;
    const int l_max;

    // TODO: remove this to use pbrt transmission instead
    const float transmission;
    const nanovdb::FloatGrid* m_temperature_grid;
    const nanovdb::FloatGrid* m_density_grid;

    // Compute cube map for a light at a given level
    void compute_cube_map_for_light(int level, int light_idx, const Vector3f& light_pos, float r_e, float h, const nanovdb::FloatGrid* density_grid);
    // Lookup shadow from cube map
    float lookup_shadow(int level, int light_idx, const Vector3f& light_pos, const Vector3f& target_pos) const;

    // Save cube map visualization functions
    void save_cube_map_face_as_ppm(const std::vector<std::vector<float>>& face_data, const std::string& filename);
    void save_all_cube_maps();

    // Display cube map visualization functions
    void display_cube_map_face(const std::vector<std::vector<float>>& face_data, int face_idx);
    void display_all_cube_maps();

    // // Save cube map as a cross-shaped visualization
    // void save_cube_map_cross(const CubeMap& cmap, const std::string& filename) {
    //     int res = cmap.resolution;
    //     // Create a cross-shaped image (3x4 faces)
    //     std::vector<std::vector<float>> cross(res * 3, std::vector<float>(res * 4, 0.0f));
        
    //     // Layout:
    //     //    +Y
    //     // -X +Z +X
    //     //    -Y
    //     //    -Z
        
    //     // Copy each face to its position in the cross
    //     for (int i = 0; i < res; ++i) {
    //         for (int j = 0; j < res; ++j) {
    //             // +Y face (top)
    //             cross[i][j + res] = cmap.faces[2][i][j];
    //             // -X face (left)
    //             cross[i + res][j] = cmap.faces[1][i][j];
    //             // +Z face (center)
    //             cross[i + res][j + res] = cmap.faces[4][i][j];
    //             // +X face (right)
    //             cross[i + res][j + 2*res] = cmap.faces[0][i][j];
    //             // -Y face (bottom)
    //             cross[i + 2*res][j + res] = cmap.faces[3][i][j];
    //             // -Z face (bottom)
    //             cross[i + 2*res][j + 3*res] = cmap.faces[5][i][j];
    //         }
    //     }
        
    //     // Save as PGM
    //     std::ofstream ofs(filename, std::ios::binary);
    //     ofs << "P5\n" << res * 4 << " " << res * 3 << "\n255\n";
        
    //     // Find min/max for normalization
    //     float min_val = cross[0][0], max_val = cross[0][0];
    //     for (const auto& row : cross)
    //         for (float v : row) {
    //             min_val = std::min(min_val, v);
    //             max_val = std::max(max_val, v);
    //         }
    //     float scale = (max_val > min_val) ? 255.0f / (max_val - min_val) : 1.0f;
        
    //     for (const auto& row : cross)
    //         for (float v : row) {
    //             unsigned char pixel = static_cast<unsigned char>(std::clamp((v - min_val) * scale, 0.0f, 255.0f));
    //             ofs.write(reinterpret_cast<char*>(&pixel), 1);
    //         }
    //     ofs.close();
    // }

private:
    void create_S0(const nanovdb::FloatGrid* temperature_grid);
    void deriveNewS(int l);//, KDTree S0);
    Vector3f calcNewPos(int l, Vector3f target_light_pos, std::vector<KDNode*> j_lights);//const Vector3f& gv, int l, const KDTree& S0) const;
    float calcNewI(int l, Vector3f target_light_pos, std::vector<KDNode*> j_lights); //const Vector3f& gv, int l, const KDTree& S0) const;

    float blendingFunction(int level, float distance, float r_l);

    std::vector<KDTree*> lighting_grids;
    std::vector<float> h;

    Vector3f BBoxMin;
    Vector3f BBoxMax;

    // --- Shadow map support ---
    // For each level, store a vector of filtered densities (one per grid vertex)
    std::vector<std::vector<float>> filtered_densities;
    // For each level, store the grid vertex positions (to match filtered_densities)
    std::vector<std::vector<Vector3f>> grid_vertices;

    // Pre-filter the density field for a given level
    void prefilter_density_field(int level, float h_l, const nanovdb::FloatGrid* density_grid);

    // Cube map structure: 6 faces, each face is a 2D array of floats
    struct CubeMap {
        std::array<std::vector<std::vector<float>>, 6> faces; // [face][u][v]
        int resolution; // texels per face edge
        float r_e;      // effective radius for this level
        float h;        // voxel size for this level
    };
    // For each level, for each light, store a cube map
    std::vector<std::vector<CubeMap>> shadow_cube_maps;

    // Filtered density at a point for a given level (trilinear interpolation)
    float filtered_density_at(int level, const Vector3f& pos) const;
    // Filtered density at a point with filter size delta (pyramidal interpolation)
    float filtered_density_with_filter(const Vector3f& pos, float delta) const;
};

#endif // LIGHTING_GRID_HIERARCHY_H
