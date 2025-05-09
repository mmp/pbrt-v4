#include "lighting_grid_hierarchy.h"
#include "nanovdb/util/SampleFromVoxels.h"
#include "pbrt/util/spectrum.h"
#include <stdio.h>
#include <pbrt/media.h>
#include <array>
#include <cmath>
#include <sstream>
#include <algorithm>  // for std::min, std::max

// LGH::LGH(pbrt::SampledGrid<float> temperature_grid, int depth, float base_voxel_size, float transmission) 
LGH::LGH(const nanovdb::FloatGrid* temperature_grid, int depth, float base_voxel_size, float transmission, pbrt::Transform transform)
    : l_max(depth), transmission(transmission), medium_transform(transform)
{
    auto worldBBox = temperature_grid->worldBBox();
    nanovdb::Vec3d minBBox = worldBBox.min();
    BBoxMin = Vector3f(minBBox[0], minBBox[1], minBBox[2]);

    nanovdb::Vec3d maxBBox = worldBBox.max();
    BBoxMax = Vector3f(maxBBox[0], maxBBox[1], maxBBox[2]);

    printf("Min Bounds: %f %f %f, Max Bounds: %f %f %f", BBoxMin.x, BBoxMin.y, BBoxMin.z, BBoxMax.x, BBoxMax.y, BBoxMax.z);

    this->m_temperature_grid = temperature_grid;

    // Initialize h
    for (int i=0; i<=depth; i++) {
        h.push_back(base_voxel_size * pow(2,i));
    }

    printf("\n============= H ==============\n");
    for (float h_val : h) {
        printf("%f ", h_val);
    }
    printf("\n\n");

    if (h.size() != l_max + 1) {
        LOG_FATAL("invalid number of h!!!");
    }

    lighting_grids = std::vector<KDTree*>(depth + 1);

    // TODO: should I pass in the accessor? Do I need the grid?
    // Initialize S0
    create_S0(temperature_grid);
    prefilter_density_field(0, h[0], temperature_grid);
    // Create rest of S_l
    for (int j=1; j<=depth; j++) {
        int i=j-1;
        deriveNewS(j);
        prefilter_density_field(j, h[j], temperature_grid);
        // Compute cube maps for each light in this level
        auto& grid = *lighting_grids[i];
        std::vector<KDNode*> nodes;
        // Collect all nodes (lights) in this grid
        //grid.radiusSearch(Vector3f(0,0,0), 1e6, nodes); // large radius to get all
        grid.getAllNodes(nodes, 500);
        printf("Number of lights in S%d: %lu\n", i, nodes.size());
        for (size_t j = 0; j < nodes.size(); ++j) {
            compute_cube_map_for_light(i, j, nodes[j]->point, h[i], h[i], temperature_grid);
        }
    }

    if (lighting_grids.size() != l_max + 1) {
        LOG_FATAL("Invalid number of grids");
    }
    //display_all_cube_maps();
}


void LGH::create_S0(const nanovdb::FloatGrid* temperature_grid)
{
    printf("\n===============CREATING S0===========\n");
    float h_0 = this->h[0];

    std::vector<std::pair<Vector3f,float>> lights;

    // x,y,z should be in world space
    for (float x = BBoxMin.x + h_0/2; x < BBoxMax.x; x += h_0) {
        for (float y = BBoxMin.y + h_0/2; y < BBoxMax.y; y += h_0) {
            for (float z = BBoxMin.z + h_0/2; z < BBoxMax.z; z += h_0) {
                nanovdb::Vec3f pIndex = temperature_grid->worldToIndexF(nanovdb::Vec3f(x, y, z));
                using Sampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
                float temperature = Sampler(temperature_grid->tree())(pIndex);

                if (temperature > 0) {
                    lights.emplace_back(Vector3f(x,y,z), temperature);
                }

                if (temperature < 0) {
                    printf("Temperature negative: %f , point %f %f %f", temperature, x, y, z);
                }
            }
        }
    }
    
    // Assign indices to lights
    for (size_t i = 0; i < lights.size(); ++i) {
        lights[i].second = i;  // Store index in the second component
    }

    this->lighting_grids[0] = new KDTree(lights);

    printf("=============================\nCreated %lu lights in S0\n=============================", lights.size());
}

void LGH::deriveNewS(int l)
{
    if (l == 0) {
        LOG_FATAL("Level should never be 0!");
        return;
    }
    float h_l = this->h[l];                     // spacing of level l
    std::vector<std::pair<Vector3f,float>> lights;

    // iterate over grid vertices of level l
    for (float x = BBoxMin.x; x < BBoxMax.x; x += h_l)
    for (float y = BBoxMin.y; y < BBoxMax.y; y += h_l)
    for (float z = BBoxMin.z; z < BBoxMax.z; z += h_l)
    {
        Vector3f target_light_pos(x,y,z);             // grid-vertex position q_i
    
        KDTree* lighting_grid_j = lighting_grids[l-1];
    
        float radius = sqrt(3) * h[l-1];
        std::vector<KDNode*> j_lights;
        lighting_grid_j->radiusSearch(target_light_pos, radius, j_lights);

        float     I  = calcNewI (l, target_light_pos, j_lights);          // Eq (1)
        Vector3f  p  = calcNewPos(l, target_light_pos, j_lights);         // Eq (2)
        if (I>0.f) lights.emplace_back(p,I);
    }

    // Assign indices to lights
    for (size_t i = 0; i < lights.size(); ++i) {
        lights[i].second = i;  // Store index in the second component
    }

    lighting_grids[l] = new KDTree(lights);
    printf("=============================\nCreated %lu lights in S%d\n=============================", lights.size(), l);
}

float calcTrilinearWeight(Vector3f p, Vector3f q, float h_l) {
    Vector3f v = Vector3f(1,1,1) - (p - q).abs() / h_l;
    float v_product = v.x * v.y * v.z;
    return std::max(0.f, std::min(1.f, v_product));
}

// -------- intensity (Eq 1) ----------------------------------------------
float LGH::calcNewI(int l, Vector3f target_light_pos, std::vector<KDNode*> j_lights) //const Vector3f& gv, int l, const KDTree& S0) const
{
    float I_sum;
    for (auto j_light : j_lights) {
        float w = calcTrilinearWeight(j_light->point, target_light_pos, this->h[l]);
        I_sum += w * j_light->intensity;
    }

    return I_sum;
}

// -------- illumination centre (Eq 2) ------------------------------------
Vector3f LGH::calcNewPos(int l, Vector3f target_light_pos, std::vector<KDNode*> j_lights) //const Vector3f& gv, int l, const KDTree& S0) const
{
    Vector3f p_num;
    float p_denom;

    for (auto j_light : j_lights) {
        float w = calcTrilinearWeight(j_light->point, target_light_pos, this->h[l]);
        float v = w * j_light->intensity; // TODO: what is the luminance component? What exactly is intensity data structure?
        p_num += v * j_light->point;
        p_denom = v;
    }

    return p_num/p_denom;
}

float LGH::blendingFunction(int level, float d, float r_l)
{
    
    if (level == 0) {
        if (d <= r_l) return 1;
        else if (d <= 2 * r_l) return 2 - d/r_l;
        else return 0;
    }

    if (level == l_max) {
        if (d <= r_l/2) return 0;
        else if (d <= r_l) return 2*d/r_l - 1;
        else return 1;
    }

    if (d <= r_l/2) return 0;
    else if (d <= r_l) return 2*d/r_l - 1;
    else if (d <= 2* r_l) return 2 - d/r_l;
    return 0;
}

// pbrt::SampledSpectrum
float LGH::get_intensity(int L,
                                         Vector3f targetPos,
                                         KDNode* light,
                                         float radius,
                                         pbrt::SampledWavelengths lambda,
                                         pbrt::Sampler sampler,
                                         pbrt::Medium medium)
{
    // Get shadow value from cube map // Add idx member to KDNode if not present
    float shadow = lookup_shadow(L, light->idx, light->point, targetPos);
    

    Vector3f dir = light->point - targetPos;

    pbrt::Ray ray = pbrt::Ray(pbrt::Point3f(targetPos.x, targetPos.y, targetPos.z), pbrt::Vector3f(dir.x, dir.y, dir.z), 0, medium);
    float tMax = Length(ray.d);
    // Initialize _RNG_ for sampling the majorant transmittance
    uint64_t hash0 = pbrt::Hash(sampler.Get1D());
    uint64_t hash1 = pbrt::Hash(sampler.Get1D());
    pbrt::RNG rng(hash0, hash1);

    pbrt::SampledSpectrum V = SampleT_maj(ray, tMax, sampler.Get1D(), rng, lambda,
                [&](pbrt::Point3f p, pbrt::MediumProperties mp, pbrt::SampledSpectrum sigma_maj, pbrt::SampledSpectrum T_maj) {
        return true;
                });//.y(lambda);

    V = pbrt::Clamp(V, 0.f, 1.f);


    float d = targetPos.distance(light->point);

    // TODO: figure out good scale for light fall-off, make sure it is relative to size of explosion
    float g = 1.f / pow(d,2); // Light fall-off, good with d * 50
    g = std::min(1.f, g);

    float B = blendingFunction(L, d, radius);

    if (B < 0) {
        LOG_FATAL("Blending function should never be negative! %f", B);
    }

    if (B > 1) {
        printf("B large: %f\n", B);
    }

    if (light->intensity < 0) {
        printf("LIGHT INTENSITY NEGATIVE!!! L: %d, %f\n", L, light->intensity);
    }

    // Convert spectral values to scalar
    float spectral_intensity = pbrt::BlackbodySpectrum(light->intensity * 1000).Sample(lambda).Average();
    
    // Apply shadow to intensity
    return g * B * spectral_intensity * (1.0f - shadow);
}

pbrt::SampledSpectrum LGH::get_total_illum(pbrt::Point3f pos,
                                           pbrt::SampledWavelengths lambda,
                                           pbrt::Sampler sampler,
                                           pbrt::Medium medium)
{

    // Note that pos passed in callback is in medium local space! Convert to world-space to access lights
    pos = medium_transform.ApplyInverse(pos);


    // nanovdb::Vec3f pIndex = m_temperature_grid->worldToIndexF(nanovdb::Vec3f(p.x, p.y, p.z));
    // TODO: actually do weighted sample for more accurate temperature
    // Vector3f v_pos(pIndex[0], pIndex[1], pIndex[2]);

    Vector3f v_pos(pos.x, pos.y, pos.z);

    int numLightsCaptured = 0;
    int numLightsS0 = 0, numLightsS1 = 0;
    float total_intensity = 0.0f;

    for (int l=0; l<=l_max; l++) {
        float radius = alpha * h[l];

        std::vector<KDNode*> results;
        lighting_grids[l]->radiusSearch(v_pos, radius, results);
        numLightsCaptured += results.size();

        // TODO: for some reason, levels beyond 0 do nothing
        if (l == 0)
            numLightsS0 = results.size();
        if (l == 1)
            numLightsS1 = results.size();
        
        for (auto light : results) {
            float intensity = get_intensity(l, v_pos, light, radius, lambda, sampler, medium);
            if (std::isfinite(intensity)) {  // Check for NaN and infinity
                total_intensity += intensity;
            }
        }
    }

    // Check for invalid total intensity
    if (!std::isfinite(total_intensity) || total_intensity < 0.0f) {
        return pbrt::SampledSpectrum(0.0f);
    }

    // Scale the intensity to a reasonable range for blackbody spectrum
    float scaled_intensity = total_intensity * 1000.0f;  // Scale up for blackbody
    if (scaled_intensity < 100.0f) {  // Minimum threshold for blackbody
        return pbrt::SampledSpectrum(0.0f);
    }

    return 0.125f * pbrt::BlackbodySpectrum(scaled_intensity).Sample(lambda);
}

// Pyramid filter function (from reference image, Eq. 6)
static float pyramid_filter(int i, int j, int k, int h_l) {
    // i, j, k in {-1, 0, 1}
    // h_l is the filter size (should be 3 for 3x3x3)
    return (1.0f / 8.0f) * (1.0f - std::abs(i) / float(h_l)) * (1.0f - std::abs(j) / float(h_l)) * (1.0f - std::abs(k) / float(h_l));
}

void LGH::prefilter_density_field(int level, float h_l, const nanovdb::FloatGrid* density_grid) {
    std::vector<float> densities;
    std::vector<Vector3f> vertices;

    // For each grid vertex at this level
    for (float x = BBoxMin.x; x < BBoxMax.x; x += h_l)
    for (float y = BBoxMin.y; y < BBoxMax.y; y += h_l)
    for (float z = BBoxMin.z; z < BBoxMax.z; z += h_l) {
        Vector3f v(x, y, z);
        float filtered_density = 0.0f;
        // 3x3x3 stencil centered at v
        for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz) {
            float nx = x + dx * h_l;
            float ny = y + dy * h_l;
            float nz = z + dz * h_l;
            // Check bounds
            if (nx < BBoxMin.x || nx >= BBoxMax.x ||
                ny < BBoxMin.y || ny >= BBoxMax.y ||
                nz < BBoxMin.z || nz >= BBoxMax.z)
                continue;
            // Convert to grid index
            nanovdb::Vec3f pIndex = density_grid->worldToIndexF(nanovdb::Vec3f(nx, ny, nz));
            using Sampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
            float density = Sampler(density_grid->tree())(pIndex);
            float w = pyramid_filter(dx, dy, dz, 2); // h_l=2 for 3x3x3
            filtered_density += w * density;
        }
        densities.push_back(filtered_density);
        vertices.push_back(v);
    }
    // Store for this level
    if (filtered_densities.size() <= level) filtered_densities.resize(level+1);
    if (grid_vertices.size() <= level) grid_vertices.resize(level+1);
    filtered_densities[level] = std::move(densities);
    grid_vertices[level] = std::move(vertices);
}

// Helper: directions for cube map faces
static const Vector3f cube_face_dirs[6][3] = {
    { {1,0,0}, {0,1,0}, {0,0,1} },   // +X
    { {-1,0,0}, {0,1,0}, {0,0,-1} }, // -X
    { {0,1,0}, {1,0,0}, {0,0,1} },   // +Y
    { {0,-1,0}, {1,0,0}, {0,0,-1} }, // -Y
    { {0,0,1}, {1,0,0}, {0,1,0} },   // +Z
    { {0,0,-1}, {1,0,0}, {0,-1,0} }  // -Z
};

// Pyramidal filter function (Eq. 6)
static float lambda_pyramid(int i, int j, int k, float h) {
    return (1.0f / 8.0f) * (1.0f - std::abs(i) / h) * (1.0f - std::abs(j) / h) * (1.0f - std::abs(k) / h);
}

// Trilinear interpolation for filtered density field
float LGH::filtered_density_at(int level, const Vector3f& pos) const {
    // Find the 8 grid vertices surrounding pos at this level
    float h_l = h[level];
    Vector3f grid_pos = (pos - BBoxMin) / h_l;
    int x0 = static_cast<int>(std::floor(grid_pos.x));
    int y0 = static_cast<int>(std::floor(grid_pos.y));
    int z0 = static_cast<int>(std::floor(grid_pos.z));
    float fx = grid_pos.x - x0;
    float fy = grid_pos.y - y0;
    float fz = grid_pos.z - z0;
    float result = 0.0f;
    for (int dx = 0; dx <= 1; ++dx)
    for (int dy = 0; dy <= 1; ++dy)
    for (int dz = 0; dz <= 1; ++dz) {
        int xi = x0 + dx;
        int yi = y0 + dy;
        int zi = z0 + dz;
        int idx = (xi * ((int)((BBoxMax.y - BBoxMin.y) / h_l) + 1) + yi) * ((int)((BBoxMax.z - BBoxMin.z) / h_l) + 1) + zi;
        if (xi < 0 || yi < 0 || zi < 0 || xi >= (int)((BBoxMax.x - BBoxMin.x) / h_l) + 1 || yi >= (int)((BBoxMax.y - BBoxMin.y) / h_l) + 1 || zi >= (int)((BBoxMax.z - BBoxMin.z) / h_l) + 1)
            continue;
        float w = ((dx ? fx : 1 - fx) * (dy ? fy : 1 - fy) * (dz ? fz : 1 - fz));
        result += w * filtered_densities[level][idx];
    }
    return result;
}

// Filtered density at x with filter size delta, using Eq. 7
float LGH::filtered_density_with_filter(const Vector3f& pos, float delta) const {
    // Find the two levels such that h_{l-1} <= delta <= h_l
    int l = 0;
    while (l + 1 < (int)h.size() && h[l + 1] < delta) ++l;
    int l0 = l, l1 = std::min(l + 1, (int)h.size() - 1);
    float h0 = h[l0], h1 = h[l1];
    float rho0 = filtered_density_at(l0, pos);
    float rho1 = filtered_density_at(l1, pos);
    if (h1 == h0) return rho0;
    float t = (delta - h0) / (h1 - h0);
    return (1 - t) * rho0 + t * rho1;
}

// Update compute_cube_map_for_light to use the variable filter size and filtered density
void LGH::compute_cube_map_for_light(int level, int light_idx, const Vector3f& light_pos, float r_e, float h0, const nanovdb::FloatGrid* density_grid) {
    int resolution = static_cast<int>(std::ceil(2 * r_e / h0));
    printf("\nCreating cube map for level %d, light %d:\n", level, light_idx);
    printf("  Light position: (%f, %f, %f)\n", light_pos.x, light_pos.y, light_pos.z);
    printf("  Effective radius (r_e): %f\n", r_e);
    printf("  Base voxel size (h0): %f\n", h0);
    printf("  Resolution: %d x %d\n", resolution, resolution);
    
    CubeMap cmap;
    cmap.resolution = resolution;
    cmap.r_e = r_e;
    cmap.h = h0;
    
    // Calculate s_{l,i} based on the level and light properties
    float s_li = h[level]; // Use the grid spacing at this level as the base filter size
    printf("  Filter size s_li: %f\n", s_li);
    
    for (int face = 0; face < 6; ++face) {
        printf("  Computing face %d\n", face);
        cmap.faces[face].resize(resolution, std::vector<float>(resolution, 0.0f));
        float max_shadow = 0.0f;
        float min_shadow = 1e10f;
        
        for (int u = 0; u < resolution; ++u) {
            for (int v = 0; v < resolution; ++v) {
                float s = 2.0f * (u + 0.5f) / resolution - 1.0f;
                float t = 2.0f * (v + 0.5f) / resolution - 1.0f;
                Vector3f x = cube_face_dirs[face][0];
                Vector3f y = cube_face_dirs[face][1];
                Vector3f z = cube_face_dirs[face][2];
                Vector3f dir = (x * s + y * t + z).abs();
                dir = dir / std::sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
                float shadow = 0.0f;
                float step = h0;
                
                // Use variable filter size based on distance from light
                for (float d = 0; d < 2 * r_e; d += step) {
                    Vector3f p = light_pos + dir * d;
                    float delta = 0.0f;
                    if (d <= r_e) {
                        delta = (1 - d / r_e) * s_li + (d / r_e) * h0;
                    } else {
                        // Beyond r_e, use the base filter size
                        delta = h0;
                    }
                    
                    // Get filtered density using the variable filter size
                    float filtered_density = filtered_density_with_filter(p, delta);
                    shadow += filtered_density * step;
                    // shadow = 0.5;
                }
                cmap.faces[face][u][v] = shadow;
                max_shadow = std::max(max_shadow, shadow);
                min_shadow = std::min(min_shadow, shadow);
            }
        }
        printf("    Face %d shadow range: [%f, %f]\n", face, min_shadow, max_shadow);
    }
    
    if (shadow_cube_maps.size() <= level) shadow_cube_maps.resize(level+1);
    if (shadow_cube_maps[level].size() <= light_idx) shadow_cube_maps[level].resize(light_idx+1);
    shadow_cube_maps[level][light_idx] = std::move(cmap);
    
    int total_texels = 6 * resolution * resolution;
    float expected_texels = 24.0f * (r_e / h0) * (r_e / h0);
    printf("Cube map: %d texels (expected: %.1f)\n", total_texels, expected_texels);
    int c=0;

//     for (int face = 0; face < 6; ++face) {
//     if (c>=1) {
//         break;
//     }
//     c++;
//     std::ostringstream oss;
//     oss << "shadowmap_L" << level << "_light" << light_idx << "_face" << face << ".pgm";
//     save_cube_map_face_as_pgm(cmap.faces[face], oss.str());
// }
}

float LGH::lookup_shadow(int level, int light_idx, const Vector3f& light_pos, const Vector3f& target_pos) const {
    // Check if shadow_cube_maps is properly initialized
    if (shadow_cube_maps.empty() || level >= shadow_cube_maps.size()) {
        return 0.0f;
    }
    
    if (light_idx >= shadow_cube_maps[level].size()) {
        return 0.0f;
    }

    // Get direction from light to target
    Vector3f dir = target_pos - light_pos;
    float dist = std::sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
    if (dist < 1e-6f) {
        return 0.0f;
    }
    dir = dir / dist;  // Normalize

    // Find the face with the largest absolute component
    int face = 0;
    float max_comp = std::abs(dir.x);
    if (std::abs(dir.y) > max_comp) {
        max_comp = std::abs(dir.y);
        face = 2;
    }
    if (std::abs(dir.z) > max_comp) {
        max_comp = std::abs(dir.z);
        face = 4;
    }
    // Adjust face index based on sign
    if (dir[face/2] < 0) face++;

    // Get the cube map for this level and light
    const CubeMap& cmap = shadow_cube_maps[level][light_idx];
    if (face >= 6 || face < 0) {
        return 0.0f;
    }
    const auto& face_data = cmap.faces[face];

    // Convert direction to face-local coordinates
    Vector3f x = cube_face_dirs[face][0];
    Vector3f y = cube_face_dirs[face][1];
    Vector3f z = cube_face_dirs[face][2];
    
    // Project direction onto face basis
    float s = (dir.x*x.x + dir.y*x.y + dir.z*x.z) / (dir.x*z.x + dir.y*z.y + dir.z*z.z);
    float t = (dir.x*y.x + dir.y*y.y + dir.z*y.z) / (dir.x*z.x + dir.y*z.y + dir.z*z.z);
    
    // Convert to pixel coordinates
    float u = (s + 1.0f) * 0.5f * (cmap.resolution - 1);
    float v = (t + 1.0f) * 0.5f * (cmap.resolution - 1);
    
    // Clamp coordinates to valid range
    u = std::max(0.0f, std::min(u, static_cast<float>(cmap.resolution - 1)));
    v = std::max(0.0f, std::min(v, static_cast<float>(cmap.resolution - 1)));
    
    // Get the nearest pixel value (no interpolation)
    int u_idx = static_cast<int>(std::round(u));
    int v_idx = static_cast<int>(std::round(v));
    
    // Ensure indices are within bounds
    u_idx = std::min(std::max(0, u_idx), cmap.resolution - 1);
    v_idx = std::min(std::max(0, v_idx), cmap.resolution - 1);
    
    // Return the shadow value from the specific face
    //return 0.5; 
    return face_data[u_idx][v_idx];
}

void LGH::save_cube_map_face_as_ppm(const std::vector<std::vector<float>>& face_data, const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        printf("Error: Could not open file %s for writing\n", filename.c_str());
        return;
    }

    int width = face_data.size();
    int height = face_data[0].size();

    // Write PPM header
    fprintf(fp, "P6\n%d %d\n255\n", width, height);

    // Write pixel data
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float shadow = face_data[x][y];
            // Convert shadow value to color
            // Red channel: shadow value
            // Green channel: 1 - shadow value
            // Blue channel: 0.5
            unsigned char r = static_cast<unsigned char>(shadow * 255);
            unsigned char g = static_cast<unsigned char>((1.0f - shadow) * 255);
            unsigned char b = static_cast<unsigned char>(128); // Fixed blue value
            fwrite(&r, 1, 1, fp);
            fwrite(&g, 1, 1, fp);
            fwrite(&b, 1, 1, fp);
        }
    }

    fclose(fp);
    printf("Saved cube map face to %s\n", filename.c_str());
}

void LGH::save_all_cube_maps() {
    for (size_t level = 0; level < shadow_cube_maps.size(); ++level) {
        for (size_t light_idx = 0; light_idx < shadow_cube_maps[level].size(); ++light_idx) {
            const CubeMap& cmap = shadow_cube_maps[level][light_idx];
            for (int face = 0; face < 6; ++face) {
                std::ostringstream oss;
                oss << "shadowmap_L" << level << "_light" << light_idx << "_face" << face << ".ppm";
                save_cube_map_face_as_ppm(cmap.faces[face], oss.str());
            }
        }
    }
}

void LGH::display_cube_map_face(const std::vector<std::vector<float>>& face_data, int face_idx) {
    // ANSI color codes
    const char* colors[] = {
        "\033[31m", // Red
        "\033[32m", // Green
        "\033[33m", // Yellow
        "\033[34m", // Blue
        "\033[35m", // Magenta
        "\033[36m"  // Cyan
    };
    const char* reset = "\033[0m";

    int width = face_data.size();
    int height = face_data[0].size();
    
    // ASCII characters for different shadow values
    const char* chars = " .:-=+*#%@";
    int num_chars = strlen(chars);

    printf("\nFace %d (%s%s%s):\n", face_idx, colors[face_idx], 
           face_idx == 0 ? "+X" : face_idx == 1 ? "-X" : 
           face_idx == 2 ? "+Y" : face_idx == 3 ? "-Y" :
           face_idx == 4 ? "+Z" : "-Z", reset);

    // Print top border
    printf("┌");
    for (int i = 0; i < width; i++) printf("─");
    printf("┐\n");

    // Print face data
    for (int y = 0; y < height; y++) {
        printf("│");
        for (int x = 0; x < width; x++) {
            float shadow = face_data[x][y];
            int char_idx = static_cast<int>(shadow * (num_chars - 1));
            char_idx = std::max(0, std::min(char_idx, num_chars - 1));
            printf("%s%c%s", colors[face_idx], chars[char_idx], reset);
        }
        printf("│\n");
    }

    // Print bottom border
    printf("└");
    for (int i = 0; i < width; i++) printf("─");
    printf("┘\n");
}

void LGH::display_all_cube_maps() {
    for (size_t level = 0; level < shadow_cube_maps.size(); ++level) {
        printf("\n\n=== Level %zu Cube Maps ===\n", level);
        for (size_t light_idx = 0; light_idx < shadow_cube_maps[level].size(); ++light_idx) {
            printf("\n--- Light %zu ---\n", light_idx);
            const CubeMap& cmap = shadow_cube_maps[level][light_idx];
            
            // Display each face
            for (int face = 0; face < 6; ++face) {
                display_cube_map_face(cmap.faces[face], face);
            }
        }
    }
}