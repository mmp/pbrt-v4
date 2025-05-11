#include "lighting_grid_hierarchy.h"
#include "nanovdb/util/SampleFromVoxels.h"
#include "pbrt/util/spectrum.h"
#include <stdio.h>
#include <pbrt/media.h>
#include <cmath>
#include "CubeDS.h"
#include <functional>  
#include <vector>
enum class Face { PosX, NegX, PosY, NegY, PosZ, NegZ };
inline float dot(const Vector3f& a, const Vector3f& b)
    { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float length(const Vector3f& a) { return std::sqrt(dot(a,a)); }
inline Vector3f normalise(const Vector3f& a) {if (length(a) != 0) return a * (1.f/length(a)); else return a; }
// LGH::LGH(pbrt::SampledGrid<float> temperature_grid, int depth, float base_voxel_size, float transmission) 
LGH::LGH(const nanovdb::FloatGrid* temperature_grid, const nanovdb::FloatGrid* density_grid, int depth, float base_voxel_size, float transmission, pbrt::Transform transform)
    : l_max(depth), transmission(transmission), medium_transform(transform)
{
    auto worldBBox = temperature_grid->worldBBox();
    nanovdb::Vec3d minBBox = worldBBox.min();
    BBoxMin = Vector3f(minBBox[0], minBBox[1], minBBox[2]);

    nanovdb::Vec3d maxBBox = worldBBox.max();
    BBoxMax = Vector3f(maxBBox[0], maxBBox[1], maxBBox[2]);

    printf("Min Bounds: %f %f %f, Max Bounds: %f %f %f", BBoxMin.x, BBoxMin.y, BBoxMin.z, BBoxMax.x, BBoxMax.y, BBoxMax.z);

    this->m_temperature_grid = temperature_grid;
    m_density_grid = density_grid;

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
    

    // Create rest of S_l
    for (int i=1; i<=depth; i++) {
        deriveNewS(i);
    }

    if (lighting_grids.size() != l_max + 1) {
        LOG_FATAL("Invalid number of grids");
    }
}





/**
 * Return the 3‑D centre of texel (i,j) on the given cube‑map face.
 *   S  – number of texels per edge on that face
 *   h  – HALF edge length of the cube (cube spans –h…+h on each axis)
 */
static Vector3f texel_center(Face face, int i, int j, int S, float h)
{
    // map texel indices to face‑local (u,v) in [‑1,1], half‑pixel centred
    const float u = ( (i + 0.5f) / S ) * 2.0f - 1.0f;   // +u → right
    const float v = ( (j + 0.5f) / S ) * 2.0f - 1.0f;   // +v → up

    switch (face) {
        case Face::PosX: return {  h,  v*h, -u*h };
        case Face::NegX: return { -h,  v*h,  u*h };
        case Face::PosY: return {  u*h,  h, -v*h };
        case Face::NegY: return {  u*h, -h,  v*h };
        case Face::PosZ: return {  u*h,  v*h,  h };
        case Face::NegZ: return { -u*h,  v*h, -h };
    }
    return {0,0,0};   // should never hit
}

// ---------------------------------------------------------------------
// Cube‑map constructor
// ---------------------------------------------------------------------
using CubeMap = CubeDS;   // Alias for clarity

/**
 * level      – mip / LOD (if you use powers of two; otherwise ignore)
 * h          – HALF edge length of the cube (unit cube ⇒ h = 1)
 * light_pos  – position of the light in world / cube space (optional use)
 *
 * Returns a fully‑populated CubeMap where every Texel.Centroid is the
 * centre of a square on one of the six faces and Texel.Transmittance is
 * initialised to 0.0f (adjust this as your algorithm requires).
 */
// ‑‑‑‑‑ basic Vector3f helpers (add your own if you already have them) ‑‑‑‑‑
// inline Vector3f operator+(const Vector3f& a, const Vector3f& b)
//     { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
// inline Vector3f operator-(const Vector3f& a, const Vector3f& b)
//     { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
// inline Vector3f operator*(const Vector3f& a, float s)
//     { return {a.x*s, a.y*s, a.z*s}; }


// -------------------------------------------------------------------------
// Ray‑march helper – Beer–Lambert transmittance between two points
// -------------------------------------------------------------------------
using DensityFn = std::function<float(const Vector3f&)>;

static float transmittance_between(const Vector3f& a,
                                   const Vector3f& b,
                                   DensityFn        density_at,
                                   float            step,
                                   float            sigma_t)
{
    Vector3f dir = b - a;
    float len    = length(dir);
    dir          = normalise(dir);

    int   N     = static_cast<int>(std::ceil(len / step));
    float tau   = 0.0f;           // optical depth

    for (int k = 0; k < N; ++k) {
        float t = (k + 0.5f) * step;
        if (t > len) break;
        tau += density_at(a + dir * t) * step;
    }
    return std::exp(-sigma_t * tau);
}

// -------------------------------------------------------------------------
// create_cube_map  – now populates Texel::Transmittance in‑place
// -------------------------------------------------------------------------
CubeDS LGH::create_cube_map(int                level,
                        float              h,
                        const Vector3f&    light_pos,
                        float              step,
                        float              sigma_t)
{
    // Face resolution: power‑of‑two mip chain (feel free to swap strategy)
    int resolution = pow(2, level+2);  // 2^level
    sigma_t = 275;//250;

    CubeMap cube_map;
    cube_map.Texels.reserve(6 * resolution * resolution);

    for (int f = 0; f < 6; ++f) {
        const Face face = static_cast<Face>(f);

        for (int v = 0; v < resolution; ++v) {     // rows (‑v = bottom)
            for (int u = 0; u < resolution; ++u) { // cols (‑u = left)
                pbrt::SampledSpectrum V(1.f);

                Vector3f centre =
                    texel_center(face, u, v, resolution, h);

                    Vector3f dir = normalise( light_pos - centre);

                    Vector3f p = centre;
                    while (p.distance(centre) < light_pos.distance(centre)) {
                        // Extinction: sigma_t = sigma_a + sigma_s
                
                        // March towards light to calculate occlusion with larger steps for less shadowing
                        // TODO: make relative to h?
                        float shadow_ds = .1;
                
                        nanovdb::Vec3f pIndex = m_density_grid->worldToIndexF(nanovdb::Vec3f(p.x, p.y, p.z));
                        using Sampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
                        float rs = Sampler(m_density_grid->tree())(pIndex);
                
                        // Apply lighter non-linear density mapping for less dramatic shadows
                        rs = pow(rs, 1.2f) * 0.8f;
                        V *= std::exp(-sigma_t * rs * shadow_ds);
                
                        // if (V.MaxComponentValue() != 1)
                        // printf("Transmission: %s, trMax:%f\n", V.ToString().c_str(), V.MaxComponentValue());
                
                        // // Optional: early termination if weight is too low
                        if (V.MaxComponentValue() < 1e-3f) {
                            // printf("     \nSTOPPINGGG Transmission: %s, tMax:%f\n\n", Tr.ToString().c_str(), tMax);
                            break;
                        }
                
                        p += dir * shadow_ds;
                    }

                cube_map.Texels.push_back({ centre, V });
            }
        }
    }
    return cube_map;
}


void LGH::create_S0(const nanovdb::FloatGrid* temperature_grid)
{
    printf("\n===============CREATING S0===========\n");
    float h_0 = this->h[0];

    std::vector<std::tuple<Vector3f,float,CubeMap>> lights;

    // x,y,z should be in world space
    for (float x = BBoxMin.x + h_0/2; x < BBoxMax.x; x += h_0) {
        for (float y = BBoxMin.y + h_0/2; y < BBoxMax.y; y += h_0) {
            for (float z = BBoxMin.z + h_0/2; z < BBoxMax.z; z += h_0) {

                // TODO: is the worldBBox in world space including the transform or without??
                // pbrt::Point3f p = medium_transform.ApplyInverse(pbrt::Point3f(x,y,z));


                nanovdb::Vec3f pIndex = temperature_grid->worldToIndexF(nanovdb::Vec3f(x, y, z));
                // TODO: actually do weighted sample for more accurate temperature

                using Sampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
                float temperature = Sampler(temperature_grid->tree())(pIndex);//(pbrt::Point3f(x,y,z));

                if (temperature > 0) {//TEMP_THRESHOLD) {
                    // TODO: derive intensity from temperature
                    // printf("S0 light: %f, pos: %f %f %f\n", temperature, x, y, z);
                    
                    CubeMap cube_map = create_cube_map(0, h_0, Vector3f(x,y,z), 0.1f, 400.f);
                    lights.emplace_back(Vector3f(x,y,z), temperature, cube_map);
                }
                // else {
                //     printf("ZERO LIGHT: S0 light: %f, pos: %f %f %f\n", temperature, x, y, z);
                // }

                if (temperature < 0) {
                    printf("Temperature negative: %f , point %f %f %f", temperature, x, y, z);
                }
            }
        }
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
    std::vector<std::tuple<Vector3f,float, CubeDS>> lights;

    // iterate over grid vertices of level l
    for (float x = BBoxMin.x; x < BBoxMax.x; x += h_l)
    for (float y = BBoxMin.y; y < BBoxMax.y; y += h_l)
    for (float z = BBoxMin.z; z < BBoxMax.z; z += h_l)
    {
        Vector3f target_light_pos(x,y,z);             // grid-vertex position q_i
    
        KDTree* lighting_grid_j = lighting_grids[l-1];
    
        // TODO: double check is radius is correct for 2x2x2 grid, used diagonal of cube
        // I SEE THE PROBLEM: currently position based on illumination centers not vertices (so not 2x2x2 grid!)
        // Should the estimated lighting determine radius based on vertex position or illumination centers? Because to do radius search in this part, we need to use vertex positions. illum centers!

        // TODO: it looks like real implementation uses exactly 8 lights...
        // Update: I believe this works as intended
        float radius = sqrt(3) * h[l-1];
        std::vector<KDNode*> j_lights;
        lighting_grid_j->radiusSearch(target_light_pos, radius, j_lights);
        // while(j_lights.size() > 8) {
        //     j_lights.pop_back();
        // }

        // printf("  level: %d, radius search lights: %lu\n", l, j_lights.size());

        float     I  = calcNewI (l, target_light_pos, j_lights);          // Eq (1)
        Vector3f  p  = calcNewPos(l, target_light_pos, j_lights);         // Eq (2)
        if (I>0.f) {
            CubeMap cube_map = create_cube_map(0, h_l, Vector3f(x,y,z), 0.1f, 400.f);
            lights.emplace_back(p, I, cube_map);
        }// TODO: Change threshold!!!
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
    float I_sum(0);
    for (auto j_light : j_lights) {
        float w = calcTrilinearWeight(j_light->point, target_light_pos, this->h[l]);
        I_sum += w * j_light->intensity;
    }
    return I_sum;
}

// -------- illumination centre (Eq 2) ------------------------------------
Vector3f LGH::calcNewPos(int l, Vector3f target_light_pos, std::vector<KDNode*> j_lights) //const Vector3f& gv, int l, const KDTree& S0) const
{
    Vector3f p_num(0,0,0);
    float p_denom(0);

    for (auto j_light : j_lights) {
        float w = calcTrilinearWeight(j_light->point, target_light_pos, this->h[l]);
        float v = w * j_light->intensity * 1000; // TODO: what is the luminance component? What exactly is intensity data structure?
        p_num += v * j_light->point;
        p_denom += v;
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

pbrt::SampledSpectrum get_closest_texel(Vector3f pos, CubeDS cube_map) {
    float min_dist = 1e10;
    Texel closest_texel;
    for (auto texel : cube_map.Texels) {
        float dist = length(texel.position - pos);
        if (dist < min_dist) {
            min_dist = dist;
            closest_texel = texel;
        }
    }
    return closest_texel.transmittance;
}

pbrt::SampledSpectrum LGH::get_intensity(int L,
                                         Vector3f targetPos,
                                         KDNode* light,
                                         float radius,
                                         pbrt::SampledWavelengths lambda,
                                         pbrt::Sampler sampler,
                                         pbrt::Medium medium)
{
    // V = Tr(x, light->point)

    // Calculate tranmittance
    // TODO: check how much time this transmittance calc adds
    Vector3f dir = light->point - targetPos;

    // pbrt::Ray ray = pbrt::Ray(pbrt::Point3f(targetPos.x, targetPos.y, targetPos.z), pbrt::Vector3f(dir.x, dir.y, dir.z), 0, medium);

    // TODO: i dont think tmax is right
    // float tMax = 1e10;//Length(ray.d);
    // ray.d = Normalize(ray.d);

    // printf("tmax: %f\n", tMax);
    // Initialize _RNG_ for sampling the majorant transmittance
    // uint64_t hash0 = pbrt::Hash(sampler.Get1D());
    // uint64_t hash1 = pbrt::Hash(sampler.Get1D());
    // pbrt::RNG rng(hash0, hash1);

    pbrt::SampledSpectrum V = get_closest_texel(targetPos, light->shadowCubeMap);

    // if (V.MaxComponentValue() != 1)
        // printf("transmittance: %s\n", V.ToString().c_str());


    float d = targetPos.distance(light->point);

    // TODO: figure out good scale for light fall-off, make sure it is relative to size of explosion
    float g = 1.f / (1 + pow(d,2)); // Light fall-off, good with d * 50
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

    if (light->intensity * 1500 < 10) {
        // printf("Too low light intensity L: %d, %f\n", L, light->intensity * 4500);
        return pbrt::SampledSpectrum(0);
    }

    // printf("Light intensity: %f, transmittance: %s, g: %f, B: %f\n", light->intensity * 1000, V.ToString().c_str(), g, B);


    // TODO NOTE * 200 untested. No idea what value it should be
    return g * B * pbrt::BlackbodySpectrum(light->intensity * 1500).Sample(lambda) * V *1500;//* 1000; //light->intensity * V;//
}

pbrt::SampledSpectrum LGH::get_total_illum(pbrt::Point3f pos,
                                           pbrt::SampledWavelengths lambda,
                                           pbrt::Sampler sampler,
                                           pbrt::Medium medium,
                                           pbrt::RNG rng,
                                           float tMax,
                                           pbrt::Ray ray)
{
    // Note that pos passed in callback is in medium local space! Convert to world-space to access lights
    pos = medium_transform.ApplyInverse(pos);

    Vector3f v_pos(pos.x, pos.y, pos.z);

    int numLightsCaptured = 0;
    int numLightsS0, numLightsS1, numLightsS2, numLightsS3, numLightsS4, numLightsS5 = 0;

    pbrt::SampledSpectrum total_intensity(0);
    for (int l=0; l<=l_max; l++) {
        float radius = alpha * h[l];

        std::vector<KDNode*> results;
        lighting_grids[l]->radiusSearch(v_pos, radius, results);
        numLightsCaptured += results.size();
        
        for (auto light : results) {
            total_intensity += get_intensity(l, v_pos, light, radius, lambda, sampler, medium);
        }
    }

    return 0.125 * total_intensity; //pbrt::BlackbodySpectrum(total_intensity).Sample(lambda);
}
// Pyramid filter function (from reference image, Eq. 6)
static float pyramid_filter(int i, int j, int k, int h_l) {
    // i, j, k in {-1, 0, 1}
    // h_l is the filter size (should be 3 for 3x3x3)
    return (1.0f / 8.0f) * (1.0f - std::abs(i) / float(h_l)) * (1.0f - std::abs(j) / float(h_l)) * (1.0f - std::abs(k) / float(h_l));
}

// void LGH::prefilter_density_field(int level, float h_l, const nanovdb::FloatGrid* density_grid) {
//     std::vector<float> densities;
//     std::vector<Vector3f> vertices;

//     // For each grid vertex at this level
//     for (float x = BBoxMin.x; x < BBoxMax.x; x += h_l)
//     for (float y = BBoxMin.y; y < BBoxMax.y; y += h_l)
//     for (float z = BBoxMin.z; z < BBoxMax.z; z += h_l) {
//         Vector3f v(x, y, z);
//         float filtered_density = 0.0f;
//         // 3x3x3 stencil centered at v
//         for (int dx = -1; dx <= 1; ++dx)
//         for (int dy = -1; dy <= 1; ++dy)
//         for (int dz = -1; dz <= 1; ++dz) {
//             float nx = x + dx * h_l;
//             float ny = y + dy * h_l;
//             float nz = z + dz * h_l;
//             // Check bounds
//             if (nx < BBoxMin.x || nx >= BBoxMax.x ||
//                 ny < BBoxMin.y || ny >= BBoxMax.y ||
//                 nz < BBoxMin.z || nz >= BBoxMax.z)
//                 continue;
//             // Convert to grid index
//             nanovdb::Vec3f pIndex = density_grid->worldToIndexF(nanovdb::Vec3f(nx, ny, nz));
//             using Sampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
//             float density = Sampler(density_grid->tree())(pIndex);
//             float w = pyramid_filter(dx, dy, dz, 2); // h_l=2 for 3x3x3
//             filtered_density += w * density;
//         }
//         densities.push_back(filtered_density);
//         vertices.push_back(v);
//     }
//     // Store for this level
//     if (filtered_densities.size() <= level) filtered_densities.resize(level+1);
//     if (grid_vertices.size() <= level) grid_vertices.resize(level+1);
//     filtered_densities[level] = std::move(densities);
//     grid_vertices[level] = std::move(vertices);
// }

// // Helper: directions for cube map faces
// static const Vector3f cube_face_dirs[6][3] = {
//     { {1,0,0}, {0,1,0}, {0,0,1} },   // +X
//     { {-1,0,0}, {0,1,0}, {0,0,-1} }, // -X
//     { {0,1,0}, {1,0,0}, {0,0,1} },   // +Y
//     { {0,-1,0}, {1,0,0}, {0,0,-1} }, // -Y
//     { {0,0,1}, {1,0,0}, {0,1,0} },   // +Z
//     { {0,0,-1}, {1,0,0}, {0,-1,0} }  // -Z
// };

// // Pyramidal filter function (Eq. 6)
// static float lambda_pyramid(int i, int j, int k, float h) {
//     return (1.0f / 8.0f) * (1.0f - std::abs(i) / h) * (1.0f - std::abs(j) / h) * (1.0f - std::abs(k) / h);
// }

// // Trilinear interpolation for filtered density field
// float LGH::filtered_density_at(int level, const Vector3f& pos) const {
//     // Find the 8 grid vertices surrounding pos at this level
//     float h_l = h[level];
//     Vector3f grid_pos = (pos - BBoxMin) / h_l;
//     int x0 = static_cast<int>(std::floor(grid_pos.x));
//     int y0 = static_cast<int>(std::floor(grid_pos.y));
//     int z0 = static_cast<int>(std::floor(grid_pos.z));
//     float fx = grid_pos.x - x0;
//     float fy = grid_pos.y - y0;
//     float fz = grid_pos.z - z0;
//     float result = 0.0f;
//     for (int dx = 0; dx <= 1; ++dx)
//     for (int dy = 0; dy <= 1; ++dy)
//     for (int dz = 0; dz <= 1; ++dz) {
//         int xi = x0 + dx;
//         int yi = y0 + dy;
//         int zi = z0 + dz;
//         int idx = (xi * ((int)((BBoxMax.y - BBoxMin.y) / h_l) + 1) + yi) * ((int)((BBoxMax.z - BBoxMin.z) / h_l) + 1) + zi;
//         if (xi < 0 || yi < 0 || zi < 0 || xi >= (int)((BBoxMax.x - BBoxMin.x) / h_l) + 1 || yi >= (int)((BBoxMax.y - BBoxMin.y) / h_l) + 1 || zi >= (int)((BBoxMax.z - BBoxMin.z) / h_l) + 1)
//             continue;
//         float w = ((dx ? fx : 1 - fx) * (dy ? fy : 1 - fy) * (dz ? fz : 1 - fz));
//         result += w * filtered_densities[level][idx];
//     }
//     return result;
// }

// // Filtered density at x with filter size delta, using Eq. 7
// float LGH::filtered_density_with_filter(const Vector3f& pos, float delta) const {
//     // Find the two levels such that h_{l-1} <= delta <= h_l
//     int l = 0;
//     while (l + 1 < (int)h.size() && h[l + 1] < delta) ++l;
//     int l0 = l, l1 = std::min(l + 1, (int)h.size() - 1);
//     float h0 = h[l0], h1 = h[l1];
//     float rho0 = filtered_density_at(l0, pos);
//     float rho1 = filtered_density_at(l1, pos);
//     if (h1 == h0) return rho0;
//     float t = (delta - h0) / (h1 - h0);
//     return (1 - t) * rho0 + t * rho1;
// }

// // Update compute_cube_map_for_light to use the variable filter size and filtered density
// void LGH::compute_cube_map_for_light(int level, int light_idx, const Vector3f& light_pos, float r_e, float h0, const nanovdb::FloatGrid* density_grid) {
//     int resolution = static_cast<int>(std::ceil(2 * r_e / h0));
//     printf("\nCreating cube map for level %d, light %d:\n", level, light_idx);
//     printf("  Light position: (%f, %f, %f)\n", light_pos.x, light_pos.y, light_pos.z);
//     printf("  Effective radius (r_e): %f\n", r_e);
//     printf("  Base voxel size (h0): %f\n", h0);
//     printf("  Resolution: %d x %d\n", resolution, resolution);
    
//     CubeMap cmap;
//     cmap.resolution = resolution;
//     cmap.r_e = r_e;
//     cmap.h = h0;

//     cmap.
    
//     // Calculate s_{l,i} based on the level and light properties
//     float s_li = h[level]; // CHANGE TO SIZE OF FORMULA
//     printf("  Filter size s_li: %f\n", s_li);
    
//     for (int face = 0; face < 6; ++face) {
//         printf("  Computing face %d\n", face);
//         cmap.faces[face].resize(resolution, std::vector<float>(resolution, 0.0f));
//         float max_shadow = 0.0f;
//         float min_shadow = 1e10f;
        
//         for (int u = 0; u < resolution; ++u) {
//             for (int v = 0; v < resolution; ++v) {
//                 float s = 2.0f * (u + 0.5f) / resolution - 1.0f;
//                 float t = 2.0f * (v + 0.5f) / resolution - 1.0f;
//                 Vector3f x = cube_face_dirs[face][0];
//                 Vector3f y = cube_face_dirs[face][1];
//                 Vector3f z = cube_face_dirs[face][2];

                
                
//                 Vector3f dir = (x * s + y * t + z).abs();
//                 dir = dir / std::sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
//                 float shadow = 0.0f;
//                 float step = h0;
                
//                 // Use variable filter size based on distance from light
//                 for (float d = 0; d < 2 * r_e; d += step) {
//                     Vector3f p = light_pos + dir * d;
//                     float delta = 0.0f;
//                     if (d <= r_e) {
//                         delta = (1 - d / r_e) * s_li + (d / r_e) * h0;
//                     } else {
//                         // Beyond r_e, use the base filter size
//                         delta = h0;
//                     }
                    
//                     // Get filtered density using the variable filter size
//                     float filtered_density = filtered_density_with_filter(p, delta);
//                     shadow += filtered_density * step;
//                     // shadow = 0.5;
//                 }
//                 cmap.faces[face][u][v] = shadow;
//                 max_shadow = std::max(max_shadow, shadow);
//                 min_shadow = std::min(min_shadow, shadow);
//             }
//         }
//         printf("    Face %d shadow range: [%f, %f]\n", face, min_shadow, max_shadow);
//     }
    
//     if (shadow_cube_maps.size() <= level) shadow_cube_maps.resize(level+1);
//     if (shadow_cube_maps[level].size() <= light_idx) shadow_cube_maps[level].resize(light_idx+1);
//     shadow_cube_maps[level][light_idx] = std::move(cmap);
    
//     int total_texels = 6 * resolution * resolution;
//     float expected_texels = 24.0f * (r_e / h0) * (r_e / h0);
//     printf("Cube map: %d texels (expected: %.1f)\n", total_texels, expected_texels);
//     int c=0;

// //     for (int face = 0; face < 6; ++face) {
// //     if (c>=1) {
// //         break;
// //     }
// //     c++;
// //     std::ostringstream oss;
// //     oss << "shadowmap_L" << level << "_light" << light_idx << "_face" << face << ".pgm";
// //     save_cube_map_face_as_pgm(cmap.faces[face], oss.str());
// // }
// }

// float LGH::lookup_shadow(int level, int light_idx, const Vector3f& light_pos, const Vector3f& target_pos) const {
//     // Check if shadow_cube_maps is properly initialized
//     if (shadow_cube_maps.empty() || level >= shadow_cube_maps.size()) {
//         return 0.0f;
//     }
    
//     if (light_idx >= shadow_cube_maps[level].size()) {
//         return 0.0f;
//     }

//     // Get direction from light to target
//     Vector3f dir = target_pos - light_pos;
//     float dist = std::sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
//     if (dist < 1e-6f) {
//         return 0.0f;
//     }
//     dir = dir / dist;  // Normalize

//     // Find the face with the largest absolute component
//     int face = 0;
//     float max_comp = std::abs(dir.x);
//     if (std::abs(dir.y) > max_comp) {
//         max_comp = std::abs(dir.y);
//         face = 2;
//     }
//     if (std::abs(dir.z) > max_comp) {
//         max_comp = std::abs(dir.z);
//         face = 4;
//     }
//     // Adjust face index based on sign
//     if (dir[face/2] < 0) face++;

//     // Get the cube map for this level and light
//     const CubeMap& cmap = shadow_cube_maps[level][light_idx];
//     if (face >= 6 || face < 0) {
//         return 0.0f;
//     }
//     const auto& face_data = cmap.faces[face];

//     // Convert direction to face-local coordinates
//     Vector3f x = cube_face_dirs[face][0];
//     Vector3f y = cube_face_dirs[face][1];
//     Vector3f z = cube_face_dirs[face][2];
    
//     // Project direction onto face basis
//     float s = (dir.x*x.x + dir.y*x.y + dir.z*x.z) / (dir.x*z.x + dir.y*z.y + dir.z*z.z);
//     float t = (dir.x*y.x + dir.y*y.y + dir.z*y.z) / (dir.x*z.x + dir.y*z.y + dir.z*z.z);
    
//     // Convert to pixel coordinates
//     float u = (s + 1.0f) * 0.5f * (cmap.resolution - 1);
//     float v = (t + 1.0f) * 0.5f * (cmap.resolution - 1);
    
//     // Clamp coordinates to valid range
//     u = std::max(0.0f, std::min(u, static_cast<float>(cmap.resolution - 1)));
//     v = std::max(0.0f, std::min(v, static_cast<float>(cmap.resolution - 1)));
    
//     // Get the nearest pixel value (no interpolation)
//     int u_idx = static_cast<int>(std::round(u));
//     int v_idx = static_cast<int>(std::round(v));
    
//     // Ensure indices are within bounds
//     u_idx = std::min(std::max(0, u_idx), cmap.resolution - 1);
//     v_idx = std::min(std::max(0, v_idx), cmap.resolution - 1);
    
//     // Return the shadow value from the specific face
//     //return 0.5; 
//     return face_data[u_idx][v_idx];
// }// #include "lighting_grid_hierarchy.h"
// #include "nanovdb/util/SampleFromVoxels.h"
// #include "pbrt/util/spectrum.h"
// #include <stdio.h>
// #include <pbrt/media.h>
// #include <cmath>

// // LGH::LGH(pbrt::SampledGrid<float> temperature_grid, int depth, float base_voxel_size, float transmission) 
// LGH::LGH(const nanovdb::FloatGrid* temperature_grid, const nanovdb::FloatGrid* density_grid, int depth, float base_voxel_size, float transmission, pbrt::Transform transform)
//     : l_max(depth), transmission(transmission), medium_transform(transform)
// {
//     auto worldBBox = temperature_grid->worldBBox();
//     nanovdb::Vec3d minBBox = worldBBox.min();
//     BBoxMin = Vector3f(minBBox[0], minBBox[1], minBBox[2]);

//     nanovdb::Vec3d maxBBox = worldBBox.max();
//     BBoxMax = Vector3f(maxBBox[0], maxBBox[1], maxBBox[2]);

//     printf("Min Bounds: %f %f %f, Max Bounds: %f %f %f", BBoxMin.x, BBoxMin.y, BBoxMin.z, BBoxMax.x, BBoxMax.y, BBoxMax.z);

//     this->m_temperature_grid = temperature_grid;
//     m_density_grid = density_grid;

//     // Initialize h
//     for (int i=0; i<=depth; i++) {
//         h.push_back(base_voxel_size * pow(2,i));
//     }

//     printf("\n============= H ==============\n");
//     for (float h_val : h) {
//         printf("%f ", h_val);
//     }
//     printf("\n\n");

//     if (h.size() != l_max + 1) {
//         LOG_FATAL("invalid number of h!!!");
//     }

//     lighting_grids = std::vector<KDTree*>(depth + 1);

//     // TODO: should I pass in the accessor? Do I need the grid?
//     // Initialize S0
//     create_S0(temperature_grid);

//     // Create rest of S_l
//     for (int i=1; i<=depth; i++) {
//         deriveNewS(i);
//     }

//     if (lighting_grids.size() != l_max + 1) {
//         LOG_FATAL("Invalid number of grids");
//     }
// }



// void LGH::create_S0(const nanovdb::FloatGrid* temperature_grid)
// {
//     printf("\n===============CREATING S0===========\n");
//     float h_0 = this->h[0];

//     std::vector<std::pair<Vector3f,float>> lights;

//     // x,y,z should be in world space
//     for (float x = BBoxMin.x + h_0/2; x < BBoxMax.x; x += h_0) {
//         for (float y = BBoxMin.y + h_0/2; y < BBoxMax.y; y += h_0) {
//             for (float z = BBoxMin.z + h_0/2; z < BBoxMax.z; z += h_0) {

//                 // TODO: is the worldBBox in world space including the transform or without??
//                 // pbrt::Point3f p = medium_transform.ApplyInverse(pbrt::Point3f(x,y,z));


//                 nanovdb::Vec3f pIndex = temperature_grid->worldToIndexF(nanovdb::Vec3f(x, y, z));
//                 // TODO: actually do weighted sample for more accurate temperature

//                 using Sampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
//                 float temperature = Sampler(temperature_grid->tree())(pIndex);//(pbrt::Point3f(x,y,z));

//                 if (temperature > 0) {//TEMP_THRESHOLD) {
//                     // TODO: derive intensity from temperature
//                     // printf("S0 light: %f, pos: %f %f %f\n", temperature, x, y, z);
//                     lights.emplace_back(Vector3f(x,y,z), temperature);
//                 }
//                 // else {
//                 //     printf("ZERO LIGHT: S0 light: %f, pos: %f %f %f\n", temperature, x, y, z);
//                 // }

//                 if (temperature < 0) {
//                     printf("Temperature negative: %f , point %f %f %f", temperature, x, y, z);
//                 }
//             }
//         }
//     }

//     this->lighting_grids[0] = new KDTree(lights);
//     printf("=============================\nCreated %lu lights in S0\n=============================", lights.size());
// }

// void LGH::deriveNewS(int l)
// {
//     if (l == 0) {
//         LOG_FATAL("Level should never be 0!");
//         return;
//     }
//     float h_l = this->h[l];                     // spacing of level l
//     std::vector<std::pair<Vector3f,float>> lights;

//     // iterate over grid vertices of level l
//     for (float x = BBoxMin.x; x < BBoxMax.x; x += h_l)
//     for (float y = BBoxMin.y; y < BBoxMax.y; y += h_l)
//     for (float z = BBoxMin.z; z < BBoxMax.z; z += h_l)
//     {
//         Vector3f target_light_pos(x,y,z);             // grid-vertex position q_i
    
//         KDTree* lighting_grid_j = lighting_grids[l-1];
    
//         // TODO: double check is radius is correct for 2x2x2 grid, used diagonal of cube
//         // I SEE THE PROBLEM: currently position based on illumination centers not vertices (so not 2x2x2 grid!)
//         // Should the estimated lighting determine radius based on vertex position or illumination centers? Because to do radius search in this part, we need to use vertex positions. illum centers!

//         // TODO: it looks like real implementation uses exactly 8 lights...
//         // Update: I believe this works as intended
//         float radius = sqrt(3) * h[l-1];
//         std::vector<KDNode*> j_lights;
//         lighting_grid_j->radiusSearch(target_light_pos, radius, j_lights);
//         // while(j_lights.size() > 8) {
//         //     j_lights.pop_back();
//         // }

//         // printf("  level: %d, radius search lights: %lu\n", l, j_lights.size());

//         float     I  = calcNewI (l, target_light_pos, j_lights);          // Eq (1)
//         Vector3f  p  = calcNewPos(l, target_light_pos, j_lights);         // Eq (2)
//         if (I>0.f) lights.emplace_back(p,I); // TODO: Change threshold!!!
//     }

//     lighting_grids[l] = new KDTree(lights);
//     printf("=============================\nCreated %lu lights in S%d\n=============================", lights.size(), l);

// }
// float calcTrilinearWeight(Vector3f p, Vector3f q, float h_l) {
//     Vector3f v = Vector3f(1,1,1) - (p - q).abs() / h_l;
//     float v_product = v.x * v.y * v.z;
//     return std::max(0.f, std::min(1.f, v_product));
// }

// // -------- intensity (Eq 1) ----------------------------------------------
// float LGH::calcNewI(int l, Vector3f target_light_pos, std::vector<KDNode*> j_lights) //const Vector3f& gv, int l, const KDTree& S0) const
// {
//     float I_sum(0);
//     for (auto j_light : j_lights) {
//         float w = calcTrilinearWeight(j_light->point, target_light_pos, this->h[l]);
//         I_sum += w * j_light->intensity;
//     }
//     return I_sum;
// }

// // -------- illumination centre (Eq 2) ------------------------------------
// Vector3f LGH::calcNewPos(int l, Vector3f target_light_pos, std::vector<KDNode*> j_lights) //const Vector3f& gv, int l, const KDTree& S0) const
// {
//     Vector3f p_num(0,0,0);
//     float p_denom(0);

//     for (auto j_light : j_lights) {
//         float w = calcTrilinearWeight(j_light->point, target_light_pos, this->h[l]);
//         float v = w * j_light->intensity * 1000; // TODO: what is the luminance component? What exactly is intensity data structure?
//         p_num += v * j_light->point;
//         p_denom += v;
//     }

//     return p_num/p_denom;
// }

// float LGH::blendingFunction(int level, float d, float r_l)
// {
    
//     if (level == 0) {
//         if (d <= r_l) return 1;
//         else if (d <= 2 * r_l) return 2 - d/r_l;
//         else return 0;
//     }

//     if (level == l_max) {
//         if (d <= r_l/2) return 0;
//         else if (d <= r_l) return 2*d/r_l - 1;
//         else return 1;
//     }

//     if (d <= r_l/2) return 0;
//     else if (d <= r_l) return 2*d/r_l - 1;
//     else if (d <= 2* r_l) return 2 - d/r_l;
//     return 0;
// }

// pbrt::SampledSpectrum LGH::get_intensity(int L,
//                                          Vector3f targetPos,
//                                          KDNode* light,
//                                          float radius,
//                                          pbrt::SampledWavelengths lambda,
//                                          pbrt::Sampler sampler,
//                                          pbrt::Medium medium)
// {
//     // V = Tr(x, light->point)

//     // Calculate tranmittance
//     // TODO: check how much time this transmittance calc adds
//     Vector3f dir = light->point - targetPos;

//     // pbrt::Ray ray = pbrt::Ray(pbrt::Point3f(targetPos.x, targetPos.y, targetPos.z), pbrt::Vector3f(dir.x, dir.y, dir.z), 0, medium);

//     // TODO: i dont think tmax is right
//     // float tMax = 1e10;//Length(ray.d);
//     // ray.d = Normalize(ray.d);

//     // printf("tmax: %f\n", tMax);
//     // Initialize _RNG_ for sampling the majorant transmittance
//     // uint64_t hash0 = pbrt::Hash(sampler.Get1D());
//     // uint64_t hash1 = pbrt::Hash(sampler.Get1D());
//     // pbrt::RNG rng(hash0, hash1);

//     pbrt::SampledSpectrum V(1.f);

//     //     //pbrt::SampledSpectrum V =
//     // SampleT_maj(ray, tMax, rng.Uniform<float>(), rng, lambda,
//     //     [&](pbrt::Point3f p, pbrt::MediumProperties mp, pbrt::SampledSpectrum sigma_maj, pbrt::SampledSpectrum T_maj) {
//     //         // printf("sigma_maj: %s, t_maj: %s, a: %s, s: %s\n", sigma_maj.ToString().c_str(), T_maj.ToString().c_str(), mp.sigma_a.ToString().c_str(), mp.sigma_s.ToString().c_str());
//     //         // Null-collision transmittance estimation

//     //         p = ray.o;
//     //         while (Length(p-ray.o) < light->point.distance(targetPos)) {
//     //             // Extinction: sigma_t = sigma_a + sigma_s
//     //             pbrt::SampledSpectrum sigma_t = mp.sigma_a + mp.sigma_s;

//     //             // // Null-collision cross section
//     //             // pbrt::SampledSpectrum sigma_n = ClampZero(sigma_maj - sigma_t);

//     //             // // Probability of null collision (used to weight the sample)
//     //             // float pr = T_maj[0] * sigma_maj[0];  // scalar for importance sampling weight

//     //             // // Transmittance update for null collision
//     //             // Tr *= T_maj * sigma_n / pr;

//     //             const float ds = 0.03f;

//     //             // // March towards light to calculate occlusion with larger steps for less shadowing
//     //             float shadow_ds = ds * 1.5f;

//     //             nanovdb::Vec3f pIndex = m_density_grid->worldToIndexF(nanovdb::Vec3f(p.x, p.y, p.z));
//     //             // TODO: actually do weighted sample for more accurate temperature

//     //             using Sampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
//     //             float rs = Sampler(m_density_grid->tree())(pIndex);

//     //             //     // Apply lighter non-linear density mapping for less dramatic shadows
//     //             //     rs = pow(rs, 1.2f) * 0.8f;
//     //             // V *= std::exp(-sigma_t.y(lambda) * rs * shadow_ds);
//     //             V *= std::exp(-4.5 * rs * shadow_ds);

//     //             // if (V.MaxComponentValue() != 1)
//     //             // printf("Transmission: %s, trMax:%f\n", V.ToString().c_str(), V.MaxComponentValue());

//     //             // // Optional: early termination if weight is too low
//     //             if (V.MaxComponentValue() < 1e-3f) {
//     //                 // printf("     \nSTOPPINGGG Transmission: %s, tMax:%f\n\n", Tr.ToString().c_str(), tMax);
//     //                 return false; // stop sampling
//     //             }

//     //             p += ds * ray.d;
//     //         }

//     //         return false; // keep sampling
//     //     });//.y(lambda);

//     Vector3f p = targetPos;
//     while (p.distance(targetPos) < light->point.distance(targetPos)) {
//         // Extinction: sigma_t = sigma_a + sigma_s

//         // TODO: NOTE THIS VALUE IS UNTESTED. Needs to be greater than 80 at least
//         float sigma_t = 400;//4.5f * 20;

//         // March towards light to calculate occlusion with larger steps for less shadowing
//         // TODO: make relative to h?
//         float shadow_ds = .1;

//         nanovdb::Vec3f pIndex = m_density_grid->worldToIndexF(nanovdb::Vec3f(p.x, p.y, p.z));
//         using Sampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
//         float rs = Sampler(m_density_grid->tree())(pIndex);

//         // Apply lighter non-linear density mapping for less dramatic shadows
//         rs = pow(rs, 1.2f) * 0.8f;
//         V *= std::exp(-sigma_t * rs * shadow_ds);

//         // if (V.MaxComponentValue() != 1)
//         // printf("Transmission: %s, trMax:%f\n", V.ToString().c_str(), V.MaxComponentValue());

//         // // Optional: early termination if weight is too low
//         if (V.MaxComponentValue() < 1e-3f) {
//             // printf("     \nSTOPPINGGG Transmission: %s, tMax:%f\n\n", Tr.ToString().c_str(), tMax);
//             break;
//         }

//         p += shadow_ds * dir;
//     }

//         // Tr_sum += V;
//     // }
//     // pbrt::SampledSpectrum V = Tr_sum / float(nSamples);

//     // float V = 1;


//     // if (V.MaxComponentValue() != 1)
//         // printf("transmittance: %s, tMax: %f\n", V.ToString().c_str(), tMax);


//     float d = targetPos.distance(light->point);

//     // TODO: figure out good scale for light fall-off, make sure it is relative to size of explosion
//     float shadow = lookup_shadow(L, light->idx, light->point, targetPos);
//     float g = 1.f / (1 + pow(d,2)); // Light fall-off, good with d * 50
//     g = std::min(1.f, g);

//     float B = blendingFunction(L, d, radius);

//     if (B < 0) {
//         LOG_FATAL("Blending function should never be negative! %f", B);
//     }

//     if (B > 1) {
//         printf("B large: %f\n", B);
//     }

//     if (light->intensity < 0) {
//         printf("LIGHT INTENSITY NEGATIVE!!! L: %d, %f\n", L, light->intensity);
//     }

//     if (light->intensity * 1000 < 10) {
//         // printf("Too low light intensity L: %d, %f\n", L, light->intensity * 4500);
//         return pbrt::SampledSpectrum(0);
//     }

//     // printf("Light intensity: %f, transmittance: %s, g: %f, B: %f\n", light->intensity * 1000, V.ToString().c_str(), g, B);


//     // TODO NOTE * 200 untested. No idea what value it should be
//     return g * B * pbrt::BlackbodySpectrum(light->intensity * 1000).Sample(lambda) * V * 200 * (1-shadow); //light->intensity * V;//
// }

// pbrt::SampledSpectrum LGH::get_total_illum(pbrt::Point3f pos,
//                                            pbrt::SampledWavelengths lambda,
//                                            pbrt::Sampler sampler,
//                                            pbrt::Medium medium,
//                                            pbrt::RNG rng,
//                                            float tMax,
//                                            pbrt::Ray ray)
// {
//     // Note that pos passed in callback is in medium local space! Convert to world-space to access lights
//     pos = medium_transform.ApplyInverse(pos);

//     Vector3f v_pos(pos.x, pos.y, pos.z);

//     int numLightsCaptured = 0;
//     int numLightsS0, numLightsS1, numLightsS2, numLightsS3, numLightsS4, numLightsS5 = 0;

//     pbrt::SampledSpectrum total_intensity(0);
//     for (int l=0; l<=l_max; l++) {
//         float radius = alpha * h[l];

//         std::vector<KDNode*> results;
//         lighting_grids[l]->radiusSearch(v_pos, radius, results);
//         numLightsCaptured += results.size();
        
//         for (auto light : results) {
//             total_intensity += get_intensity(l, v_pos, light, radius, lambda, sampler, medium);
//         }
//     }

//     return 0.125 * total_intensity; //pbrt::BlackbodySpectrum(total_intensity).Sample(lambda);
// }
// // Pyramid filter function (from reference image, Eq. 6)
// static float pyramid_filter(int i, int j, int k, int h_l) {
//     // i, j, k in {-1, 0, 1}
//     // h_l is the filter size (should be 3 for 3x3x3)
//     return (1.0f / 8.0f) * (1.0f - std::abs(i) / float(h_l)) * (1.0f - std::abs(j) / float(h_l)) * (1.0f - std::abs(k) / float(h_l));
// }

// void LGH::prefilter_density_field(int level, float h_l, const nanovdb::FloatGrid* density_grid) {
//     std::vector<float> densities;
//     std::vector<Vector3f> vertices;

//     // For each grid vertex at this level
//     for (float x = BBoxMin.x; x < BBoxMax.x; x += h_l)
//     for (float y = BBoxMin.y; y < BBoxMax.y; y += h_l)
//     for (float z = BBoxMin.z; z < BBoxMax.z; z += h_l) {
//         Vector3f v(x, y, z);
//         float filtered_density = 0.0f;
//         // 3x3x3 stencil centered at v
//         for (int dx = -1; dx <= 1; ++dx)
//         for (int dy = -1; dy <= 1; ++dy)
//         for (int dz = -1; dz <= 1; ++dz) {
//             float nx = x + dx * h_l;
//             float ny = y + dy * h_l;
//             float nz = z + dz * h_l;
//             // Check bounds
//             if (nx < BBoxMin.x || nx >= BBoxMax.x ||
//                 ny < BBoxMin.y || ny >= BBoxMax.y ||
//                 nz < BBoxMin.z || nz >= BBoxMax.z)
//                 continue;
//             // Convert to grid index
//             nanovdb::Vec3f pIndex = density_grid->worldToIndexF(nanovdb::Vec3f(nx, ny, nz));
//             using Sampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
//             float density = Sampler(density_grid->tree())(pIndex);
//             float w = pyramid_filter(dx, dy, dz, 2); // h_l=2 for 3x3x3
//             filtered_density += w * density;
//         }
//         densities.push_back(filtered_density);
//         vertices.push_back(v);
//     }
//     // Store for this level
//     if (filtered_densities.size() <= level) filtered_densities.resize(level+1);
//     if (grid_vertices.size() <= level) grid_vertices.resize(level+1);
//     filtered_densities[level] = std::move(densities);
//     grid_vertices[level] = std::move(vertices);
// }

// // Helper: directions for cube map faces
// static const Vector3f cube_face_dirs[6][3] = {
//     { {1,0,0}, {0,1,0}, {0,0,1} },   // +X
//     { {-1,0,0}, {0,1,0}, {0,0,-1} }, // -X
//     { {0,1,0}, {1,0,0}, {0,0,1} },   // +Y
//     { {0,-1,0}, {1,0,0}, {0,0,-1} }, // -Y
//     { {0,0,1}, {1,0,0}, {0,1,0} },   // +Z
//     { {0,0,-1}, {1,0,0}, {0,-1,0} }  // -Z
// };

// // Pyramidal filter function (Eq. 6)
// static float lambda_pyramid(int i, int j, int k, float h) {
//     return (1.0f / 8.0f) * (1.0f - std::abs(i) / h) * (1.0f - std::abs(j) / h) * (1.0f - std::abs(k) / h);
// }

// // Trilinear interpolation for filtered density field
// float LGH::filtered_density_at(int level, const Vector3f& pos) const {
//     // Find the 8 grid vertices surrounding pos at this level
//     float h_l = h[level];
//     Vector3f grid_pos = (pos - BBoxMin) / h_l;
//     int x0 = static_cast<int>(std::floor(grid_pos.x));
//     int y0 = static_cast<int>(std::floor(grid_pos.y));
//     int z0 = static_cast<int>(std::floor(grid_pos.z));
//     float fx = grid_pos.x - x0;
//     float fy = grid_pos.y - y0;
//     float fz = grid_pos.z - z0;
//     float result = 0.0f;
//     for (int dx = 0; dx <= 1; ++dx)
//     for (int dy = 0; dy <= 1; ++dy)
//     for (int dz = 0; dz <= 1; ++dz) {
//         int xi = x0 + dx;
//         int yi = y0 + dy;
//         int zi = z0 + dz;
//         int idx = (xi * ((int)((BBoxMax.y - BBoxMin.y) / h_l) + 1) + yi) * ((int)((BBoxMax.z - BBoxMin.z) / h_l) + 1) + zi;
//         if (xi < 0 || yi < 0 || zi < 0 || xi >= (int)((BBoxMax.x - BBoxMin.x) / h_l) + 1 || yi >= (int)((BBoxMax.y - BBoxMin.y) / h_l) + 1 || zi >= (int)((BBoxMax.z - BBoxMin.z) / h_l) + 1)
//             continue;
//         float w = ((dx ? fx : 1 - fx) * (dy ? fy : 1 - fy) * (dz ? fz : 1 - fz));
//         result += w * filtered_densities[level][idx];
//     }
//     return result;
// }

// // Filtered density at x with filter size delta, using Eq. 7
// float LGH::filtered_density_with_filter(const Vector3f& pos, float delta) const {
//     // Find the two levels such that h_{l-1} <= delta <= h_l
//     int l = 0;
//     while (l + 1 < (int)h.size() && h[l + 1] < delta) ++l;
//     int l0 = l, l1 = std::min(l + 1, (int)h.size() - 1);
//     float h0 = h[l0], h1 = h[l1];
//     float rho0 = filtered_density_at(l0, pos);
//     float rho1 = filtered_density_at(l1, pos);
//     if (h1 == h0) return rho0;
//     float t = (delta - h0) / (h1 - h0);
//     return (1 - t) * rho0 + t * rho1;
// }

// // Update compute_cube_map_for_light to use the variable filter size and filtered density
// void LGH::compute_cube_map_for_light(int level, int light_idx, const Vector3f& light_pos, float r_e, float h0, const nanovdb::FloatGrid* density_grid) {
//     int resolution = static_cast<int>(std::ceil(2 * r_e / h0));
//     printf("\nCreating cube map for level %d, light %d:\n", level, light_idx);
//     printf("  Light position: (%f, %f, %f)\n", light_pos.x, light_pos.y, light_pos.z);
//     printf("  Effective radius (r_e): %f\n", r_e);
//     printf("  Base voxel size (h0): %f\n", h0);
//     printf("  Resolution: %d x %d\n", resolution, resolution);
    
//     CubeMap cmap;
//     cmap.resolution = resolution;
//     cmap.r_e = r_e;
//     cmap.h = h0;
    
//     // Calculate s_{l,i} based on the level and light properties
//     float s_li = h[level]; // Use the grid spacing at this level as the base filter size
//     printf("  Filter size s_li: %f\n", s_li);
    
//     for (int face = 0; face < 6; ++face) {
//         printf("  Computing face %d\n", face);
//         cmap.faces[face].resize(resolution, std::vector<float>(resolution, 0.0f));
//         float max_shadow = 0.0f;
//         float min_shadow = 1e10f;
        
//         for (int u = 0; u < resolution; ++u) {
//             for (int v = 0; v < resolution; ++v) {
//                 float s = 2.0f * (u + 0.5f) / resolution - 1.0f;
//                 float t = 2.0f * (v + 0.5f) / resolution - 1.0f;
//                 Vector3f x = cube_face_dirs[face][0];
//                 Vector3f y = cube_face_dirs[face][1];
//                 Vector3f z = cube_face_dirs[face][2];
//                 Vector3f dir = (x * s + y * t + z).abs();
//                 dir = dir / std::sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
//                 float shadow = 0.0f;
//                 float step = h0;
                
//                 // Use variable filter size based on distance from light
//                 for (float d = 0; d < 2 * r_e; d += step) {
//                     Vector3f p = light_pos + dir * d;
//                     float delta = 0.0f;
//                     if (d <= r_e) {
//                         delta = (1 - d / r_e) * s_li + (d / r_e) * h0;
//                     } else {
//                         // Beyond r_e, use the base filter size
//                         delta = h0;
//                     }
                    
//                     // Get filtered density using the variable filter size
//                     float filtered_density = filtered_density_with_filter(p, delta);
//                     shadow += filtered_density * step;
//                      shadow = shadow * exp(-filtered_density * step);
//                     // shadow = 0.5;
//                 }
//                 cmap.faces[face][u][v] = shadow;
//                 max_shadow = std::max(max_shadow, shadow);
//                 min_shadow = std::min(min_shadow, shadow);
//             }
//         }
//         printf("    Face %d shadow range: [%f, %f]\n", face, min_shadow, max_shadow);
//     }
    
//     if (shadow_cube_maps.size() <= level) shadow_cube_maps.resize(level+1);
//     if (shadow_cube_maps[level].size() <= light_idx) shadow_cube_maps[level].resize(light_idx+1);
//     shadow_cube_maps[level][light_idx] = std::move(cmap);
    
//     int total_texels = 6 * resolution * resolution;
//     float expected_texels = 24.0f * (r_e / h0) * (r_e / h0);
//     printf("Cube map: %d texels (expected: %.1f)\n", total_texels, expected_texels);
//     int c=0;

// //     for (int face = 0; face < 6; ++face) {
// //     if (c>=1) {
// //         break;
// //     }
// //     c++;
// //     std::ostringstream oss;
// //     oss << "shadowmap_L" << level << "_light" << light_idx << "_face" << face << ".pgm";
// //     save_cube_map_face_as_pgm(cmap.faces[face], oss.str());
// // }
// }

// float LGH::lookup_shadow(int level, int light_idx, const Vector3f& light_pos, const Vector3f& target_pos) const {
//     // Check if shadow_cube_maps is properly initialized
//     if (shadow_cube_maps.empty() || level >= shadow_cube_maps.size()) {
//         return 0.0f;
//     }
    
//     if (light_idx >= shadow_cube_maps[level].size()) {
//         return 0.0f;
//     }

//     // Get direction from light to target
//     Vector3f dir = target_pos - light_pos;
//     float dist = std::sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
//     if (dist < 1e-6f) {
//         return 0.0f;
//     }
//     dir = dir / dist;  // Normalize

//     // Find the face with the largest absolute component
//     int face = 0;
//     float max_comp = std::abs(dir.x);
//     if (std::abs(dir.y) > max_comp) {
//         max_comp = std::abs(dir.y);
//         face = 2;
//     }
//     if (std::abs(dir.z) > max_comp) {
//         max_comp = std::abs(dir.z);
//         face = 4;
//     }
//     // Adjust face index based on sign
//     if (dir[face/2] < 0) face++;

//     // Get the cube map for this level and light
//     const CubeMap& cmap = shadow_cube_maps[level][light_idx];
//     if (face >= 6 || face < 0) {
//         return 0.0f;
//     }
//     const auto& face_data = cmap.faces[face];

//     // Convert direction to face-local coordinates
//     Vector3f x = cube_face_dirs[face][0];
//     Vector3f y = cube_face_dirs[face][1];
//     Vector3f z = cube_face_dirs[face][2];
    
//     // Project direction onto face basis
//     float s = (dir.x*x.x + dir.y*x.y + dir.z*x.z) / (dir.x*z.x + dir.y*z.y + dir.z*z.z);
//     float t = (dir.x*y.x + dir.y*y.y + dir.z*y.z) / (dir.x*z.x + dir.y*z.y + dir.z*z.z);
    
//     // Convert to pixel coordinates
//     float u = (s + 1.0f) * 0.5f * (cmap.resolution - 1);
//     float v = (t + 1.0f) * 0.5f * (cmap.resolution - 1);
    
//     // Clamp coordinates to valid range
//     u = std::max(0.0f, std::min(u, static_cast<float>(cmap.resolution - 1)));
//     v = std::max(0.0f, std::min(v, static_cast<float>(cmap.resolution - 1)));
    
//     // Get the nearest pixel value (no interpolation)
//     int u_idx = static_cast<int>(std::round(u));
//     int v_idx = static_cast<int>(std::round(v));
    
//     // Ensure indices are within bounds
//     u_idx = std::min(std::max(0, u_idx), cmap.resolution - 1);
//     v_idx = std::min(std::max(0, v_idx), cmap.resolution - 1);
    
//     // Return the shadow value from the specific face
//     //return 0.5; 
//     return face_data[u_idx][v_idx];
// }`

