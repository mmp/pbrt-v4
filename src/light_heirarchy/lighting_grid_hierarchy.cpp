#include "lighting_grid_hierarchy.h"
#include "nanovdb/util/SampleFromVoxels.h"
#include "pbrt/util/spectrum.h"
#include <stdio.h>
#include <pbrt/media.h>
#include <cmath>

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

// pbrt::SampledSpectrum
float LGH::get_intensity(int L,
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

    pbrt::SampledSpectrum V(1.f);

    //     //pbrt::SampledSpectrum V =
    // SampleT_maj(ray, tMax, rng.Uniform<float>(), rng, lambda,
    //     [&](pbrt::Point3f p, pbrt::MediumProperties mp, pbrt::SampledSpectrum sigma_maj, pbrt::SampledSpectrum T_maj) {
    //         // printf("sigma_maj: %s, t_maj: %s, a: %s, s: %s\n", sigma_maj.ToString().c_str(), T_maj.ToString().c_str(), mp.sigma_a.ToString().c_str(), mp.sigma_s.ToString().c_str());
    //         // Null-collision transmittance estimation

    //         p = ray.o;
    //         while (Length(p-ray.o) < light->point.distance(targetPos)) {
    //             // Extinction: sigma_t = sigma_a + sigma_s
    //             pbrt::SampledSpectrum sigma_t = mp.sigma_a + mp.sigma_s;

    //             // // Null-collision cross section
    //             // pbrt::SampledSpectrum sigma_n = ClampZero(sigma_maj - sigma_t);

    //             // // Probability of null collision (used to weight the sample)
    //             // float pr = T_maj[0] * sigma_maj[0];  // scalar for importance sampling weight

    //             // // Transmittance update for null collision
    //             // Tr *= T_maj * sigma_n / pr;

    //             const float ds = 0.03f;

    //             // // March towards light to calculate occlusion with larger steps for less shadowing
    //             float shadow_ds = ds * 1.5f;

    //             nanovdb::Vec3f pIndex = m_density_grid->worldToIndexF(nanovdb::Vec3f(p.x, p.y, p.z));
    //             // TODO: actually do weighted sample for more accurate temperature

    //             using Sampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
    //             float rs = Sampler(m_density_grid->tree())(pIndex);

    //             //     // Apply lighter non-linear density mapping for less dramatic shadows
    //             //     rs = pow(rs, 1.2f) * 0.8f;
    //             // V *= std::exp(-sigma_t.y(lambda) * rs * shadow_ds);
    //             V *= std::exp(-4.5 * rs * shadow_ds);

    //             // if (V.MaxComponentValue() != 1)
    //             // printf("Transmission: %s, trMax:%f\n", V.ToString().c_str(), V.MaxComponentValue());

    //             // // Optional: early termination if weight is too low
    //             if (V.MaxComponentValue() < 1e-3f) {
    //                 // printf("     \nSTOPPINGGG Transmission: %s, tMax:%f\n\n", Tr.ToString().c_str(), tMax);
    //                 return false; // stop sampling
    //             }

    //             p += ds * ray.d;
    //         }

    //         return false; // keep sampling
    //     });//.y(lambda);

    Vector3f p = targetPos;
    while (p.distance(targetPos) < light->point.distance(targetPos)) {
        // Extinction: sigma_t = sigma_a + sigma_s

        // TODO: NOTE THIS VALUE IS UNTESTED. Needs to be greater than 80 at least
        float sigma_t = 400;//4.5f * 20;

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

        p += shadow_ds * dir;
    }

        // Tr_sum += V;
    // }
    // pbrt::SampledSpectrum V = Tr_sum / float(nSamples);

    // float V = 1;


    // if (V.MaxComponentValue() != 1)
        // printf("transmittance: %s, tMax: %f\n", V.ToString().c_str(), tMax);


    float d = targetPos.distance(light->point);

    // TODO: figure out good scale for light fall-off, make sure it is relative to size of explosion
    float g = 1.f / (1 + pow(d,2)); // Light fall-off, good with d * 50
    g = std::min(1.f, g);

    float B = blendingFunction(L, d, radius);

    if (B < 0) {
        LOG_FATAL("Blending function should never be negative! %f", B);
    }

    if (light->intensity < 0) {
        printf("LIGHT INTENSITY NEGATIVE!!! L: %d, %f\n", L, light->intensity);
    }

    if (light->intensity * 1000 < 10) {
        // printf("Too low light intensity L: %d, %f\n", L, light->intensity * 4500);
        return pbrt::SampledSpectrum(0);
    }

    // printf("Light intensity: %f, transmittance: %s, g: %f, B: %f\n", light->intensity * 1000, V.ToString().c_str(), g, B);


    // TODO NOTE * 200 untested. No idea what value it should be
    return g * B * pbrt::BlackbodySpectrum(light->intensity * 1000).Sample(lambda) * V * 200; //light->intensity * V;//
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

    for (int l=0; l<=l_max; l++) {
        float radius = alpha * h[l];
        std::vector<KDNode*> results;
        lighting_grids[l]->radiusSearch(v_pos, radius, results);
        numLightsCaptured += results.size();
        
        for (auto light : results) {
            float intensity = get_intensity(l, v_pos, light, radius, lambda, sampler, medium);
            if (std::isfinite(intensity)) {  // Check for NaN and infinity
                total_intensity += intensity;
            }
        }
    }

    return 0.125 * total_intensity; //pbrt::BlackbodySpectrum(total_intensity).Sample(lambda);
}
