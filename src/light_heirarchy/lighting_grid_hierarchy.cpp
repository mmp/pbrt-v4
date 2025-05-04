#include "lighting_grid_hierarchy.h"
#include "nanovdb/util/SampleFromVoxels.h"
#include "pbrt/util/spectrum.h"
#include <stdio.h>

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

    // Create rest of S_l
    for (int i=1; i<=depth; i++) {
        deriveNewS(i);
    }

    if (lighting_grids.size() != l_max + 1) {
        LOG_FATAL("Invalid number of grids");
    }
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

                // TODO: is the worldBBox in world space including the transform or without??
                pbrt::Point3f p = medium_transform.ApplyInverse(pbrt::Point3f(x,y,z));


                nanovdb::Vec3f pIndex = temperature_grid->worldToIndexF(nanovdb::Vec3f(x, y, z));
                // TODO: actually do weighted sample for more accurate temperature

                using Sampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
                float temperature = Sampler(temperature_grid->tree())(pIndex);//(pbrt::Point3f(x,y,z));

                if (temperature > 0) {//TEMP_THRESHOLD) {
                    // TODO: derive intensity from temperature
                    // printf("S0 light: %f, pos: %f %f %f\n", temperature, x, y, z);
                    lights.emplace_back(Vector3f(x,y,z), temperature);
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
    std::vector<std::pair<Vector3f,float>> lights;

    // iterate over grid vertices of level l
    for (float x = BBoxMin.x; x < BBoxMax.x; x += h_l)
    for (float y = BBoxMin.y; y < BBoxMax.y; y += h_l)
    for (float z = BBoxMin.z; z < BBoxMax.z; z += h_l)
    {
        Vector3f target_light_pos(x,y,z);             // grid-vertex position q_i
    
        const float h_l     = this->h[l];
        KDTree* lighting_grid_j = lighting_grids[l-1];
    
        // TODO: double check is radius is correct for 2x2x2 grid, used diagonal of cube
        // I SEE THE PROBLEM: currently position based on illumination centers not vertices (so not 2x2x2 grid!)
        // Should the estimated lighting determine radius based on vertex position or illumination centers? Because to do radius search in this part, we need to use vertex positions

        // TODO: either the radius I gave or the radiusSearch function is broken
        float radius = sqrt(3) * h[l-1];
        std::vector<KDNode*> j_lights;
        lighting_grid_j->radiusSearch(target_light_pos, radius , j_lights);

        // printf("radius search lights: %lu\n", j_lights.size());

        float     I  = calcNewI (l, target_light_pos, j_lights);          // Eq (1)
        Vector3f  p  = calcNewPos(l, target_light_pos, j_lights);         // Eq (2)
        if (I>0.f) lights.emplace_back(p,I);
    }

    lighting_grids[l] = new KDTree(lights);
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

float LGH::get_intensity(int L, Vector3f targetPos, KDNode* light, float radius)
{
    // TODO: get Transmission function somehow
    // V = Tr(x, light->point)

    // Calculate tranmittance

    // TODO: create ray, rng, and lambda
    // SampledSpectrum V = SampleT_maj(ray, tMax, rng.Uniform<Float>(), rng, lambda,
    //             [&](Point3f p, MediumProperties mp, SampledSpectrum sigma_maj, SampledSpectrum T_maj) {
    //     return true;
    //             });


    float d = targetPos.distance(light->point);
    float g = 1.f / pow(d,2); // Light fall-off
    float B = blendingFunction(L, d, radius);

    if (B < 0) {
        LOG_FATAL("Blending function should never be negative! %f", B);
    }

    if (light->intensity < 0) {
        printf("LIGHT INTENSITY NEGATIVE!!! L: %d, %f\n", L, light->intensity);
    }

    return g * B * light->intensity; // * V
}

// TODO: replace Le in pbrt
pbrt::SampledSpectrum LGH::get_total_illum(pbrt::Point3f pos, pbrt::SampledWavelengths lambda)
{

    // Convert point from world space to grid space
    pos = medium_transform.ApplyInverse(pos);


    // temp = (temp - temperatureOffset) * temperatureScale;
    // if (temp <= 100.f)
    //     return SampledSpectrum(0.f);
    // return LeScale * BlackbodySpectrum(temp).Sample(lambda);

    // nanovdb::Vec3f pIndex = m_temperature_grid->worldToIndexF(nanovdb::Vec3f(p.x, p.y, p.z));
    // TODO: actually do weighted sample for more accurate temperature
    // Vector3f v_pos(pIndex[0], pIndex[1], pIndex[2]);

    Vector3f v_pos(pos.x, pos.y, pos.z);

    int numLightsCaptured = 0;
    int numLightsS0;
    float total_intensity = 0;
    for (int l=0; l<=l_max; l++) {
        float radius = alpha * h[l];

        std::vector<KDNode*> results;
        lighting_grids[l]->radiusSearch(v_pos, radius, results);
        numLightsCaptured += results.size();

        if (l == 0)
            numLightsS0 = results.size();
        //printf("  radius search size: %lu\n", results.size());
        
        for (auto light : results) {
            total_intensity += get_intensity(l, v_pos, light, radius);
        }
    }

    printf("  captured lights %d, S0: %d\n", numLightsCaptured, numLightsS0);

    // TODO: most total_intensity values seem to be 0 and idk why, could be because of radius search?

    // total_intensity *= 4500 * 5;// * 10000;
    // printf("tota")

    // total_intensity = std::clamp(total_intensity, 1000.f, 10000.f);

    if (total_intensity < 0) {
        LOG_FATAL("NEGATIVE INTENSITY!!! %f\n", total_intensity);
        return pbrt::SampledSpectrum(0.f);
    }

    if (total_intensity == 0) {
        return pbrt::SampledSpectrum(0.f);
    }
    // else if (total_intensity < 100) {
    //     return pbrt::SampledSpectrum(0.f);
    // }

    // printf("\tIntensity: %f, point %f %f %f\t\n", total_intensity, v_pos.x, v_pos.y, v_pos.z);

    return 0.125 * pbrt::BlackbodySpectrum(total_intensity).Sample(lambda);

    // return total_intensity;
}
