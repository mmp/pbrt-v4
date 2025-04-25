#include "lighting_grid_hierarchy.h"

LGH::LGH(pbrt::SampledGrid<float> temperature_grid, int depth, float base_voxel_size) 
    : l_max(depth)
{
    XSize = temperature_grid.XSize();
    YSize = temperature_grid.YSize();
    ZSize = temperature_grid.ZSize();

    // Initialize h
    for (int i=0; i<depth; i++) {
        h.push_back(base_voxel_size * pow(2,i));
    }

    // Initialize S0
    create_S0(temperature_grid);

    // Create rest of S_l
    for (int i=1; i<depth; i++) {
        deriveNewS(i);
    }

    if (h.size() != l_max + 1) {
        LOG_FATAL("invalid number of h!!!");
    }
    if (lighting_grids.size() != l_max + 1) {
        LOG_FATAL("Invalid number of grids");
    }
}


void LGH::create_S0(pbrt::SampledGrid<float> temperature_grid)
{
    float h_0 = this->h[0];

    std::vector<std::pair<Vector3f,float>> lights;
    for (float x = h_0; x < XSize; x += h_0) {
        for (float y = h_0; y < YSize; y += h_0) {
            for (float z = h_0; z < ZSize; z += h_0) {
                float temperature = temperature_grid.Lookup(pbrt::Point3f(x,y,z));
                if (temperature < TEMP_THRESHOLD) {
                    // TODO: derive intensity from temperature
                    lights.emplace_back(Vector3f(x,y,z), temperature);
                }
            }
        }
    }

    this->lighting_grids[0] = &KDTree(lights);
}

void LGH::deriveNewS(int l)//, const KDTree& S0, const pbrt::SampledGrid<float>& tempGrid)
{
    if (l == 0) {
        LOG_FATAL("Level should never be 0!");
        return;
    }
    float h_l = this->h[l];                     // spacing of level l
    std::vector<std::pair<Vector3f,float>> lights;

    // iterate over grid vertices of level l
    for (float x = .0f; x < XSize; x += h_l)
    for (float y = .0f; y < YSize; y += h_l)
    for (float z = .0f; z < ZSize; z += h_l)
    {
        Vector3f target_light_pos(x,y,z);             // grid-vertex position q_i
    
        const float h_l     = this->h[l];
        KDTree* lighting_grid_j = lighting_grids[l-1];
    
        // TODO: double check is radius is correct for 2x2x2 grid, used diagonal of cube
        float radius = sqrt(3) * h_l; 
        std::vector<KDNode*> j_lights;
        lighting_grid_j->radiusSearch(target_light_pos, radius , j_lights);

        float     I  = calcNewI (l, target_light_pos, j_lights);          // Eq (1)
        Vector3f  p  = calcNewPos(l, target_light_pos, j_lights);         // Eq (2)
        if (I>0.f) lights.emplace_back(p,I);
    }

    lighting_grids[l] = &KDTree(lights);
}


// axis-aligned bounding box for 2×2×2 block around a grid‐vertex
// static inline AABB makeCellBlock(const Vector3f& gv, float h)
// {
//     return AABB(gv-Vector3f(h/2), gv+Vector3f(h/2));
// }

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


    // const float inv_h   = 1.f / h_l;
    // float I_sum = 0.f;

    // std::vector<LightRecord> cand;      // pos & intensity of S0 lights
    // S0.rangeQuery(makeCellBlock(gv,h_l), cand);

    // for (const auto& lt : cand)
    // {
    //     Vector3f d = (lt.pos - gv) * inv_h;       // normalised offset
    //     float w = std::max(0.f,1.f-std::abs(d.x)) *
    //               std::max(0.f,1.f-std::abs(d.y)) *
    //               std::max(0.f,1.f-std::abs(d.z));
    //     I_sum += w * lt.I;                        // Eq (1)
    // }
    // return I_sum;
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

    // const float h_l   = this->h[l];
    // const float inv_h = 1.f / h_l;
    // Vector3f num(0.f);     // ∑ w·I·p
    // float    denom = 0.f;  // ∑ w·I_L

    // std::vector<LightRecord> cand;
    // S0.rangeQuery(makeCellBlock(gv,h_l), cand);

    // for (const auto& lt : cand)
    // {
    //     Vector3f d = (lt.pos - gv) * inv_h;
    //     float w = std::max(0.f,1.f-std::abs(d.x)) *
    //               std::max(0.f,1.f-std::abs(d.y)) *
    //               std::max(0.f,1.f-std::abs(d.z));

    //     float omega = w * lt.I;          // using scalar intensity as luminance
    //     num   += lt.pos * omega;
    //     denom += omega;
    // }
    // return (denom>0.f) ? num/denom : gv;        // Eq (2)
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
    float d = targetPos.distance(light->point);
    float g = 1.f / pow(d,2); // Light fall-off
    float B = blendingFunction(L, d, radius);
    return g * B * light->intensity; // * V
}

// TODO: replace Le in pbrt
float LGH::get_total_illum(Vector3f pos)
{
    float total_intensity = 0;
    for (int l=0; l<l_max; l++) {
        float radius = alpha * h[l];

        std::vector<KDNode*> results;
        lighting_grids[l]->radiusSearch(pos, radius, results);
        
        for (auto light : results) {
            total_intensity += get_intensity(l, pos, light, radius);
        }
    }

    return total_intensity;
}
