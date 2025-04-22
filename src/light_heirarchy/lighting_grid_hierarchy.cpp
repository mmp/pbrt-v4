#include "lighting_grid_hierarchy.h"

LGH::LGH(pbrt::SampledGrid<float> temperature_grid, int depth, float base_voxel_size)
{
    XSize = temperature_grid.XSize();
    YSize = temperature_grid.YSize();
    ZSize = temperature_grid.ZSize();
    maxDepth = depth;

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

    if (h.size() != maxDepth + 1) {
        LOG_FATAL("invalid number of h!!!");
    }
    if (lighting_grids.size() != maxDepth + 1) {
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
                    lights.push_back(std::make_pair(Vector3f(x,y,z), temperature));
                }
            }
        }
    }

    this->lighting_grids[0] = KDTree(lights);
}

void LGH::deriveNewS(int l, const KDTree& S0, const pbrt::SampledGrid<float>& tempGrid)
{
    float h_l = this->h[l];                     // spacing of level l
    std::vector<std::pair<Vector3f,float>> lights;

    // iterate over grid vertices of level l
    for (float x = .0f; x < tempGrid.XSize(); x += h_l)
    for (float y = .0f; y < tempGrid.YSize(); y += h_l)
    for (float z = .0f; z < tempGrid.ZSize(); z += h_l)
    {
        Vector3f gv(x,y,z);             // grid-vertex position q_i
        float     I  = calcNewI (gv,l,S0);          // Eq (1)
        Vector3f  p  = calcNewPos(gv,l,S0);         // Eq (2)
        if (I>0.f) lights.emplace_back(p,I);
    }

    lighting_grids[l] = KDTree(lights);
}


// axis-aligned bounding box for 2×2×2 block around a grid‐vertex
static inline AABB makeCellBlock(const Vector3f& gv, float h)
{
    return AABB(gv-Vector3f(h/2), gv+Vector3f(h/2));
}

// -------- intensity (Eq 1) ----------------------------------------------
float LGH::calcNewI(const Vector3f& gv, int l, const KDTree& S0) const
{
    const float h_l     = this->h[l];
    const float inv_h   = 1.f / h_l;
    float I_sum = 0.f;

    std::vector<LightRecord> cand;      // pos & intensity of S0 lights
    S0.rangeQuery(makeCellBlock(gv,h_l), cand);

    for (const auto& lt : cand)
    {
        Vector3f d = (lt.pos - gv) * inv_h;       // normalised offset
        float w = std::max(0.f,1.f-std::abs(d.x)) *
                  std::max(0.f,1.f-std::abs(d.y)) *
                  std::max(0.f,1.f-std::abs(d.z));
        I_sum += w * lt.I;                        // Eq (1)
    }
    return I_sum;
}

// -------- illumination centre (Eq 2) ------------------------------------
Vector3f LGH::calcNewPos(const Vector3f& gv, int l, const KDTree& S0) const
{
    const float h_l   = this->h[l];
    const float inv_h = 1.f / h_l;
    Vector3f num(0.f);     // ∑ w·I·p
    float    denom = 0.f;  // ∑ w·I_L

    std::vector<LightRecord> cand;
    S0.rangeQuery(makeCellBlock(gv,h_l), cand);

    for (const auto& lt : cand)
    {
        Vector3f d = (lt.pos - gv) * inv_h;
        float w = std::max(0.f,1.f-std::abs(d.x)) *
                  std::max(0.f,1.f-std::abs(d.y)) *
                  std::max(0.f,1.f-std::abs(d.z));

        float omega = w * lt.I;          // using scalar intensity as luminance
        num   += lt.pos * omega;
        denom += omega;
    }
    return (denom>0.f) ? num/denom : gv;        // Eq (2)
}


float LGH::get_intensity(int L, Vector3f lightPos)
{
    return 0.0f;
}

float LGH::get_total_illum(Vector3f pos)
{
    float total_intensity = 0;
    for (int l=0; l<maxDepth; l++) {
        float radius = alpha * h[l];

        std::vector<KDNode*> results;
        lighting_grids[l].radiusSearch(pos, radius, results);
        
        for (auto light : results) {
            total_intensity += get_intensity(l, light->point);
        }
    }

    return total_intensity;
}
