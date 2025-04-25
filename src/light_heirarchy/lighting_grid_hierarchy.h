#ifndef LIGHTING_GRID_HIERARCHY_H
#define LIGHTING_GRID_HIERARCHY_H

#include "kdtree3d.h"
#include "pbrt/util/containers.h"
#include "pbrt/pbrt.h"

class LGH
{
public:
    // struct AABB
    // {
    //     Vector3f min;   // inclusive
    //     Vector3f max;   // inclusive

    //     AABB() = default;
    //     AABB(const Vector3f& mi, const Vector3f& ma) : min(mi), max(ma) {}

    //     // point-inside test (handy if you ever need it)
    //     bool contains(const Vector3f& p) const
    //     {
    //         return (p.x >= min.x && p.x <= max.x) &&
    //             (p.y >= min.y && p.y <= max.y) &&
    //             (p.z >= min.z && p.z <= max.z);
    //     }
    // };

    // TODO: fix inputs
    LGH(pbrt::SampledGrid<float> temperature_grid, int depth, float base_voxel_size);

    float get_intensity(int L, Vector3f targetPos, KDNode* light, float radius);
    float get_total_illum(Vector3f pos);

    const float TEMP_THRESHOLD = 1.0f;


    // static void extract_lights(pbrt::SampledGrid<float> temperature_grid);

    const float alpha = 1.0f;
    const int l_max;

private:
    void create_S0(pbrt::SampledGrid<float> temperature_grid);
    void deriveNewS(int l);//, KDTree S0);
    Vector3f calcNewPos(int l, Vector3f target_light_pos, std::vector<KDNode*> j_lights);//const Vector3f& gv, int l, const KDTree& S0) const;
    float calcNewI(int l, Vector3f target_light_pos, std::vector<KDNode*> j_lights); //const Vector3f& gv, int l, const KDTree& S0) const;

    float blendingFunction(int level, float distance, float r_l);

    std::vector<KDTree*> lighting_grids;
    std::vector<float> h;

    float XSize;
    float YSize;
    float ZSize;
};

#endif // LIGHTING_GRID_HIERARCHY_H
