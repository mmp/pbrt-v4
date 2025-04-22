#ifndef LIGHTING_GRID_HIERARCHY_H
#define LIGHTING_GRID_HIERARCHY_H

#include "kdtree3d.h"
#include "pbrt/util/containers.h"

class lighting_grid_hierarchy
{
public:
    lighting_grid_hierarchy(pbrt::SampledGrid<float> temperature_grid, int depth, float base_voxel_size);
    static void extract_lights(pbrt::SampledGrid<float> temperature_grid);
    std::vector<KDTree> lighting_grids;
    std::vector<float> h;
    // const float alpha;

private:
    void light_contribution(int level, Vector3f x);
};

#endif // LIGHTING_GRID_HIERARCHY_H
