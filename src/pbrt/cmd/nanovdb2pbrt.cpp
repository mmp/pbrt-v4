// nanovdb2pbrt.cpp
// pbrt is Copyright(c) 1998-2023 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/media.h>
#include <pbrt/util/args.h>

#include <nanovdb/NanoVDB.h>
#define NANOVDB_USE_ZIP 1
#include <nanovdb/util/IO.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/SampleFromVoxels.h>

#include <stdio.h>
#include <vector>

using namespace pbrt;

template <typename Buffer>
static nanovdb::GridHandle<Buffer> readGrid(const std::string &filename,
                                            const std::string &gridName,
                                            Allocator alloc) {
    NanoVDBBuffer buf(alloc);
    nanovdb::GridHandle<Buffer> grid;
    try {
        grid =
            nanovdb::io::readGrid<Buffer>(filename, gridName, 0 /* not verbose */, buf);
    } catch (const std::exception &e) {
        ErrorExit("nanovdb: %s: %s", filename, e.what());
    }

    if (grid) {
        if (!grid.gridMetaData()->isFogVolume() && !grid.gridMetaData()->isUnknown())
            ErrorExit("%s: \"%s\" isn't a FogVolume grid?", filename, gridName);

        LOG_VERBOSE("%s: found %d \"%s\" voxels", filename,
                    grid.gridMetaData()->activeVoxelCount(), gridName);
    }

    return grid;
}

static void usage(const std::string &msg = {}) {
    if (!msg.empty())
        fprintf(stderr, "nanovdb2pbrt: %s\n\n", msg.c_str());

    fprintf(stderr,
            R"(usage: nanovdb2pbrt [<options>] <filename.nvdb>

Options:
  --grid <name>        Name of grid to extract. Default: "density"
)");
    exit(msg.empty() ? 0 : 1);
}

int main(int argc, char *argv[]) {
    std::vector<std::string> args = GetCommandLineArguments(argv);

    auto onError = [](const std::string &err) {
        usage(err);
        exit(1);
    };

    std::string filename;
    std::string grid = "density";
    int downsample = 0;
    for (auto iter = args.begin(); iter != args.end(); ++iter) {
        if ((*iter)[0] != '-') {
            if (filename.empty())
                filename = *iter;
            else {
                usage();
                exit(1);
            }
        } else if (ParseArg(&iter, args.end(), "downsample", &downsample, onError) ||
                   ParseArg(&iter, args.end(), "grid", &grid, onError)) {
            // success
        } else {
            usage();
            exit(1);
        }
    }

    if (filename.empty())
        usage("must specify a nanovdb filename");

    Allocator alloc;
    nanovdb::GridHandle<NanoVDBBuffer> nanoGrid =
        readGrid<NanoVDBBuffer>(filename, grid, alloc);
    if (!nanoGrid)
        ErrorExit("%s: didn't find \"%s\" grid.", filename, grid);
    const nanovdb::FloatGrid *floatGrid = nanoGrid.grid<float>();

    nanovdb::BBox<nanovdb::Vec3R> bbox = floatGrid->worldBBox();
    Bounds3f bounds(Point3f(bbox.min()[0], bbox.min()[1], bbox.min()[2]),
                    Point3f(bbox.max()[0], bbox.max()[1], bbox.max()[2]));

    std::vector<Float> values;

    int x0 = floatGrid->indexBBox().min()[0], x1 = floatGrid->indexBBox().max()[0]+1;
    int y0 = floatGrid->indexBBox().min()[1], y1 = floatGrid->indexBBox().max()[1]+1;
    int z0 = floatGrid->indexBBox().min()[2], z1 = floatGrid->indexBBox().max()[2]+1;

    for (int z = z0; z <= z1; ++z)
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                values.push_back(floatGrid->tree().getValue({x, y, z}));
            }

    printf("\"integer nx\" %d \"integer ny\" %d  \"integer nz\" %d\n", 1+x1-x0, 1+y1-y0, 1+z1-z0);
    printf("\t\"point3 p0\" [ %f %f %f ] \"point3 p1\" [ %f %f %f ]\n",
           bounds.pMin.x, bounds.pMin.y, bounds.pMin.z,
           bounds.pMax.x, bounds.pMax.y, bounds.pMax.z);
    printf("\t\"float %s\" [\n", grid.c_str());
    for (int i = 0; i < values.size(); ++i) {
        Float d = values[i];
        if (d == 0) printf("0 ");
        else printf("%f ", d);
        if ((i % 20) == 19) printf("\n");
    }
    printf("]\n");
}


