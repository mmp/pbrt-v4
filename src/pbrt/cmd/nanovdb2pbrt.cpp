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
  --downsample <n>     Number of times to 2x downsample the volume.
                       Default: 0
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

    int nx = floatGrid->indexBBox().dim()[0];
    int ny = floatGrid->indexBBox().dim()[1];
    int nz = floatGrid->indexBBox().dim()[2];

    std::vector<Float> values;

    int z0 = 0, z1 = nz;
    int y0 = 0, y1 = ny;
    int x0 = 0, x1 = nx;

    // Fix the resolution to be a multiple of 2^downsample just to make
    // downsampling easy. Chop off one at a time from the bottom and top
    // of the range until we get there; the bounding box is updated as
    // well so that the remaining volume doesn't shift spatially.
    auto round = [=](int &low, int &high, Float &c0, Float &c1) {
        Float delta = (c1-c0) / (high-low);
        int mult = 1 << downsample; // want a multiple of this in resolution
        while ((high - low) % mult) {
            ++low;
            c0 += delta;
            if ((high - low) % mult) {
                --high;
                c1 -= delta;
            }
        }
        return high - low;
    };
    nz = round(z0, z1, bounds.pMin.z, bounds.pMax.z);
    ny = round(y0, y1, bounds.pMin.y, bounds.pMax.y);
    nx = round(x0, x1, bounds.pMin.x, bounds.pMax.x);

    for (int z = z0; z < z1; ++z)
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x) {
                values.push_back(floatGrid->tree().getValue({x, y, z}));
            }

    while (downsample > 0) {
        std::vector<Float> v2;
        for (int z = 0; z < nz/2; ++z)
            for (int y = 0; y < ny/2; ++y)
                for (int x = 0; x < nx/2; ++x) {
                    auto v = [&](int dx, int dy, int dz) -> Float{
                        return values[(2*x+dx) + nx * ((2*y+dy) + ny * (2*z+dz))];
                    };
                    v2.push_back((v(0,0,0) + v(1,0,0) + v(0,1,0) + v(1,1,0) +
                                  v(0,0,1) + v(1,0,1) + v(0,1,1) + v(1,1,1))/8);
                }

        values = std::move(v2);
        nx /= 2;
        ny /= 2;
        nz /= 2;
        --downsample;
    }

    printf("\"integer nx\" %d \"integer ny\" %d  \"integer nz\" %d\n", nx, ny, nz);
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


