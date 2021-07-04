// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/options.h>
#include <pbrt/pbrt.h>
#include <pbrt/util/args.h>
#include <pbrt/util/file.h>
#include <pbrt/util/mesh.h>
#include <pbrt/util/string.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>
#include <stdlib.h>

#include <cstdarg>
#include <map>
#include <string>
#include <vector>

using namespace pbrt;

static void usage(const char *msg = nullptr, ...) {
    if (msg != nullptr) {
        va_list args;
        va_start(args, msg);
        fprintf(stderr, "splitply: ");
        vfprintf(stderr, msg, args);
        fprintf(stderr, "\n\n");
    }

    printf(R"(splitply splits PLY format files into multiple PLY files so that
none of the resulting files has more than a specified number of faces.
If the provided PLY file has fewer than the number of faces, then no output
is generated.

usage: splitply [source.ply] [<options>]

Options:
  --maxfaces <n>    Maximum number of faces in an output PLY file.
                    (Default: 1000000)
  --outbase <name>  Base name for emitted PLY files.  Consecutive numbers
                    and a ".ply" suffix will be appended to <name>.
                    (Default: based on <source.ply>.)
)");

    CleanupPBRT();

    exit(1);
}

int main(int argc, char *argv[]) {
    InitPBRT(PBRTOptions());

    std::vector<std::string> args = GetCommandLineArguments(argv);

    std::string inPLY, outPLYBase;
    int maxFaces = 1000000;
    for (auto iter = args.begin(); iter != args.end(); ++iter) {
        auto onError = [](const std::string &err) {
            usage("%s", err.c_str());
            exit(1);
        };
        if (ParseArg(&iter, args.end(), "maxfaces", &maxFaces, onError) ||
            ParseArg(&iter, args.end(), "outbase", &outPLYBase, onError))
            ;  // yaay
        else if (inPLY.empty())
            inPLY = *iter;
        else
            usage("unexpected argument \"%s\"", iter->c_str());
    }

    if (inPLY.empty()) usage("must specify source PLY filename.");

    if (outPLYBase.empty()) outPLYBase = RemoveExtension(inPLY);

    TriQuadMesh mesh = TriQuadMesh::ReadPLY(inPLY);

    if (mesh.quadIndices.size() > 0) {
        fprintf(stderr,
                "%s: sorry, mesh has quad faces. splitply currently only "
                "supports triangle meshes.\n",
                inPLY.c_str());
        CleanupPBRT();
        return 1;
    }
    if (mesh.faceIndices.size() > 0) {
        fprintf(stderr,
                "%s: sorry, mesh has faceIndices, which are not currently "
                "supported by splitply.\n",
                inPLY.c_str());
        CleanupPBRT();
        return 1;
    }

    int nFaces = mesh.triIndices.size() / 3;
    if (nFaces <= maxFaces) {
        fprintf(stderr, "%s: mesh has %d faces and so has not been split up.\n",
                inPLY.c_str(), nFaces);
        CleanupPBRT();
        return 0;
    }

    int nMeshes = (nFaces + maxFaces - 1) / maxFaces;
    fprintf(stderr, "%s: mesh has %d faces and will be split into %d meshes.\n",
            inPLY.c_str(), nFaces, nMeshes);

    int nFacesPerMesh = nFaces / nMeshes;  // more or less...
    for (int i = 0; i < nMeshes; ++i) {
        int firstFaceIndex = i * nFacesPerMesh;
        int lastFaceIndex = (i + 1) * nFacesPerMesh;
        if (i == nMeshes - 1) lastFaceIndex = nFaces;

        std::map<int, int> vertexIndexRemap;
        std::vector<int> indices;
        std::vector<Point3f> p;
        std::vector<Normal3f> n;
        std::vector<Point2f> uv;
        for (int vertexIndex = 3 * firstFaceIndex;
             vertexIndex < 3 * lastFaceIndex; ++vertexIndex) {
            int index = mesh.triIndices[vertexIndex];
            if (auto iter = vertexIndexRemap.find(index); iter != vertexIndexRemap.end())
                indices.push_back(iter->second);
            else {
                int newIndex = int(vertexIndexRemap.size());
                vertexIndexRemap[index] = newIndex;
                indices.push_back(newIndex);

                p.push_back(mesh.p[index]);
                if (!mesh.n.empty()) n.push_back(mesh.n[index]);
                // TODO: there's no "s" in TriQuadMesh???
                if (!mesh.uv.empty()) uv.push_back(mesh.uv[index]);
            }
        }

        TriangleMesh triMesh(Transform(), false /* reverse orientation */,
                             indices, p, std::vector<Vector3f>(), n, uv, std::vector<int>());
        if (!triMesh.WritePLY(StringPrintf("%s-%03d.ply", outPLYBase, i))) {
            CleanupPBRT();
            return 1;
        }
    }

    CleanupPBRT();

    return 0;
}
