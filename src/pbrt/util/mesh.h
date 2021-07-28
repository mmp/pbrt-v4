// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_MESH_H
#define PBRT_UTIL_MESH_H

#include <pbrt/pbrt.h>

#include <pbrt/util/containers.h>
#include <pbrt/util/error.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#include <array>
#include <string>
#include <vector>

namespace pbrt {

// TriangleMesh Definition
class TriangleMesh {
  public:
    // TriangleMesh Public Methods
    TriangleMesh(const Transform &renderFromObject, bool reverseOrientation,
                 std::vector<int> vertexIndices, std::vector<Point3f> p,
                 std::vector<Vector3f> S, std::vector<Normal3f> N,
                 std::vector<Point2f> uv, std::vector<int> faceIndices, Allocator alloc);

    std::string ToString() const;

    bool WritePLY(std::string filename) const;

    static void Init(Allocator alloc);

    // TriangleMesh Public Members
    int nTriangles, nVertices;
    const int *vertexIndices = nullptr;
    const Point3f *p = nullptr;
    const Normal3f *n = nullptr;
    const Vector3f *s = nullptr;
    const Point2f *uv = nullptr;
    const int *faceIndices = nullptr;
    bool reverseOrientation, transformSwapsHandedness;
};

// BilinearPatchMesh Definition
class BilinearPatchMesh {
  public:
    // BilinearPatchMesh Public Methods
    BilinearPatchMesh(const Transform &renderFromObject, bool reverseOrientation,
                      std::vector<int> vertexIndices, std::vector<Point3f> p,
                      std::vector<Normal3f> N, std::vector<Point2f> uv,
                      std::vector<int> faceIndices, PiecewiseConstant2D *imageDist,
                      Allocator alloc);

    std::string ToString() const;

    static void Init(Allocator alloc);

    // BilinearPatchMesh Public Members
    bool reverseOrientation, transformSwapsHandedness;
    int nPatches, nVertices;
    const int *vertexIndices = nullptr;
    const Point3f *p = nullptr;
    const Normal3f *n = nullptr;
    const Point2f *uv = nullptr;
    const int *faceIndices = nullptr;
    PiecewiseConstant2D *imageDistribution;
};

// HashIntPair Definition
struct HashIntPair {
    PBRT_CPU_GPU
    size_t operator()(std::pair<int, int> p) const {
        return MixBits(uint64_t(p.first) << 32 | p.second);
    };
};

struct TriQuadMesh {
    // TriQuadMesh Public Methods
    static TriQuadMesh ReadPLY(const std::string &filename);

    void ConvertToOnlyTriangles();
    void ComputeNormals();

    std::string ToString() const;

    template <typename Dist, typename Disp>
    TriQuadMesh Displace(Dist &&dist, Float maxDist, Disp &&displace,
                         const FileLoc *loc = nullptr) const {
        if (uv.empty())
            ErrorExit(loc, "Vertex uvs are currently required by Displace(). Sorry.\n");

        // Prepare the output mesh
        TriQuadMesh outputMesh = *this;
        outputMesh.ConvertToOnlyTriangles();
        if (outputMesh.n.empty())
            outputMesh.ComputeNormals();
        outputMesh.triIndices.clear();

        // Refine
        HashMap<std::pair<int, int>, int, HashIntPair> edgeSplit({});
        for (int i = 0; i < triIndices.size() / 3; ++i)
            outputMesh.Refine(dist, maxDist, triIndices[3 * i], triIndices[3 * i + 1],
                              triIndices[3 * i + 2], edgeSplit);

        // Displace
        displace(outputMesh.p.data(), outputMesh.n.data(), outputMesh.uv.data(),
                 outputMesh.p.size());

        outputMesh.ComputeNormals();

        return outputMesh;
    }

    std::vector<Point3f> p;
    std::vector<Normal3f> n;
    std::vector<Point2f> uv;
    std::vector<int> faceIndices;
    std::vector<int> triIndices, quadIndices;

  private:
    // TriQuadMesh Private Methods
    template <typename Dist>
    void Refine(Dist &&distance, Float maxDist, int v0, int v1, int v2,
                HashMap<std::pair<int, int>, int, HashIntPair> &edgeSplit) {
        Point3f p0 = p[v0], p1 = p[v1], p2 = p[v2];
        Float d01 = distance(p0, p1), d12 = distance(p1, p2), d20 = distance(p2, p0);

        if (d01 < maxDist && d12 < maxDist && d20 < maxDist) {
            triIndices.push_back(v0);
            triIndices.push_back(v1);
            triIndices.push_back(v2);
            return;
        }

        // order so that the first two vertices have the longest edge
        std::array<int, 3> v;
        if (d01 > d12) {
            if (d01 > d20)
                v = {v0, v1, v2};
            else
                v = {v2, v0, v1};
        } else {
            if (d12 > d20)
                v = {v1, v2, v0};
            else
                v = {v2, v0, v1};
        }

        // has the edge been spilt before?
        std::pair<int, int> edge(v[0], v[1]);
        if (v[0] > v[1])
            std::swap(edge.first, edge.second);

        int vmid;
        if (edgeSplit.HasKey(edge)) {
            vmid = edgeSplit[edge];
        } else {
            vmid = p.size();
            edgeSplit.Insert(edge, vmid);
            p.push_back((p[v[0]] + p[v[1]]) / 2);
            if (!n.empty()) {
                Normal3f nn = n[v[0]] + n[v[1]];
                if (LengthSquared(nn) > 0)
                    nn = Normalize(nn);
                n.push_back(nn);
            }
            if (!uv.empty())
                uv.push_back((uv[v[0]] + uv[v[1]]) / 2);
        }

        Refine(distance, maxDist, v[0], vmid, v[2], edgeSplit);
        Refine(distance, maxDist, vmid, v[1], v[2], edgeSplit);
    }
};

bool WritePLY(std::string filename, pstd::span<const int> triIndices,
              pstd::span<const int> quadIndices, pstd::span<const Point3f> p,
              pstd::span<const Normal3f> n, pstd::span<const Point2f> uv,
              pstd::span<const int> faceIndices);

}  // namespace pbrt

#endif  // PBRT_UTIL_MESH_H
