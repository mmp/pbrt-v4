// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_MESH_H
#define PBRT_UTIL_MESH_H

#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

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
                 std::vector<Point2f> uv, std::vector<int> faceIndices);

    std::string ToString() const;

    bool WritePLY(const std::string &filename) const;

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
                      std::vector<int> faceIndices, PiecewiseConstant2D *imageDist);

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

struct TriQuadMesh {
    static TriQuadMesh ReadPLY(const std::string &filename);

    void ConvertToOnlyTriangles();
    std::string ToString() const;

    std::vector<Point3f> p;
    std::vector<Normal3f> n;
    std::vector<Point2f> uv;
    std::vector<int> faceIndices;
    std::vector<int> triIndices, quadIndices;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_MESH_H
