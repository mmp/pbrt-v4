// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/loopsubdiv.h>

#include <pbrt/util/containers.h>
#include <pbrt/util/error.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/mesh.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include <algorithm>
#include <map>
#include <set>

namespace pbrt {

struct SDFace;
struct SDVertex;

// LoopSubdiv Macros
#define NEXT(i) (((i) + 1) % 3)
#define PREV(i) (((i) + 2) % 3)

// LoopSubdiv Local Structures
struct SDVertex {
    // SDVertex Constructor
    SDVertex(const Point3f &p = Point3f(0, 0, 0)) : p(p) {}

    // SDVertex Methods
    int valence();
    void oneRing(Point3f *p);
    Point3f p;
    SDFace *startFace = nullptr;
    SDVertex *child = nullptr;
    bool regular = false, boundary = false;
};

struct SDFace {
    // SDFace Constructor
    SDFace() {
        for (int i = 0; i < 3; ++i) {
            v[i] = nullptr;
            f[i] = nullptr;
        }
        for (int i = 0; i < 4; ++i)
            children[i] = nullptr;
    }

    // SDFace Methods
    int vnum(SDVertex *vert) const {
        for (int i = 0; i < 3; ++i)
            if (v[i] == vert)
                return i;
        LOG_FATAL("Basic logic error in SDFace::vnum()");
        return -1;
    }
    SDFace *nextFace(SDVertex *vert) { return f[vnum(vert)]; }
    SDFace *prevFace(SDVertex *vert) { return f[PREV(vnum(vert))]; }
    SDVertex *nextVert(SDVertex *vert) { return v[NEXT(vnum(vert))]; }
    SDVertex *prevVert(SDVertex *vert) { return v[PREV(vnum(vert))]; }
    SDVertex *otherVert(SDVertex *v0, SDVertex *v1) {
        for (int i = 0; i < 3; ++i)
            if (v[i] != v0 && v[i] != v1)
                return v[i];
        LOG_FATAL("Basic logic error in SDFace::otherVert()");
        return nullptr;
    }
    SDVertex *v[3];
    SDFace *f[3];
    SDFace *children[4];
};

struct SDEdge {
    // SDEdge Constructor
    SDEdge(SDVertex *v0 = nullptr, SDVertex *v1 = nullptr) {
        v[0] = std::min(v0, v1);
        v[1] = std::max(v0, v1);
        f[0] = f[1] = nullptr;
        f0edgeNum = -1;
    }

    // SDEdge Comparison Function
    bool operator<(const SDEdge &e2) const {
        if (v[0] == e2.v[0])
            return v[1] < e2.v[1];
        return v[0] < e2.v[0];
    }
    SDVertex *v[2];
    SDFace *f[2];
    int f0edgeNum;
};

// LoopSubdiv Local Declarations
static Point3f weightOneRing(SDVertex *vert, Float beta);
static Point3f weightBoundary(SDVertex *vert, Float beta);

// LoopSubdiv Inline Functions
inline int SDVertex::valence() {
    SDFace *f = startFace;
    if (!boundary) {
        // Compute valence of interior vertex
        int nf = 1;
        while ((f = f->nextFace(this)) != startFace)
            ++nf;
        return nf;
    } else {
        // Compute valence of boundary vertex
        int nf = 1;
        while ((f = f->nextFace(this)) != nullptr)
            ++nf;
        f = startFace;
        while ((f = f->prevFace(this)) != nullptr)
            ++nf;
        return nf + 1;
    }
}

inline Float beta(int valence) {
    if (valence == 3)
        return 3.f / 16.f;
    else
        return 3.f / (8.f * valence);
}

inline Float loopGamma(int valence) {
    return 1.f / (valence + 3.f / (8.f * beta(valence)));
}

// LoopSubdiv Function Definitions
TriangleMesh *LoopSubdivide(const Transform *renderFromObject, bool reverseOrientation,
                            int nLevels, pstd::span<const int> vertexIndices,
                            pstd::span<const Point3f> p, Allocator alloc) {
    std::vector<SDVertex *> vertices;
    std::vector<SDFace *> faces;
    // Allocate _LoopSubdiv_ vertices and faces
    std::unique_ptr<SDVertex[]> verts = std::make_unique<SDVertex[]>(p.size());
    for (int i = 0; i < p.size(); ++i) {
        verts[i] = SDVertex(p[i]);
        vertices.push_back(&verts[i]);
    }
    size_t nFaces = vertexIndices.size() / 3;
    std::unique_ptr<SDFace[]> fs = std::make_unique<SDFace[]>(nFaces);
    for (int i = 0; i < nFaces; ++i)
        faces.push_back(&fs[i]);

    // Set face to vertex pointers
    const int *vp = vertexIndices.data();
    for (size_t i = 0; i < nFaces; ++i, vp += 3) {
        SDFace *f = faces[i];
        for (int j = 0; j < 3; ++j) {
            SDVertex *v = vertices[vp[j]];
            f->v[j] = v;
            v->startFace = f;
        }
    }

    // Set neighbor pointers in _faces_
    std::set<SDEdge> edges;
    for (int i = 0; i < nFaces; ++i) {
        SDFace *f = faces[i];
        for (int edgeNum = 0; edgeNum < 3; ++edgeNum) {
            // Update neighbor pointer for _edgeNum_
            int v0 = edgeNum, v1 = NEXT(edgeNum);
            SDEdge e(f->v[v0], f->v[v1]);
            if (edges.find(e) == edges.end()) {
                // Handle new edge
                e.f[0] = f;
                e.f0edgeNum = edgeNum;
                edges.insert(e);
            } else {
                // Handle previously seen edge
                e = *edges.find(e);
                e.f[0]->f[e.f0edgeNum] = f;
                f->f[edgeNum] = e.f[0];
                edges.erase(e);
            }
        }
    }

    // Finish vertex initialization
    for (size_t i = 0; i < p.size(); ++i) {
        SDVertex *v = vertices[i];
        SDFace *f = v->startFace;
        do {
            f = f->nextFace(v);
        } while ((f != nullptr) && f != v->startFace);
        v->boundary = (f == nullptr);
        if (!v->boundary && v->valence() == 6)
            v->regular = true;
        else if (v->boundary && v->valence() == 4)
            v->regular = true;
        else
            v->regular = false;
    }

    // Refine _LoopSubdiv_ into triangles
    std::vector<SDFace *> f = faces;
    std::vector<SDVertex *> v = vertices;
    pstd::pmr::monotonic_buffer_resource resource;
    Allocator vertexFaceAllocator(&resource);

    for (int i = 0; i < nLevels; ++i) {
        // Update _f_ and _v_ for next level of subdivision
        std::vector<SDFace *> newFaces;
        std::vector<SDVertex *> newVertices;

        // Allocate next level of children in mesh tree
        for (SDVertex *vertex : v) {
            vertex->child = vertexFaceAllocator.new_object<SDVertex>();
            vertex->child->regular = vertex->regular;
            vertex->child->boundary = vertex->boundary;
            newVertices.push_back(vertex->child);
        }
        for (SDFace *face : f) {
            for (int k = 0; k < 4; ++k) {
                face->children[k] = vertexFaceAllocator.new_object<SDFace>();
                newFaces.push_back(face->children[k]);
            }
        }

        // Update vertex positions and create new edge vertices

        // Update vertex positions for even vertices
        for (SDVertex *vertex : v) {
            if (!vertex->boundary) {
                // Apply one-ring rule for even vertex
                if (vertex->regular)
                    vertex->child->p = weightOneRing(vertex, 1.f / 16.f);
                else
                    vertex->child->p = weightOneRing(vertex, beta(vertex->valence()));
            } else {
                // Apply boundary rule for even vertex
                vertex->child->p = weightBoundary(vertex, 1.f / 8.f);
            }
        }

        // Compute new odd edge vertices
        std::map<SDEdge, SDVertex *> edgeVerts;
        for (SDFace *face : f) {
            for (int k = 0; k < 3; ++k) {
                // Compute odd vertex on _k_th edge
                SDEdge edge(face->v[k], face->v[NEXT(k)]);
                SDVertex *vert = edgeVerts[edge];
                if (vert == nullptr) {
                    // Create and initialize new odd vertex
                    vert = vertexFaceAllocator.new_object<SDVertex>();
                    newVertices.push_back(vert);
                    vert->regular = true;
                    vert->boundary = (face->f[k] == nullptr);
                    vert->startFace = face->children[3];

                    // Apply edge rules to compute new vertex position
                    if (vert->boundary) {
                        vert->p = 0.5f * edge.v[0]->p;
                        vert->p += 0.5f * edge.v[1]->p;
                    } else {
                        vert->p = 3.f / 8.f * edge.v[0]->p;
                        vert->p += 3.f / 8.f * edge.v[1]->p;
                        vert->p += 1.f / 8.f * face->otherVert(edge.v[0], edge.v[1])->p;
                        vert->p +=
                            1.f / 8.f * face->f[k]->otherVert(edge.v[0], edge.v[1])->p;
                    }
                    edgeVerts[edge] = vert;
                }
            }
        }

        // Update new mesh topology

        // Update even vertex face pointers
        for (SDVertex *vertex : v) {
            int vertNum = vertex->startFace->vnum(vertex);
            vertex->child->startFace = vertex->startFace->children[vertNum];
        }

        // Update face neighbor pointers
        for (SDFace *face : f) {
            for (int j = 0; j < 3; ++j) {
                // Update children _f_ pointers for siblings
                face->children[3]->f[j] = face->children[NEXT(j)];
                face->children[j]->f[NEXT(j)] = face->children[3];

                // Update children _f_ pointers for neighbor children
                SDFace *f2 = face->f[j];
                face->children[j]->f[j] =
                    f2 != nullptr ? f2->children[f2->vnum(face->v[j])] : nullptr;
                f2 = face->f[PREV(j)];
                face->children[j]->f[PREV(j)] =
                    f2 != nullptr ? f2->children[f2->vnum(face->v[j])] : nullptr;
            }
        }

        // Update face vertex pointers
        for (SDFace *face : f) {
            for (int j = 0; j < 3; ++j) {
                // Update child vertex pointer to new even vertex
                face->children[j]->v[j] = face->v[j]->child;

                // Update child vertex pointer to new odd vertex
                SDVertex *vert = edgeVerts[SDEdge(face->v[j], face->v[NEXT(j)])];
                face->children[j]->v[NEXT(j)] = vert;
                face->children[NEXT(j)]->v[j] = vert;
                face->children[3]->v[j] = vert;
            }
        }

        // Prepare for next level of subdivision
        f = newFaces;
        v = newVertices;
    }

    // Push vertices to limit surface
    std::vector<Point3f> pLimit(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i]->boundary)
            pLimit[i] = weightBoundary(v[i], 1.f / 5.f);
        else
            pLimit[i] = weightOneRing(v[i], loopGamma(v[i]->valence()));
    }
    for (size_t i = 0; i < v.size(); ++i)
        v[i]->p = pLimit[i];

    // Compute vertex tangents on limit surface
    std::vector<Normal3f> Ns;
    Ns.reserve(v.size());
    std::vector<Point3f> pRing(16, Point3f());
    for (SDVertex *vertex : v) {
        Vector3f S(0, 0, 0), T(0, 0, 0);
        int valence = vertex->valence();
        if (valence > (int)pRing.size())
            pRing.resize(valence);
        vertex->oneRing(&pRing[0]);
        if (!vertex->boundary) {
            // Compute tangents of interior face
            for (int j = 0; j < valence; ++j) {
                S += std::cos(2 * Pi * j / valence) * Vector3f(pRing[j]);
                T += std::sin(2 * Pi * j / valence) * Vector3f(pRing[j]);
            }
        } else {
            // Compute tangents of boundary face
            S = pRing[valence - 1] - pRing[0];
            if (valence == 2)
                T = Vector3f(pRing[0] + pRing[1] - 2 * vertex->p);
            else if (valence == 3)
                T = pRing[1] - vertex->p;
            else if (valence == 4)  // regular
                T = Vector3f(-1 * pRing[0] + 2 * pRing[1] + 2 * pRing[2] + -1 * pRing[3] +
                             -2 * vertex->p);
            else {
                Float theta = Pi / float(valence - 1);
                T = Vector3f(std::sin(theta) * (pRing[0] + pRing[valence - 1]));
                for (int k = 1; k < valence - 1; ++k) {
                    Float wt = (2 * std::cos(theta) - 2) * std::sin((k)*theta);
                    T += Vector3f(wt * pRing[k]);
                }
                T = -T;
            }
        }
        Ns.push_back(Normal3f(Cross(S, T)));
    }

    // Create triangle mesh from subdivision mesh
    {
        size_t ntris = f.size();
        std::vector<int> verts(3 * ntris);
        int *vp = verts.data();
        size_t totVerts = v.size();
        std::map<SDVertex *, int> usedVerts;
        for (size_t i = 0; i < totVerts; ++i)
            usedVerts[v[i]] = i;
        for (size_t i = 0; i < ntris; ++i) {
            for (int j = 0; j < 3; ++j) {
                *vp = usedVerts[f[i]->v[j]];
                ++vp;
            }
        }
        return alloc.new_object<TriangleMesh>(
            *renderFromObject, reverseOrientation, verts, pLimit, std::vector<Vector3f>(),
            Ns, std::vector<Point2f>(), std::vector<int>(), alloc);
    }
}

static Point3f weightOneRing(SDVertex *vert, Float beta) {
    // Put _vert_ one-ring in _pRing_
    int valence = vert->valence();
    InlinedVector<Point3f, 16> pRing(valence);

    vert->oneRing(pRing.data());
    Point3f p = (1 - valence * beta) * vert->p;
    for (int i = 0; i < valence; ++i)
        p += beta * pRing[i];
    return p;
}

void SDVertex::oneRing(Point3f *p) {
    if (!boundary) {
        // Get one-ring vertices for interior vertex
        SDFace *face = startFace;
        do {
            *p++ = face->nextVert(this)->p;
            face = face->nextFace(this);
        } while (face != startFace);
    } else {
        // Get one-ring vertices for boundary vertex
        SDFace *face = startFace, *f2;
        while ((f2 = face->nextFace(this)) != nullptr)
            face = f2;
        *p++ = face->nextVert(this)->p;
        do {
            *p++ = face->prevVert(this)->p;
            face = face->prevFace(this);
        } while (face != nullptr);
    }
}

static Point3f weightBoundary(SDVertex *vert, Float beta) {
    // Put _vert_ one-ring in _pRing_
    int valence = vert->valence();
    InlinedVector<Point3f, 16> pRing(valence);

    vert->oneRing(pRing.data());
    Point3f p = (1 - 2 * beta) * vert->p;
    p += beta * pRing[0];
    p += beta * pRing[valence - 1];
    return p;
}

}  // namespace pbrt
