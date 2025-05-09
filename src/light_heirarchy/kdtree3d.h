#ifndef KDTREE3D_H
#define KDTREE3D_H

#include "Vector3f.h"
#include <vector>
#include <memory>
#include <algorithm>
#include <tuple>
#include "CubeDS.h"

struct KDNode {
    Vector3f point;
    float intensity;
    CubeDS shadowCubeMap;
    
    std::unique_ptr<KDNode> left;
    std::unique_ptr<KDNode> right;

    KDNode(const Vector3f& pos, float intensity, const CubeDS& cubeMap) 
        : point(pos), intensity(intensity), shadowCubeMap(cubeMap), left(nullptr), right(nullptr) {}
};

class KDTree {
public:
    KDTree(const std::vector<std::tuple<Vector3f, float, CubeDS>>& lights) {
        root = build(lights, 0);
    }

    void radiusSearch(const Vector3f& target, float radius, std::vector<KDNode*>& results) const {
        radiusSearchRecursive(root.get(), target, radius * radius, 0, results);
    }

private:
    std::unique_ptr<KDNode> root;

    std::unique_ptr<KDNode> build(std::vector<std::tuple<Vector3f,float,CubeDS>> points, int depth) {
        if (points.empty()) return nullptr;

        int axis = depth % 3;
        size_t median = points.size() / 2;

        // Sorts based on position
        std::nth_element(points.begin(), points.begin() + median, points.end(),
            [axis](const std::tuple<Vector3f, float, CubeDS>& a, const std::tuple<Vector3f, float, CubeDS>& b) {
                return std::get<0>(a)[axis] < std::get<0>(b)[axis];
            });

        auto node = std::make_unique<KDNode>(
            std::get<0>(points[median]), 
            std::get<1>(points[median]),
            std::get<2>(points[median])
        );

        std::vector<std::tuple<Vector3f,float,CubeDS>> leftPoints(points.begin(), points.begin() + median);
        std::vector<std::tuple<Vector3f,float,CubeDS>> rightPoints(points.begin() + median + 1, points.end());

        node->left = build(std::move(leftPoints), depth + 1);
        node->right = build(std::move(rightPoints), depth + 1);

        return node;
    }

    void radiusSearchRecursive(KDNode* node, const Vector3f& target, float radiusSquared,
                               int depth, std::vector<KDNode*>& results) const {
        if (!node) return;

        if (target.distanceSquared(node->point) <= radiusSquared) {
            results.push_back(node);
        }

        int axis = depth % 3;
        float diff = target[axis] - node->point[axis];

        if (diff <= 0) {
            radiusSearchRecursive(node->left.get(), target, radiusSquared, depth + 1, results);
            if (diff * diff <= radiusSquared)
                radiusSearchRecursive(node->right.get(), target, radiusSquared, depth + 1, results);
        } else {
            radiusSearchRecursive(node->right.get(), target, radiusSquared, depth + 1, results);
            if (diff * diff <= radiusSquared)
                radiusSearchRecursive(node->left.get(), target, radiusSquared, depth + 1, results);
        }
    }
};

#endif // KDTREE3D_H
