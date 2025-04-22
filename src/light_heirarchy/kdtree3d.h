#pragma once

#include "Vector3f.h"
#include <vector>
#include <memory>
#include <limits>

struct KDNode {
    Vector3f point;
    std::unique_ptr<KDNode> left;
    std::unique_ptr<KDNode> right;

    KDNode(const Vector3f& pt) : point(pt) {}
};

class KDTree {
public:

    KDTree(const std::vector<Vector3f>& points) {
        root = build(points, 0);
    }

    void radiusSearch(const Vector3f& target, float radius, std::vector<Vector3f>& results) const {
        radiusSearchRecursive(root.get(), target, radius * radius, 0, results);
    }

private:
    std::unique_ptr<KDNode> root;

    std::unique_ptr<KDNode> build(std::vector<Vector3f> points, int depth) {
        if (points.empty()) return nullptr;

        int axis = depth % 3;
        size_t median = points.size() / 2;
        std::nth_element(points.begin(), points.begin() + median, points.end(),
                         [axis](const Vector3f& a, const Vector3f& b) {
                             return a[axis] < b[axis];
                         });

        std::unique_ptr<KDNode> node = std::make_unique<KDNode>(points[median]);

        std::vector<Vector3f> leftPoints(points.begin(), points.begin() + median);
        std::vector<Vector3f> rightPoints(points.begin() + median + 1, points.end());

        node->left = build(leftPoints, depth + 1);
        node->right = build(rightPoints, depth + 1);

        return node;
    }

    void radiusSearchRecursive(KDNode* node, const Vector3f& target, float radiusSquared,
                               int depth, std::vector<Vector3f>& results) const {
        if (!node) return;

        if (target.distanceSquared(node->point) <= radiusSquared) {
            results.push_back(node->point);
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
