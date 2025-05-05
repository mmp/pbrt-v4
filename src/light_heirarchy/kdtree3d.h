#pragma once

#include "Vector3f.h"
#include <vector>
#include <memory>

struct KDNode {
    Vector3f point;
    float intensity;
    std::unique_ptr<KDNode> left;
    std::unique_ptr<KDNode> right;

    KDNode(const Vector3f pos, const float intensity) : point(pos), intensity(intensity) {}
};

class KDTree {
public:

    KDTree(const std::vector<std::pair<Vector3f, float>>& lights) {
        root = build(lights, 0);
    }

    // TODO: modify to return intensity, not point (fixed)
    void radiusSearch(const Vector3f& target, float radius, std::vector<KDNode*>& results) const {
        radiusSearchRecursive(root.get(), target, radius * radius, 0, results);
    }

    // TODO: needs lookup for a single node given position

private:
    // put this, for example, in KDTree.h or a shared Lights.h
struct LightRecord
{
    Vector3f pos;   // world-space position of the light

    // you can decide what you store for intensity:
    //  * float  I            – already a luminance
    //  * Vector3f rgb        – if you keep RGB, we'll convert to luminance

    float  I;       // simplest: store luminance directly
    // Vector3f rgb;   // <-- alternate form if you need full colour
};

    std::unique_ptr<KDNode> root;

    std::unique_ptr<KDNode> build(std::vector<std::pair<Vector3f,float>> points, int depth) {
        if (points.empty()) return nullptr;

        int axis = depth % 3;
        size_t median = points.size() / 2;

        // Sorts based on position
        std::nth_element(points.begin(), points.begin() + median, points.end(),
            [axis](const std::pair<Vector3f, float>& a, const std::pair<Vector3f, float>& b) {
                return a.first[axis] < b.first[axis];
            });

        std::unique_ptr<KDNode> node = std::make_unique<KDNode>(points[median].first, points[median].second);

        std::vector<std::pair<Vector3f,float>> leftPoints(points.begin(), points.begin() + median);
        std::vector<std::pair<Vector3f,float>> rightPoints(points.begin() + median + 1, points.end());

        node->left = build(leftPoints, depth + 1);
        node->right = build(rightPoints, depth + 1);

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
