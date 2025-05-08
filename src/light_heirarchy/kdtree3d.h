#pragma once

#include "Vector3f.h"
#include <vector>
#include <memory>
#include <random>

struct KDNode {
    Vector3f point;
    float intensity;
    int idx;  // Index of the light in its level
    std::unique_ptr<KDNode> left;
    std::unique_ptr<KDNode> right;

    KDNode(const Vector3f pos, const float intensity, int idx = -1) 
        : point(pos), intensity(intensity), idx(idx) {}
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
    void getAllNodes(std::vector<KDNode*>& results, int num_samples = 0) const {
        std::vector<KDNode*> all;
        getAllNodesRecursive(root.get(), all);
        if (num_samples > 0 && num_samples < (int)all.size()) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(all.begin(), all.end(), g);
            results.assign(all.begin(), all.begin() + num_samples);
        } else {
            results = std::move(all);
        }
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

        // Create node with position, intensity, and index
        std::unique_ptr<KDNode> node = std::make_unique<KDNode>(
            points[median].first,  // position
            points[median].second, // intensity
            static_cast<int>(median)  // index
        );

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
    
    void getAllNodesRecursive(KDNode* node, std::vector<KDNode*>& results) const {
        if (!node) return;
        results.push_back(node);
        getAllNodesRecursive(node->left.get(), results);
        getAllNodesRecursive(node->right.get(), results);
    }
};
