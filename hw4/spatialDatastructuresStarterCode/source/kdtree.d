module kdtree;

import std.stdio;
import std.math;
import std.algorithm;
import std.range;

import common;

struct KDTree(size_t Dim){

    class Node(size_t splitDimension) {

        enum thisLevel = splitDimension;
        enum nextLevel = (splitDimension + 1) % Dim;

        Node!nextLevel left, right;

        Point!Dim splitPoint;
        Point!Dim[] storedPoints;

        this(Point!Dim[] points) {

            if (points.length < 3) {
                storedPoints = points;
                return;
            }

            auto leftHalf = points.medianByDimension!thisLevel;
            auto rightHalf = points[leftHalf.length + 1 .. $];
            splitPoint = points[leftHalf.length];

            left = new Node!nextLevel(leftHalf);
            right = new Node!nextLevel(rightHalf);
        }
    }

    Node!0 root;

    this(Point!Dim[] points){
        root = new Node!0(points);
    }

    Point!Dim[] knnQuery(Point!Dim queryPoint, int k) {
        auto pq = makePriorityQueue!Dim(queryPoint);

        void recurse(size_t dim)(Node!dim n, AABB!Dim aabb) {
            if (pq.length < k || distance(queryPoint, closest(aabb, queryPoint)) < distance(queryPoint, pq.front)) {
                if (isNaN(n.splitPoint[0])) {
                    foreach(point; n.storedPoints) {
                        if (pq.length < k) {
                            pq.insert(point);
                        }
                        else if (distance(point, queryPoint) < distance(queryPoint, pq.front)) {
                            pq.popFront;
                            pq.insert(point);
                        }
                    }
                }
                else {
                    if (pq.length < k || distance(queryPoint, n.splitPoint) < distance(queryPoint, pq.front)) {
                        if (pq.length < k) {
                            pq.insert(n.splitPoint);
                        }
                        else if (distance(n.splitPoint, queryPoint) < distance(queryPoint, pq.front)) {
                            pq.popFront;
                            pq.insert(n.splitPoint);
                        }
                    }

                    auto leftAABB = aabb;
                    leftAABB.max[n.thisLevel] = n.splitPoint[n.thisLevel];
                    recurse(n.left, leftAABB);

                    auto rightAABB = aabb;
                    rightAABB.min[n.thisLevel] = n.splitPoint[n.thisLevel];
                    recurse(n.right, rightAABB);
                }
            }
        }

        AABB!Dim rootAABB;
        foreach(i; 0 .. Dim){
            rootAABB.min[i] = -float.infinity;
            rootAABB.max[i] = float.infinity;
        }

        recurse!0(root, rootAABB);

        Point!Dim[] result;
        while(!pq.empty){
            result ~= pq.front;
            pq.popFront;
        }

        return result.reverse;
    }

    Point!Dim[] rangeQuery( Point!Dim queryPoint, float radius ) {
        Point!Dim[] result;

        void recurse(size_t dim)(Node!dim n) {
            if (distance(n.splitPoint, queryPoint) < radius) {
                result ~= n.splitPoint;
            }
            foreach(point; n.storedPoints) {
                if (distance(point, queryPoint) < radius) {
                    result ~= point;
                }
            }

            if(n.left && queryPoint[n.thisLevel] - radius < n.splitPoint[n.thisLevel]) {
                recurse(n.left);
            }
            if(n.right && queryPoint[n.thisLevel] + radius > n.splitPoint[n.thisLevel]) {
                recurse(n.right);
            }
        }

        recurse!0(root);
        return result;
    }
}

unittest {
    // Test with uniform points
    auto uniformPoints = getUniformPoints!2(1000);
    auto kdtreeUniform = KDTree!2(uniformPoints);

    writeln("Testing KDTree with uniform points:");
    foreach(p; kdtreeUniform.rangeQuery(Point!2([0.5,0.5]), 0.1)){
        writeln(p);
    }
    writeln("Uniform KDTree range query passed!");

    foreach(p; kdtreeUniform.knnQuery(Point!2([0.5,0.5]), 10)){
        writeln(p);
    }
    writeln("Uniform KDTree KNN query passed!");

    // Test with gaussian points
    auto gaussianPoints = getGaussianPoints!2(1000);
    auto kdtreeGaussian = KDTree!2(gaussianPoints);

    writeln("Testing KDTree with gaussian points:");
    foreach(p; kdtreeGaussian.rangeQuery(Point!2([0,0]), 1.0)){
        writeln(p);
    }
    writeln("Gaussian KDTree range query passed!");

    foreach(p; kdtreeGaussian.knnQuery(Point!2([0,0]), 10)){
        writeln(p);
    }
    writeln("Gaussian KDTree KNN query passed!");
}