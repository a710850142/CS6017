module quadtree;

import std.stdio;
import std.math;
import std.algorithm;
import std.range;

import common;

struct QuadTree(size_t dim){

    struct Node {
        AABB!2 aabb;
        Node[] children;
        Point!2[] points;
        bool leaf;
        Point!2 midpoint;

        int MAX_POINTS_IN_NODE = 10;

        this(Point!2[] points, AABB!2 aabb) {
            if(points.length > MAX_POINTS_IN_NODE) {
                midpoint = (aabb.max + aabb.min) / 2;

                auto rightHalf = points.partitionByDimension!0(midpoint[0]);
                auto leftHalf = points[0 .. $ - rightHalf.length];

                auto tl = leftHalf.partitionByDimension!1(midpoint[1]);
                auto tr = rightHalf.partitionByDimension!1(midpoint[1]);
                auto bl = leftHalf[0 .. $ - tl.length];
                auto br = rightHalf[0 .. $ - tr.length];

                auto aabbTL = boundingBox!dim([Point!dim([aabb.min[0], midpoint[1]]), Point!dim([midpoint[0], aabb.max[1]])]);
                auto aabbTR = boundingBox!dim([midpoint, aabb.max]);
                auto aabbBL = boundingBox!dim([aabb.min, midpoint]);
                auto aabbBR = boundingBox!dim([Point!dim([midpoint[0], aabb.min[1]]), Point!dim([aabb.max[0], midpoint[1]])]);

                children ~= Node(tl, aabbTL);
                children ~= Node(tr, aabbTR);
                children ~= Node(bl, aabbBL);
                children ~= Node(br, aabbBR);
            }
            else {
                this.points = points.dup;
                this.leaf = true;
            }
        }

        Point!2[] rangeQuery(Point!2 queryPoint, float radius) {
            Point!2[] result;

            if (leaf) {
                foreach(const ref point; points) {
                    if (distance(queryPoint, point) < radius) {
                        result ~= point;
                    }
                }
            }
            else {
                foreach(child; children) {
                    if (distance(queryPoint, closest(child.aabb, queryPoint)) < radius) {
                        result ~= child.rangeQuery(queryPoint, radius);
                    }
                }
            }

            return result;
        }

        Point!2[] knnQuery(Point!2 queryPoint, int k) {
            auto pq = makePriorityQueue!2(queryPoint);

            void recurse(Node n) {
                if (n.leaf) {
                    foreach(point; n.points) {
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
                    foreach(child; n.children) {
                        if (pq.length < k || distance(queryPoint, closest(child.aabb, queryPoint)) < distance(queryPoint, pq.front)) {
                            recurse(child);
                        }
                    }
                }
            }

            recurse(this);

            Point!2[] result;
            while(!pq.empty){
                result ~= pq.front;
                pq.popFront;
            }

            return result.reverse;
        }
    }

    Node root;

    this(Point!2[] points, AABB!2 initialAABB) {
        this.root = Node(points, initialAABB);
    }

    Point!2[] rangeQuery(Point!2 queryPoint, float radius) {
        return root.rangeQuery(queryPoint, radius);
    }

    Point!2[] knnQuery(Point!2 queryPoint, int k) {
        return root.knnQuery(queryPoint, k);
    }
}

unittest{
    // Test with uniform points
    auto uniformPoints = getUniformPoints!2(1000);
    auto quadtreeUniform = QuadTree!2(uniformPoints, boundingBox(uniformPoints));

    writeln("Testing QuadTree with uniform points:");
    foreach(p; quadtreeUniform.rangeQuery(Point!2([0.5,0.5]), 0.1)){
        writeln(p);
    }
    writeln("Uniform QuadTree range query passed!");

    foreach(p; quadtreeUniform.knnQuery(Point!2([0.5,0.5]), 10)){
        writeln(p);
    }
    writeln("Uniform QuadTree KNN query passed!");

    // Test with gaussian points
    auto gaussianPoints = getGaussianPoints!2(1000);
    auto quadtreeGaussian = QuadTree!2(gaussianPoints, boundingBox(gaussianPoints));

    writeln("Testing QuadTree with gaussian points:");
    foreach(p; quadtreeGaussian.rangeQuery(Point!2([0,0]), 1.0)){
        writeln(p);
    }
    writeln("Gaussian QuadTree range query passed!");

    foreach(p; quadtreeGaussian.knnQuery(Point!2([0,0]), 10)){
        writeln(p);
    }
    writeln("Gaussian QuadTree KNN query passed!");
}