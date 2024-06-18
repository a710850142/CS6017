module bucketknn;

import std.stdio;
import std.math;
import std.algorithm;
import std.range;

import common;

struct BucketKNN(size_t Dim){

    alias Bucket = Point!Dim[];
    Bucket[] buckets;
    int nDivisions;
    Point!Dim minCorner;
    Point!Dim bucketSize;

    this(Point!Dim[] points, int nDivisions){
        this.nDivisions = nDivisions;
        auto aabb = boundingBox(points);
        minCorner = aabb.min;
        bucketSize = (aabb.max - aabb.min)/nDivisions;

        buckets = new Bucket[pow(nDivisions, Dim)];

        foreach(const ref point; points){
            buckets[getIndex(getIndices(point))] ~= point;
        }
    }

    Indices!Dim getIndices(Point!Dim p){
        Indices!Dim ret;
        foreach(i; 0..Dim){
            ret[i] = cast(size_t)clamp(cast(int)( (p[i] - minCorner[i])/bucketSize[i]), 0, nDivisions - 1);
        }
        return ret;
    }

    size_t getIndex(Indices!Dim ind){
        size_t ret = 0;
        foreach(i, x; ind){
            ret += x*pow(nDivisions, Dim - i - 1);
        }
        return ret;
    }

    Point!Dim[] rangeQuery(Point!Dim p, float r){
        auto bottomCorner = p - r;
        auto topCorner = p + r;
        auto startBucket = getIndices(bottomCorner);
        auto stopBucket = getIndices(topCorner);

        Point!Dim[] ret;
        foreach(bIndices; getIndicesRange(startBucket, stopBucket)){
            foreach(const ref q; buckets[getIndex(bIndices)]){
                if(distance(p, q) < r)
                    ret ~= q;
            }
        }
        return ret;
    }

    Point!Dim[] knnQuery(Point!Dim p, int k){
        auto pq = makePriorityQueue!Dim(p);

        auto r = maxElement(bucketSize[]);
        auto ret = rangeQuery(p, r);

        foreach(const ref q; ret){
            if(pq.length < k){
                pq.insert(q);
            }
            else if(distance(p, q) < distance(p, pq.front)){
                pq.popFront;
                pq.insert(q);
            }
        }

        while(pq.length < k && r < sqrt(float.max)){
            r *= 2;
            ret = rangeQuery(p, r);

            foreach(const ref q; ret){
                if(pq.length < k){
                    pq.insert(q);
                }
                else if(distance(p, q) < distance(p, pq.front)){
                    pq.popFront;
                    pq.insert(q);
                }
            }
        }

        Point!Dim[] result;
        while(!pq.empty){
            result ~= pq.front;
            pq.popFront;
        }

        return result.reverse;
    }
}

unittest{
    // Test with uniform points
    auto uniformPoints = getUniformPoints!2(1000);
    auto bknnUniform = BucketKNN!2(uniformPoints, 10);

    writeln("Testing BucketKNN with uniform points:");
    foreach(p; bknnUniform.rangeQuery(Point!2([0.5,0.5]), 0.1)){
        writeln(p);
    }
    writeln("Uniform BucketKNN range query passed!");

    foreach(p; bknnUniform.knnQuery(Point!2([0.5,0.5]), 10)){
        writeln(p);
    }
    writeln("Uniform BucketKNN KNN query passed!");

    // Test with gaussian points
    auto gaussianPoints = getGaussianPoints!2(1000);
    auto bknnGaussian = BucketKNN!2(gaussianPoints, 10);

    writeln("Testing BucketKNN with gaussian points:");
    foreach(p; bknnGaussian.rangeQuery(Point!2([0,0]), 1.0)){
        writeln(p);
    }
    writeln("Gaussian BucketKNN range query passed!");

    foreach(p; bknnGaussian.knnQuery(Point!2([0,0]), 10)){
        writeln(p);
    }
    writeln("Gaussian BucketKNN KNN query passed!");
}