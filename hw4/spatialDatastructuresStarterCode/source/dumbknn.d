module dumbknn;

import std.stdio;
import std.algorithm;
import std.range;
import std.traits;
import std.math;

import common;

struct DumbKNN(size_t Dim){

    alias PT = Point!Dim;
    private PT[] points;

    this(PT[] points){
        this.points = points.dup;
    }

    PT[] rangeQuery(PT p, float r){
        PT[] ret;
        foreach(const ref q; points){
            if(distance(p, q) < r){
                ret ~= q;
            }
        }
        return ret;
    }

    PT[] knnQuery(PT p, int k){
        auto pq = makePriorityQueue!Dim(p);

        foreach(const ref q; points){
            if(pq.length < k){
                pq.insert(q);
            }
            else if(distance(p, q) < distance(p, pq.front)){
                pq.popFront;
                pq.insert(q);
            }
        }

        PT[] ret;
        while(!pq.empty){
            ret ~= pq.front;
            pq.popFront;
        }

        return ret.reverse;
    }
}

unittest{
    auto points = [Point!2([0.5, 0.5]), Point!2([1, 1]),
                   Point!2([0.75, 0.4]), Point!2([0.4, 0.74])];
    auto dumb = DumbKNN!2(points);

    writeln("dumbknn rq");
    foreach(p; dumb.rangeQuery(Point!2([1,1]), 0.7)){
        writeln(p);
    }
    assert(dumb.rangeQuery(Point!2([1,1]), 0.7).length == 3);

    writeln("dumb knn");
    foreach(p; dumb.knnQuery(Point!2([1,1]), 3)){
        writeln(p);
    }
}