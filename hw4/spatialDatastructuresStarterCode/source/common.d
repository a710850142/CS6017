
public import std.stdio;
public import std.range;
public import std.array;
public import std.algorithm;
public import std.traits;
public import std.math;
public import std.datetime.stopwatch;
public import std.container.binaryheap;


struct Point(size_t N){
    float[N] data;
    alias data this;

    Point opBinary(string op)(float r){
        Point ret;
        foreach(i; 0..N){
            ret[i] = mixin("data[i] " ~ op ~ "r");
        }
        return ret;
    }

    Point opBinary(string op)(Point p){
        Point ret;
        foreach(i; 0..N){
            ret[i] = mixin("data[i] " ~ op ~ "p.data[i]");
        }
        return ret;
    }
}

struct Indices(size_t Dim){
    size_t[Dim] data;
    alias data this;
}


float distance(T)(T a, T b) if(isInstanceOf!(Point, T)){
    return sqrt(a[].zip(b[])
                .map!( x => (x[0] - x[1])*(x[0] - x[1]))
                .sum);
}


unittest {
    auto x = Point!2([3, 0]);
    auto y = Point!2([0, -4]);
    assert(distance(x, y) == 5);
}



struct AABB(size_t N){
    Point!N min, max;
}


AABB!N boundingBox(size_t N)(Point!N[] points){
    AABB!N ret;
    ret.min[] = float.infinity;
    ret.max[] = -float.infinity;
    foreach(const ref p; points){
        foreach(i; 0 .. N){
            ret.min[i] = min(ret.min[i], p[i]);
            ret.max[i] = max(ret.max[i], p[i]);
        }
    }
    return ret;
}



Point!N closest(size_t N)(AABB!N aabb, Point!N p){
    foreach(i; 0 .. N){
        p[i] = clamp(p[i], aabb.min[i], aabb.max[i]);
    }
    return p;
}

unittest{

    auto points = [Point!2([1,2]), Point!2([-2, 5])];
    auto aabb = boundingBox(points);
    assert(aabb.min == Point!2([-2, 2]));
    assert(aabb.max == Point!2([1, 5]));

    assert(closest(aabb, Point!2([0,0])) == Point!2([0,2]));
    assert(aabb.closest(Point!2([0.5,3])) == Point!2([0.5,3]));
}


auto getIndicesRange(size_t Dim)(Indices!Dim start, Indices!Dim stop){
    auto helper(size_t N)(){
        auto thisIota = iota(start[N], stop[N] +1).map!(x => [x]);
        static if(N == Dim -1){
            return thisIota;
        } else {
            return cartesianProduct(thisIota, helper!(N+1)())
                .map!(function(x){
                        return x[0] ~ x[1];
                    });
        }
    }
    return helper!0().map!(function Indices!Dim(x) {
            return Indices!Dim(x[0..Dim]);
        });
}

unittest{

    auto bottomLeft = Indices!3([0, 2, 3]);
    auto topRight = Indices!3([2, 3, 5]);
    writeln("indices between ", bottomLeft, " and ", topRight);
    foreach(ind; getIndicesRange(bottomLeft, topRight)){
        writeln(ind);
    }

}




auto medianByDimension(size_t SortingDim, size_t PointDim)(Point!PointDim[] points){

    return points.topN!((a, b) => a[SortingDim] < b[SortingDim])(points.length/2);
}


unittest{
    auto points = [Point!2([1,2]), Point!2([3,1]), Point!2([2,3])];
    points.medianByDimension!0;

    assert(points == [Point!2([1,2]), Point!2([2,3]), Point!2([3,1])]);

    points.medianByDimension!1;
    assert(points == [Point!2([3,1]), Point!2([1,2]), Point!2([2,3])]);
}



auto partitionByDimension(size_t sortingDim, size_t PointDim)(Point!PointDim[] points, float splitValue){
    return points.partition!(x => x[sortingDim] < splitValue);
}

unittest{
    auto points = [Point!2([1,2]), Point!2([3,1]), Point!2([2,3])];
    auto rightHalf = points.partitionByDimension!0(2.5);

    auto leftHalf = points[0 .. $ - rightHalf.length];
    assert(rightHalf == [Point!2([3,1])]);
    assert(leftHalf.length == 2);


}


auto makePriorityQueue(size_t Dim)(Point!Dim p){
    Point!Dim[] storage;
    return BinaryHeap!(Point!Dim[], (a, b) => distance(a, p) < distance(b, p))(storage);
}

unittest{
    auto points = [Point!2([1,2]), Point!2([3,1]), Point!2([2,3])];
    auto pq = makePriorityQueue(Point!2([0,0]));
    foreach(p; points) pq.insert(p);
    assert(pq.front == Point!2([2,3]));
    pq.popFront;

    assert(pq.release == [Point!2([3,1]), Point!2([1,2])]);
}



void sortByDistance(size_t Dim)(Point!Dim[] points, Point!Dim p){
    points.sort!((a, b) => distance(a, p) < distance(b, p));
}


void topNByDistance(size_t Dim)(Point!Dim[] points, Point!Dim p, int k){
    points.topN!((a, b) => distance(a, p) < distance(b, p))(k);
}



Point!Dim[] getUniformPoints(size_t Dim)(size_t n){

    import std.random : uniform01;
    auto ret = new Point!Dim[n];
    foreach(ref p; ret){
        foreach(i; 0..Dim){
            p[i] = uniform01!float;
        }
    }
    return ret;
}


Point!Dim[] getGaussianPoints(size_t Dim)(size_t n){
    import std.mathspecial: normalDistributionInverse;
    return getUniformPoints!Dim(n).map!( function(Point!Dim x){
            Point!Dim ret;
            foreach(i; 0..Dim){
                ret[i] = normalDistributionInverse(x[i]);
            }
            return ret;
        }).array;
}


unittest{

    auto uPoints = getUniformPoints!2(1000);
    auto uBounds = boundingBox(uPoints);

    assert(uBounds.min[0] >= 0);
    assert(uBounds.min[1] >= 0);
    assert(uBounds.max[0] <= 1);
    assert(uBounds.max[1] <= 1);

    auto gPoints = getGaussianPoints!3(10000);

    writeln("gaussian points bounding box: ", boundingBox(gPoints));
}
