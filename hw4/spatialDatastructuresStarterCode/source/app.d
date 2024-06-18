import std.stdio;
import std.range;
import std.array;
import std.algorithm;
import std.traits;
import std.math;
import std.datetime.stopwatch;
import std.container.binaryheap;
import std.file;

import common;
import dumbknn;
import bucketknn;
import kdtree;
import quadtree;

void main()
{
    auto uniformFile = File("uniform_output.csv", "w");
    uniformFile.writeln("data_structure,time,D,k,N");

    auto gaussianFile = File("gaussian_output.csv", "w");
    gaussianFile.writeln("data_structure,time,D,k,N");

    static foreach(dim; 1..8){{
        foreach(N; [100, 1000, 10000, 100000]) {
            auto testingPoints = getUniformPoints!dim(100);

            // Uniform distribution
            auto uniformTrainingPoints = getUniformPoints!dim(N);
            auto uniformKD = DumbKNN!dim(uniformTrainingPoints);
            auto uniformSW = StopWatch(AutoStart.no);

            foreach(k; [1,2,5,10,20,50,100]) {
                uniformSW.start;
                foreach(const ref qp; testingPoints){
                    uniformKD.knnQuery(qp, k);
                }
                uniformSW.stop;
                uniformFile.writefln("%s,%s,%s,%s,%s", "dumbKNN", uniformSW.peek.total!"usecs", dim, k, N);
            }

            // Gaussian distribution
            auto gaussianTrainingPoints = getGaussianPoints!dim(N);
            auto gaussianKD = DumbKNN!dim(gaussianTrainingPoints);
            auto gaussianSW = StopWatch(AutoStart.no);

            foreach(k; [1,2,5,10,20,50,100]) {
                gaussianSW.start;
                foreach(const ref qp; testingPoints){
                    gaussianKD.knnQuery(qp, k);
                }
                gaussianSW.stop;
                gaussianFile.writefln("%s,%s,%s,%s,%s", "dumbKNN", gaussianSW.peek.total!"usecs", dim, k, N);
            }
        }
    }}

    static foreach(dim; 1..8){{
        foreach(N; [100, 1000, 10000, 100000]) {
            auto testingPoints = getUniformPoints!dim(100);

            // Uniform distribution
            auto uniformTrainingPoints = getUniformPoints!dim(N);
            auto uniformBK = BucketKNN!dim(uniformTrainingPoints, cast(int)pow(N/64, 1.0/dim));
            auto uniformSW = StopWatch(AutoStart.no);

            foreach(k; [1,2,5,10,20,50,100]) {
                uniformSW.start;
                foreach(const ref qp; testingPoints){
                    uniformBK.knnQuery(qp, k);
                }
                uniformSW.stop;
                uniformFile.writefln("%s,%s,%s,%s,%s", "bucketKNN", uniformSW.peek.total!"usecs", dim, k, N);
            }

            // Gaussian distribution
            auto gaussianTrainingPoints = getGaussianPoints!dim(N);
            auto gaussianBK = BucketKNN!dim(gaussianTrainingPoints, cast(int)pow(N/64, 1.0/dim));
            auto gaussianSW = StopWatch(AutoStart.no);

            foreach(k; [1,2,5,10,20,50,100]) {
                gaussianSW.start;
                foreach(const ref qp; testingPoints){
                    gaussianBK.knnQuery(qp, k);
                }
                gaussianSW.stop;
                gaussianFile.writefln("%s,%s,%s,%s,%s", "bucketKNN", gaussianSW.peek.total!"usecs", dim, k, N);
            }
        }
    }}

    static foreach(dim; 2..3){{
        foreach(N; [100, 1000, 10000, 100000]) {
            auto testingPoints = getUniformPoints!dim(100);

            // Uniform distribution
            auto uniformTrainingPoints = getUniformPoints!dim(N);
            auto uniformQuad = QuadTree!dim(uniformTrainingPoints, boundingBox(uniformTrainingPoints));
            auto uniformSW = StopWatch(AutoStart.no);

            foreach(k; [1,2,5,10,20,50,100]) {
                uniformSW.start;
                foreach(const ref qp; testingPoints){
                    uniformQuad.knnQuery(qp, k);
                }
                uniformSW.stop;
                uniformFile.writefln("%s,%s,%s,%s,%s", "quadTree", uniformSW.peek.total!"usecs", dim, k, N);
            }

            // Gaussian distribution
            auto gaussianTrainingPoints = getGaussianPoints!dim(N);
            auto gaussianQuad = QuadTree!dim(gaussianTrainingPoints, boundingBox(gaussianTrainingPoints));
            auto gaussianSW = StopWatch(AutoStart.no);

            foreach(k; [1,2,5,10,20,50,100]) {
                gaussianSW.start;
                foreach(const ref qp; testingPoints){
                    gaussianQuad.knnQuery(qp, k);
                }
                gaussianSW.stop;
                gaussianFile.writefln("%s,%s,%s,%s,%s", "quadTree", gaussianSW.peek.total!"usecs", dim, k, N);
            }
        }
    }}

    static foreach(dim; 1..8){{
        foreach(N; [100, 1000, 10000, 100000]) {
            auto testingPoints = getUniformPoints!dim(100);

            // Uniform distribution
            auto uniformTrainingPoints = getUniformPoints!dim(N);
            auto uniformKD = KDTree!dim(uniformTrainingPoints);
            auto uniformSW = StopWatch(AutoStart.no);

            foreach(k; [1,2,5,10,20,50,100]){
                uniformSW.start;
                foreach(const ref qp; testingPoints){
                    uniformKD.knnQuery(qp, k);
                }
                uniformSW.stop;
                uniformFile.writefln("%s,%s,%s,%s,%s", "kdTree", uniformSW.peek.total!"usecs", dim, k, N);
            }

            // Gaussian distribution
            auto gaussianTrainingPoints = getGaussianPoints!dim(N);
            auto gaussianKD = KDTree!dim(gaussianTrainingPoints);
            auto gaussianSW = StopWatch(AutoStart.no);

            foreach(k; [1,2,5,10,20,50,100]){
                gaussianSW.start;
                foreach(const ref qp; testingPoints){
                    gaussianKD.knnQuery(qp, k);
                }
                gaussianSW.stop;
                gaussianFile.writefln("%s,%s,%s,%s,%s", "kdTree", gaussianSW.peek.total!"usecs", dim, k, N);
            }
        }
    }}

    uniformFile.close();
    gaussianFile.close();
}