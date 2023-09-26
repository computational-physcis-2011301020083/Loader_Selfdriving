#include "src/model.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>

using keras2cpp::Model;
using keras2cpp::Tensor;

//#定义点
class Point {
public:
    double x;
    double y;

    Point(double x, double y) {
        this->x = x;
        this->y = y;
    }
};

bool ccw(Point A, Point B, Point C) {
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x);
}

//#判断两个线段是否相交
bool intersect(Point A, Point B, Point C, Point D) {
    return ccw(A, C, D) != ccw(B, C, D) && ccw(A, B, C) != ccw(A, B, D);
}

//#判断p1 p2 p2夹角正负，左为负，右为正
double LoaderPM(Point A, Point B) {
    return A.x * B.y - A.y * B.x;
}

//#求p2p1 p2p3夹角
double Angle(Point A, Point B) {
    double cosA = (A.x * B.x + A.y * B.y) / (sqrt(A.x * A.x + A.y * A.y) * sqrt(B.x * B.x + B.y * B.y));
    return acos(cosA);
}


int main() {
    std::cout << "Begin" << std::endl;
    Point p0(1, 2); //#装载机坐标
    Point p1(5, 2); //#装载机附近三点路径坐标，按先后顺序p1 p2 p3
    Point p2(2.5, 3);
    Point p3(-1.1, 5);
    Point p12(p2.x - p1.x, p2.y - p1.y);
    Point p21(p1.x - p2.x, p1.y - p2.y);
    Point p23(p3.x - p2.x, p3.y - p2.y);
    double A0 = 1.2; //#车身夹角，单位弧度，左为负，右为正
    double dt = 0.01; //#更新时间间隔
    double v;

    std::cout << "Calculate v" << std::endl;
    //#判断状态，计算角速度，左为负，右为负
    if (LoaderPM(p21, p23) * A0 < 0) {
        v = -1.1 * (A0 / abs(A0)) * (M_PI - abs(A0)) / dt;
    } else if (LoaderPM(p21, p23) * A0 > 0 && intersect(p0, p2, p1, p3)) {
        v = -1.1 * (A0 / abs(A0)) * std::clamp(Angle(p21, p23) - A0, 0.0, 4.0) / dt;
    } else {
        v = 1.1 * (A0 / abs(A0)) * std::clamp(A0 - Angle(p21, p23), 0.0, 4.0) / dt;
    }
    //#限制v的大小
    double maxv = 0.2;
    v = std::clamp(v, -maxv,maxv);

    // 加载模型，预测相应的PWM值
    // Initialize model.
    std::cout << "Predict PWM" << std::endl;
    auto model = Model::load("LSTM_VPWM.model");

    // Create a 1D Tensor on length 10 for input data.
    Tensor in{1,1};
    in.data_[0] = v;

    // Run prediction.
    Tensor PWM = model(in);
    std::cout << v << std::endl;
    PWM.print(); //PWM正为左转，负为右转
    return 0;
}
