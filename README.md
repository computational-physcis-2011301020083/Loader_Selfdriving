The project is compatible with Keras 2.x (all versions) and Python 3.x

# Example

python_model.py: generate  NN model and export to suitable format.

```python
import numpy as np
from keras import Sequential
from keras.layers import Dense
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, GRU,LSTM
import math
import tensorflow as tf
from decimal import Decimal

#create random data
def get_XY(dat, time_steps):
    # Indices of target array
    Y_ind = np.arange(time_steps, dat.shape[0], time_steps)
    Y = dat[Y_ind,:]
    # Prepare X
    rows_x = Y.shape[0]
    X = dat[range(time_steps*rows_x),:]
    X = np.reshape(X, (rows_x, time_steps, 2))    
    return X, Y

itxt="5.txt"
with open(itxt,"r") as f:
    datalist = f.readlines()
datanp=np.zeros((len(datalist),11))
count=0
for i in datalist:
    datai=i.split("\n")[0].split("\t")
    for j in range(11):
        datanp[count:count+1,j]=Decimal(datai[j])
    count=count+1
datanp=datanp[2000:,]
lpwm=datanp[:,1:2]*datanp[:,3:4]
rpwm=datanp[:,2:3]*datanp[:,4:5]*(-1)
pwm=lpwm+rpwm
v=datanp[:,6:7]
train_data = np.hstack((v,pwm))
time_steps = 1
trainX, trainY = get_XY(train_data, time_steps)
#trainX=np.squeeze(trainX[:,:,:])
#print(trainX.shape,trainY.shape)

def create_RNN(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(LSTM(hidden_units, input_shape=input_shape, activation=activation[0]))
    model.add(Dense(units=dense_units[0], activation=activation[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
model = create_RNN(hidden_units=200, dense_units=[1], input_shape=(time_steps,1), activation=['relu', 'linear']) 

model.fit(trainX[:,:,0:1], trainY[:,1:2], epochs=100,verbose=False)

#predict
data = np.array([[[-0.2]]])
prediction = model.predict(data)
print(prediction)

#save model
from keras2cpp import export_model
export_model(model, 'LSTM_VPWM.model')
```

cpp_model.cc: algorithm for self-driving

```c++
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

//#判断p1 p2 p2夹角正负
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
    double A0 = 1.2;
    double dt = 0.01;
    double v;

    std::cout << "Calculate v" << std::endl;
    //#判断状态，计算角度
    if (LoaderPM(p12, p23) * A0 < 0) {
        v = -1.1 * (A0 / abs(A0)) * (M_PI - abs(A0)) / dt;
    } else if (LoaderPM(p12, p23) * A0 > 0 && intersect(p0, p2, p1, p3)) {
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
    //cout << "Predict PWM";
    auto model = Model::load("LSTM_VPWM.model");

    // Create a 1D Tensor on length 10 for input data.
    Tensor in{1,1};
    in.data_[0] = v;

    // Run prediction.
    Tensor PWM = model(in);
    //cout << "PWM";
    std::cout << v << std::endl;
    PWM.print();
    return 0;
}
```

# How to build and run

*Tested with Keras 2.2.1, Python 3.6*

```bash
$ git clone https://github.com/computational-physcis-2011301020083/Loader_Selfdriving.git
$ cd Loader_Selfdriving
$ mkdir build && cd build
$cp ../5.txt .  
$ python3 ../python_model.py

$ cmake ..
$ cmake --build .
$ ./keras2cpp
```
or 

```bash
$ git clone https://github.com/computational-physcis-2011301020083/Loader_Selfdriving.git
$ cd Loader_Selfdriving
$ cd build
$ cmake ..
$ cmake --build .
$ ./keras2cpp
```

