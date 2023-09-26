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
