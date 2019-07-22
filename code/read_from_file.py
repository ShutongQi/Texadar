# --------read-----------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import serial
import sys,os
from seglearn.transform import SegmentX
from seglearn.feature_functions import maximum,minimum
from sklearn.preprocessing import normalize,MinMaxScaler

batchSize = 2048

FILE = '../data/gesture/' + sys.argv[1] + '.txt'

with open(FILE,'r') as file:
    y = np.loadtxt(file,delimiter=',')
print(y.shape)
y = np.reshape(y,(-1,batchSize))
y = np.reshape(y,(1,-1))
print(y.shape)
segment = SegmentX(width=batchSize, step=batchSize, shuffle=False, random_state=None, order='F')
y = segment.transform(y)[0]
print(y.shape)
maxy = maximum(y)
miny = minimum(y)
print(maxy.shape)
print(miny.shape)
# y=np.transpose(y)
# scaler = MinMaxScaler()
# scaler.fit(y)
# y=scaler.transform(y)
# y=np.transpose(y)
print(y)

# mean_y = np.mean(y, axis=1)
# output = np.transpose(np.tile(mean_y, (batchSize, 1)))
output = y


# ---------------------visualization-------------------------
fig, ax = plt.subplots()
line, = ax.plot(np.random.rand(10))

idx=0
def update(data):
    line.set_ydata(data)
    return line,

def run(data):
    t,y = data
    line.set_data(t, y)
    return line,

def data_gen():
    pltx = np.array([])
    plty = np.array([])
    win_max = output.max()+np.ptp(output)*0.1
    win_min = output.min()-np.ptp(output)*0.1
    global idx
    for i in output:
        try:
            dat = i
            
        except:
            dat = np.zeros(batchSize)

        plt.axis([idx-batchSize*10, idx+batchSize, win_min, win_max])
        for i in range(batchSize):
            pltx=np.append(pltx,idx)
            idx += 1
        # print(dat)
        plty = np.concatenate((plty,dat))
        pltx = pltx[-batchSize*10:]
        plty = plty[-batchSize*10:]
        plt.vlines(idx,win_min,win_max,colors="r")
        yield pltx, plty

ani = animation.FuncAnimation(fig, run, data_gen, interval=200)
plt.show()