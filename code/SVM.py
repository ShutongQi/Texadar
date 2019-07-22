import numpy as np
import scipy.io as scio
from sklearn import svm
from sklearn.model_selection import train_test_split
root = './dataset.txt'
fh = open(root, 'r')
file = []
for line in fh:
    line = line.rstrip()
    words = line.split()
    file.append((words[0], int(words[1])))

data = []
for i in range(300):
    fn = file[i][0]
    label = file[i][1]
    mat = scio.loadmat(fn)
    temp = mat['data']
    temp = temp[0]
    temp = np.append(temp,label)
    data.append(temp)
data=np.array(data)

# print(data)
fftnum = 512
x, y = np.split(data, (fftnum,), axis=1)
# x = x[:, :2]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.6)
clf = svm.SVC(C=0.6, kernel='poly', gamma=10, decision_function_shape='ovo')
clf.fit(x_train, y_train.ravel())

print (clf.score(x_train, y_train) ) # 精度
y_hat = clf.predict(x_train)
#show_accuracy(y_hat, y_train, '训练集')
print(clf.score(x_test, y_test))
y_hat = clf.predict(x_test)
#show_accuracy(y_hat, y_test, '测试集')
