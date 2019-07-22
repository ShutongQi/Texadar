import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms
import xlrd
import scipy.io as scio

class MyDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, root, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        fh = open(root, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        data = []  # 创建一个名为data的空列表，一会儿用来装东西
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            data.append((words[0], int(words[1])))  # 把txt里的内容读入data列表保存，具体是data几要看txt内容而定，data[0]是图片信息，data[1]是lable
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.data[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        '''
        root = 'D:\ADS-B\code'
        # fn = r'\data1.xlsx'
        filename = xlrd.open_workbook(root + fn)
        sheet = filename.sheet_by_index(0)
        num = sheet.col_values(0)
        '''
        # root = 'D:\ADS-B\code'
        filename = fn
        mat = scio.loadmat(filename)
        '''
                if 'datasec' in mat:
            num = mat['datasec']
        else:
            num = mat['secdata']
        '''
        num = mat['data']
        if self.transform is not None:
            num = self.transform(num)  # 是否进行transform
        return num, label # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容


    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.data)


def data_tf(x):
    x = x[:1023]
    x = np.array(x, dtype='float32')
    ave = np.mean(x)
    stan = np.std(x)
    # x = (x + 0.5) / 0.5
    x=(x-ave)/stan
    x = x.reshape((-1,))

    x = torch.from_numpy(x)

    return x


# train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True) # 重新载入数据集，申明定义的数据变换
# test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)
# root='D:\ADS-B\code'
# trainadr='\trainset.txt'
# testadr='\testset.txt'
train_set = MyDataset('./trainset.txt', transform=data_tf)
test_set = MyDataset('./testset.txt', transform=data_tf)

from torch.utils.data import DataLoader
train_data = DataLoader(train_set, batch_size=16, shuffle=True)
test_data = DataLoader(test_set, batch_size=4, shuffle=False)

a, a_label = next(iter(train_data))

net = nn.Sequential(
    nn.Linear(1024, 256),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.Dropout(0.3),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.Dropout(0.3),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.Dropout(0.1),
    nn.ReLU(),
    nn.Linear(16, 10),

)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 1e-3, momentum=0.9) # 使用随机梯度下降，学习率 0.1
# optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-4, alpha=0.9)
# optimizer = torch.optim.Adadelta(net.parameters(), rho=0.9)

# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []

for e in range(1000):
    train_loss = 0
    train_acc = 0
    #begin training
    net.train()
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        train_acc += acc

    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))
    eval_loss = 0
    eval_acc = 0
    net.eval()
    #begin predicting
    for im, label in test_data:
        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = criterion(out, label)
        eval_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_data), train_acc / len(train_data),
                  eval_loss / len(test_data), eval_acc / len(test_data)))
torch.save(net,'./modelnew.pth')
import matplotlib.pyplot as plt

plt.title('train loss')
plt.plot(np.arange(len(losses)), losses)
plt.show()

plt.plot(np.arange(len(acces)), acces)
plt.title('train acc')
plt.show()
plt.plot(np.arange(len(eval_losses)), eval_losses)
plt.title('test loss')
plt.show()
plt.plot(np.arange(len(eval_acces)), eval_acces)
plt.title('test acc')
plt.show()