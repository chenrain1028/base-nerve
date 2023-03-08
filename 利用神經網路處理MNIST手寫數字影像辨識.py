import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as plt_font
twfont1=plt_font.FontProperties(fname="字型/kaiu.ttf")
from IPython import display
import torch
#建立神經網路用
import torch.nn as nn
import torch.nn.functional as F
#載入優化器
from torch import optim
#預處理資料用
from torch.utils.data import Dataset,DataLoader


# 檢查 PyTorch 版本
print("PyTorch version:", torch.__version__)
# 檢查有哪些 GPU 可用
print("Available GPUs:", torch.cuda.device_count())
# 檢查目前使用的裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Currently using", device)

class MyDataset(Dataset):
  def __init__(self,Train=True):
    if Train==True:
      data=np.loadtxt("MNIST/mnist_train.csv",delimiter=",")
    else:
      data=np.loadtxt("MNIST/mnist_test.csv",delimiter=",")
    self.data = torch.tensor(data[:,1:]/255)        #這邊做簡單的歸一化，故這邊統一除以255
    self.label = torch.tensor(data[:,0])
  def __getitem__(self, index):
    return self.data[index], self.label[index]
  def __len__(self):
    return len(self.data)

#實體化訓練和測試DataSet和DataLoader
TrainDS=MyDataset(Train=True)
TestDS=MyDataset(Train=False)
TrainDL=DataLoader(dataset=TrainDS,batch_size=200,shuffle=True)
TestDL=DataLoader(dataset=TestDS,batch_size=len(TestDS),shuffle=True)
#顯示手寫數字照片
plt.imshow(TestDS[0][0].reshape( (28,-1) ),cmap="gray" )    #2維的矩陣
print("手寫數字",TestDS[0][1])

#建立神經網路類別
class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1=nn.Linear(784,128)     #輸入層有784個特徵
    self.Bn1=nn.BatchNorm1d(128)        #用BatchNormld對第一個隱藏層的神經元作歸一化
    self.Dp1=nn.Dropout(0.2)            #用dropout隨機讓一定的神經元休息不訓練
    self.fc2=nn.Linear(128,64)
    self.Bn2=nn.BatchNorm1d(64)
    self.fc3=nn.Linear(64,10)
  def forward(self,x):
    x=self.Dp1(self.Bn1(torch.relu(self.fc1(x))))
    x=self.Bn2(torch.relu(self.fc2(x)))
    x=self.fc3(x)
    return x
#實體化神經網路，指定損失函數和優化器
net=Net()
optimizer = optim.SGD(net.parameters(), lr=0.01)
LossFun=nn.CrossEntropyLoss()
Step_L=[]
#開始訓練神經網路
net.train()
for i in range(150):
  for data,label in TrainDL:
    optimizer.zero_grad()
    Yh=net(data.float())
    loss=LossFun(Yh,label.long())
    Step_L.append(float(loss))
    loss.backward()
    optimizer.step()
  print("訓練回合：",i+1,"損失值：",loss)
  display.clear_output(wait=True)
#觀察訓練過程中的損失函數Loss變化
plt.figure(figsize=(8,5))
plt.title("Loss隨訓練次數的變化",fontproperties=twfont1,fontsize=20)
plt.xlabel("訓練次數",fontproperties=twfont1,fontsize=20)
plt.ylabel("Loss值",fontproperties=twfont1,fontsize=20)
plt.plot(Step_L,":o")
plt.show()
net.eval()
with torch.no_grad():
  data,label=next(TestDL.__iter__())
  output = net(data.float())
print("驗證資料準確度：",(torch.argmax(output,dim=1)==label).sum().item()*100/len(TestDS),"%")