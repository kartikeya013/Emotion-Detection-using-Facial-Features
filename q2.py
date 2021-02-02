import torch
import numpy as np
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
xTrainPath = str(sys.argv[1])
xTestPath = str(sys.argv[2])
outputFile = str(sys.argv[3])
def getData(xPath):
  x_in = np.genfromtxt(xPath ,delimiter=",")
  x = x_in[:,1:]
  y = x_in[:,0]
  # x = x/255
  return x,y

xTrain_in,yTrain = getData(xTrainPath)
xTest_in,yTest = getData(xTestPath)
r = len(np.unique(yTrain))
n = np.shape(xTrain_in)[1]
numFeatures = n
m = np.shape(xTrain_in)[0]
# print("data read")
xTrainTrans = np.reshape(xTrain_in,(m,48,48))
print(np.shape(xTrainTrans))

# final_train_data = []
# final_target_train = []
# for i in (range(m)):
#     image = xTrainTrans[i]
#     final_train_data.append(image)
#     final_train_data.append(rotate(image, angle=45, mode = 'wrap'))
#     final_train_data.append(np.fliplr(image))
#     final_train_data.append(np.flipud(image))
#     final_train_data.append(random_noise(image,var=0.2**2))
#     for j in range(5):
#         final_target_train.append(yTrain[i])
# print("data augmented")
from torchvision import transforms 
from torchvision.transforms import Compose 
# model = model.cuda() ##Check whether cuda is available or not
xTrain = torch.tensor(xTrain_in,dtype=torch.float).to(device)
yTrain = torch.tensor(yTrain,dtype=torch.long).to(device)

# xTrain = xTrain.view(m,1,48,48)
# train_transform = Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomCrop(40, padding=4),
#     # transforms.ToTensor(),
# ])
# xTrain = train_transform(xTrain)
# print(xTrain.size())

model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(64),
        torch.nn.MaxPool2d(2, stride=2),
        torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(128),
        torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(128),
        torch.nn.MaxPool2d(2, stride=2),
        torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(256),
        torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(256),
        torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(256),
        torch.nn.MaxPool2d(2, stride=2),
        torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(512),
        torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(512),
        torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(512),
        torch.nn.MaxPool2d(2, stride=2),
        torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(512),
        torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(512),
        torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(512),
        torch.nn.MaxPool2d(2, stride=2),
        torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=3),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(1024),
        torch.nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(1024),
        torch.nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(256),
        torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(512),
        torch.nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(128),
        torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(256),
        # torch.nn.MaxPool2d(2, stride=2),
        # torch.nn.MaxPool2d(2, stride=2),
        # torch.nn.AvgPool2d(1, stride=1),
        torch.nn.Flatten(),
        # torch.nn.Dropout(0.5),
        torch.nn.Linear(in_features=1024, out_features=256),
        torch.nn.ReLU(),
        # torch.nn.Dropout(0.5),
        torch.nn.Linear(in_features=256, out_features=128),
        torch.nn.ReLU(),
        # torch.nn.Dropout(0.5),
        torch.nn.Linear(in_features=128, out_features=64),
        torch.nn.ReLU(),
        # torch.nn.Dropout(0.5),
        torch.nn.Linear(in_features=64, out_features=r)
    )

model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.9,weight_decay=0.0001) # stochastic gradient descent
criterion = torch.nn.CrossEntropyLoss()

xTest = torch.tensor(xTest_in,dtype=torch.float).to(device)
yTest = torch.tensor(yTest,dtype=torch.long).to(device)
# xTest = xTest.view(xTest.size(0),1,48,48)
# xTest =  train_transform(xTest)
# xTest = xTest.view(xTest.size(0),1,48,48)




# print("Training Started")
EPOCHS = 2000
M = 64
model = model.to(device)
x = False
prevJ = 0
earlyStop = 0
epoch = 0

while epoch<32 and earlyStop<10:
    epoch += 1 
    for i in range(0,int(xTrain.size(0)/M)):
      # optimizer.zero_grad()
      x_train = (xTrain[(i)*M:(i)*M + M,:]).to(device)
      y_train = (yTrain[i*M:i*M + M]).to(device)
      # x_train = x_train.unsqueeze(1)
      x_train = x_train.view(M,1,48,48)
      outputs = model(x_train)
      loss = criterion(outputs, y_train)
      l1_regularization = 0
      # for param in model.parameters():
      #   l1_regularization += torch.norm(param, 2)**2
      
      # loss += l1_regularization*0.001
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # if (i+1) % 10 == 0:
      #   print (epoch,'Loss: {:.4f}, Accuracy'.format(loss.item()),calAccuracy(model))
      if abs(loss.item()-prevJ) < 1e-5:
        earlyStop += 10
      prevJ = loss.item()
      

# with torch.no_grad():
#     # print("h")
#     correct = 0
#     total = 0
#     M = 100
#     # print(m/M)
#     for i in range(0,int(m/M)):
#       # print(i)
#       # optimizer.zero_grad()
#       x_train = (xTrain[(i)*M:(i)*M + M,:]).to(device)
#       y_train = (yTrain[i*M:i*M + M]).to(device)
#       # x_train = x_train.unsqueeze(1)
#       x_train = x_train.view(x_train.size(0),1,48,48)
#       outputs = model(x_train)
#       y_predicted = torch.argmax(outputs,dim=1)
#       # print(y_predicted)
#       # _, predicted = torch.max(outputs, 1)
#       total += x_train.size(0)
#       for j in range(len(y_predicted)):
#         if y_train[j] == y_predicted[j]:
#           correct += 1
#     print(correct,total)
#     print('Train Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
yPredicted = []

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

with torch.no_grad():
    # print("h")
    correct = 0
    total = 0
    M = 64
    # print(m/M)
    for i in range(0,int(xTest.size(0)/M)+1):
      # print(i)
      # optimizer.zero_grad()
      x_train = (xTest[(i)*M:(i)*M + M,:])
      y_train = (yTest[i*M:i*M + M])
      # x_train = x_train.unsqueeze(1)
      x_train = x_train.view(x_train.size(0),1,48,48)
      outputs = model(x_train)
      y_predicted = torch.argmax(outputs,dim=1)
      # print(y_predicted)
      # _, predicted = torch.max(outputs, 1)
      total += x_train.size(0)
      for j in range(len(y_predicted)):
        yPredicted.append(y_predicted[j])
    
write_predictions(outputFile,yPredicted)