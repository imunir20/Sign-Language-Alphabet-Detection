import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class VGG16(nn.Module):
    def __init__(self, inputChannels=3, numClasses=25):
        super(VGG16, self).__init__()
        self.inputChannels = inputChannels
        self.convolutionLayers = self.buildLayers()
        self.fullyConnectedLayers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, numClasses)
            )
        # VGG-16 architecture layers barebones
        # 0 represents a max pool layer

    
    def forward(self, x):
        x = self.convolutionLayers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fullyConnectedLayers(x)
        return x
        
    def buildLayers(self):
        layers = []
        inputChannels = self.inputChannels
        #vgg_architecture = [64, 64, 0, 128, 128, 0, 256, 256, 256, 0, 512, 512, 512, 0, 512, 512, 512, 0]
        vgg_architecture = [8, 8, 0, 16, 16, 0]

        for x in vgg_architecture:
            if x != 0:  # Convolution layer
                outputChannels = x
                layers += [nn.Conv2d(in_channels=inputChannels, out_channels=outputChannels,
                                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(x), nn.ReLU()]
                inputChannels = x
            elif x == 0:  # Max pooling layer
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        
        return nn.Sequential(*layers)
    
    def trainNetwork(self, num_epochs, trainLoader):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            overallLoss = 0.0
            print('Starting epoch', epoch)
            
            for i, data in enumerate(trainLoader, 0):
                print(data)
                inputs, labels = data
                optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                overallLoss += loss.item()
                if i % 2000 == 1999:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    overallLoss = 0.0

        print('Completed Training')
        torch.save(self.state_dict(), r'C:\Users\imuni\OneDrive\Desktop\GMU\Classes\CS\CS482\finalproj\models')
    
    def evaluateNetwork(self, dataLoader):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in dataLoader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
        
        
    def displayResults(dataLoader):
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1


        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

