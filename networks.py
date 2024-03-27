import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2

class Critic(nn.Module):
    def __init__(self, action_dim):
        super(Critic, self).__init__()

        # Convolution and pooling layers
        # conv2d(in_channels (3 for rgb), out_channels (number of filters), kernel_size (filter size), stride)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=6, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        ...
    
    def forward(self, state):
        state = state.float() / 255.0

        x = F.relu(self.pool1(self.conv1(state)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))

        return x
    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print()
    testShapeNN = Critic(1).to(device)

    img = cv2.imread('testImage.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    tensor = transform(img).to(device)
    print(tensor.shape)
    output = testShapeNN.forward(tensor)

    print(output.shape)