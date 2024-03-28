import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        # Convolution and pooling layers
        # conv2d(in_channels (3 for rgb), out_channels (number of filters), kernel_size (filter size), stride)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.flatten_size = 128*6*8 # last 3 dimensions of pooling layer multiplied together (flattening)
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 416) # last FC layer is 416 to accomodate concatenation with other representations

        # Fully connected layers for vehicle ego state
        self.v_fc1 = nn.Linear(29, 128)
        self.v_fc2 = nn.Linear(128, 64)

        # Fully connected layers for actions
        self.a_fc1 = nn.Linear(3, 128)
        self.a_fc2 = nn.Linear(128, 32)
        
        # FC Layers with features from fc2 concatenated with vehicle ego state and actions from the actor network
        self.fc3 = nn.Linear(512, 512) 
        self.fc4 = nn.Linear(512, 512)
        self.q_value_stream = nn.Linear(512, 1) # returns expected total reward for current state-action pair
    
    def forward(self, state, vehicle_ego_state, actions):
        state = state.float() / 255.0

        # Pass state image through conv and pooling layers
        # Shape of each layer output follows format: [batch_size, num channels, height, width]
        x = F.relu(self.pool1(self.conv1(state)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))

        # Flatten and pass through fully connected layers
        x = x.reshape(x.size(0), -1) # flatten by keeping the batch dim and transforming features: channels*height*width
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2)

        # Pass vehicle state through their respective fc layers
        vehicle_x = F.relu(self.v_fc1(vehicle_ego_state))
        vehicle_x = F.relu(self.v_fc2(vehicle_x))

        # Pass actions through their respective fc layers
        action_x = F.relu(self.a_fc1(actions))
        action_x = F.relu(self.a_fc2(action_x))

        # Concatenate features from FC layers with vehicle ego state and actor actions, and pass through final FC layers
        x = torch.cat((x, vehicle_x, action_x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # Q value stream
        Q = self.q_value_stream(x)

        return Q
    

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