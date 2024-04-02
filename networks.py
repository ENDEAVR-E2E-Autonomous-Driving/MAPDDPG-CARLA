import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
from cbam import CBAM

"""
Critic Network
"""
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        # Convolution layers (no pooling)
        # conv2d(in_channels (3 for rgb), out_channels (number of filters), kernel_size (filter size), stride)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2)

        # Fully connected layers
        self.flatten_size = 128*28*38 # last 3 dimensions of pooling layer multiplied together (flattening)
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
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

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
        x = torch.cat((x, vehicle_x, action_x), dim=1) # along feature dimension (dim=1 for batch data)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # Q value stream
        Q = self.q_value_stream(x)

        return Q


"""
Actor Network with 
"""
class Actor(nn.Module):
    def __init__(self, seq_len, num_gru_layers, hidden_size):
        super(Actor, self).__init__()

        # Convolution layers (no pooling)
        # conv2d(in_channels (3 for rgb), out_channels (number of filters), kernel_size (filter size), stride)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2)

        # Convolutional block attention mechanism
        self.cbam = CBAM(in_channels=128)

        # GRU layer
        self.gru = nn.GRU(input_size=128*28*38, hidden_size=hidden_size, num_layers=num_gru_layers, batch_first=True)

        # fully connected layers
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=416)

        # Sensor and vehicle state layers
        self.v_fc1 = nn.Linear(in_features=29, out_features=128)
        self.v_fc2 = nn.Linear(in_features=128, out_features=96)

        # Combined fully connected layers
        self.final_fc1 = nn.Linear(512, 512)
        self.final_fc2 = nn.Linear(512, 512)
        self.final_fc3 = nn.Linear(512, 3)

    def forward(self, images, vehicle_state):
        batch_size, seq_len, c, h, w = images.size()

        # process each frame individually
        gru_input = torch.zeros(batch_size, seq_len, self.gru.input_size, device=images.device)
        for t in range(seq_len):
            frame = images[:, t] # get the t-th frame in the sequence

            # forward pass through conv layers and cbam
            x = F.relu(self.conv1(frame))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = F.relu(self.conv5(x))

            # apply CBAM to feature map
            cbam_out = self.cbam(x)
            cbam_out = cbam_out.view(batch_size, -1) # flatten cbam output for GRU
            gru_input[:, t] = cbam_out
        
        # pass sequence through gru
        gru_out, _ = self.gru(gru_input)
        
        # use only the output from the last GRU step for decision making
        gru_out = gru_out[:, -1]

        # forward pass through fc layers
        x = F.relu(self.fc1(gru_out))
        x = F.relu(self.fc2(x))

        # process vehicle states
        vehicle_x = F.relu(self.v_fc1(vehicle_state))
        vehicle_x = F.relu(self.v_fc2(vehicle_x))

        # concatenate vehicle state with learned features
        x = torch.cat((x, vehicle_x), dim=1)

        # final forward pass through fc layers
        x = F.relu(self.final_fc1(x))
        x = F.relu(self.final_fc2(x))
        actions = self.final_fc3(x)

        return actions


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else torch.cpu())
    # # print(device)
    # # print()
    # # testShapeNN = Critic(1).to(device)

    # img = cv2.imread('testImage.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RG
    #     transforms.ToTensor()
    # ])
    # tensor = transform(img).to(device)
    # print(tensor.shape)
    # # output = testShapeNN.forward(tensor)

    # # print(output.shape)

    # print(device)
    # test_shape = Actor().to(device)
    # test_shape.forward(tensor)