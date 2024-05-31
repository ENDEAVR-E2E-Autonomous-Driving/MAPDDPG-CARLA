import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2

"""
Convolution Block Attention Mechanism
"""
# -------------------------------------
"""
Channel Attention
"""
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16) -> None:
        super(ChannelAttention, self).__init__()
        # reduction_ratio reduces number of channels in the hidden layer of the MLP to decrease number of parameters
        # apply max and average pooling to the spatial dimensions
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # shared MLP (multi-layer perceptron)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # accounting for sequence length for temporal information with GRU
        batch, c, h, w = x.size()
        # x = x.view(batch * seq_len, c, h, w,)

        maxpooled = self.maxpool(x)
        avgpooled = self.avgpool(x)

        # flatten before passing through MLP
        maxpooled = maxpooled.view(x.size(0), -1)
        avgpooled = avgpooled.view(x.size(0), -1)

        # pass through MLP
        max_out = self.shared_mlp(maxpooled)
        avg_out = self.shared_mlp(avgpooled)

        # Concatenate channel-wise features
        out = max_out + avg_out
        out = out.view(x.size(0), -1, 1, 1) # reshape back to the original size but with spatial dims 1x1

        attention = self.sigmoid(out)

        # reshape back to [batch, seq len, channels, height, width]
        # attention = attention.view(batch, seq_len, c, 1, 1)
        attention = attention.view(batch, c, 1, 1)


        return attention
    

"""
Spatial Attention
Consider modifying later
"""
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7) -> None:
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # accounting for sequence length dimension
        batch, c, h, w = x.size()
        # x = x.view(batch * seq_len, c, h, w,)

        # average and max pooling across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)

        x = self.conv(x)
        x = self.sigmoid(x)

        # reshape back to [batch, seq len, channels, height, width]
        # attention = x.view(batch, seq_len, 1, h, w)
        attention = x.view(batch, 1, h, w)


        return attention
    

"""
Convolutional Block Attention Mechanism
"""
class CBAM(nn.Module):
    def __init__(self, in_channels) -> None:
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, f):
        batch, seq_len, c, h, w = f.size()

        # process each frame in the sequence individually
        cbam_outputs = []
        for t in range(seq_len):
            frame = f[:, t] # get the t-th frame in the sequence
            channel_attention = self.channel_attention(frame)
            f_channel = channel_attention * frame # element-wise multiplication
            spatial_attention = self.spatial_attention(f_channel)
            cbam_output = spatial_attention * f_channel
            cbam_outputs.append(cbam_output)
        
        cbam_outputs = torch.stack(cbam_outputs, dim=1)

        return cbam_outputs
    
# ----------------------------------------------------------
"""
Actor-Critic Networks
"""

# ----------------------------------------------------------

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
        self.flatten_size = 128*4*11 # last 3 dimensions of pooling layer multiplied together (flattening)
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 416) # last FC layer is 416 to accomodate concatenation with other representations

        # Fully connected layers for vehicle ego state
        self.v_fc1 = nn.Linear(10, 128)
        self.v_fc2 = nn.Linear(128, 64)

        # Fully connected layers for actions
        self.a_fc1 = nn.Linear(3, 128)
        self.a_fc2 = nn.Linear(128, 32)
        
        # FC Layers with features from fc2 concatenated with vehicle ego state and actions from the actor network
        self.fc3 = nn.Linear(512, 512) 
        self.fc4 = nn.Linear(512, 512)
        self.q_value_stream = nn.Linear(512, 1) # returns expected total reward for current state-action pair
    
    def forward(self, state, vehicle_ego_state, actions):
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
        x = F.relu(self.fc2(x))

        # ensure vehicle state is 2D (add a batch dimension if needed)
        if len(vehicle_ego_state.size()) == 1:
            vehicle_ego_state = vehicle_ego_state.unsqueeze(0)

        if len(actions.size()) == 1:
            actions = actions.unsqueeze(0)

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
Actor Network with convolutional block attention mechanism and GRU layer
"""
class Actor(nn.Module):
    def __init__(self, num_gru_layers=1, hidden_size=256):
        super(Actor, self).__init__()

        # convolution layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=2)

        # convolutional block attention mechanism
        self.cbam = CBAM(in_channels=128)

        # GRU layer
        self.gru = nn.GRU(input_size=128 * 4 * 11, hidden_size=hidden_size, num_layers=num_gru_layers, batch_first=True)

        # fully connected layers
        self.fc1 = nn.Linear(hidden_size, 512)
        self.fc2 = nn.Linear(512, 416)

        # sensor and vehicle state layers
        self.v_fc1 = nn.Linear(10, 128)
        self.v_fc2 = nn.Linear(128, 96)

        # combined fully connected layers
        self.final_fc1 = nn.Linear(512, 512)
        self.final_fc2 = nn.Linear(512, 512)
        self.final_fc3 = nn.Linear(512, 3)  # throttle, steer, brake

    def forward(self, images, vehicle_state):
        batch_size, seq_len, c, h, w = images.size()

        # flatten the image sequence into one batch dimension
        images = images.view(batch_size * seq_len, c, h, w)

        # pass through convolution layers
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        print(x.size())

        # apply CBAM
        x = x.view(batch_size, seq_len, 128, 4, 11)
        x = self.cbam(x)

        # reshape for GRU: (batch_size, seq_len, input_size)
        x = x.reshape(batch_size, seq_len, 128*4*11)

        # pass through GRU
        gru_out, _ = self.gru(x)

        # use only the output from the last GRU step
        gru_out = gru_out[:, -1]

        # pass through fully connected layers
        x = F.relu(self.fc1(gru_out))
        x = F.relu(self.fc2(x))

        # ensure vehicle state is 2D (add a batch dimension if needed)
        if len(vehicle_state.size()) == 1:
            vehicle_state = vehicle_state.unsqueeze(0)

        # process vehicle states
        vehicle_x = F.relu(self.v_fc1(vehicle_state))
        vehicle_x = F.relu(self.v_fc2(vehicle_x))

        # concatenate with learned features and final pass through FC layers
        x = torch.cat((x, vehicle_x), dim=1)
        x = F.relu(self.final_fc1(x))
        x = F.relu(self.final_fc2(x))
        action_preds = self.final_fc3(x)

        # map predictions to action ranges
        throttle = torch.sigmoid(action_preds[:, 0])
        steer = torch.tanh(action_preds[:, 1])
        brake = torch.sigmoid(action_preds[:, 2])

        return throttle, steer, brake
