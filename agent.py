import torch
import torch.autograd
import torch.optim as optim
import torch.nn.functional as F
from torchrl.data import ListStorage, PrioritizedReplayBuffer
from mapddpg.networks import *
from mapddpg.networks import Actor, Critic
from mapddpg.ou_noise import OUActionNoise

"""
Agent class that encapsulates functionalites of the agent, such as:
- choosing actions
- learning from chosen actions
- interacting with the environment
- updating its networks with sampled experiences
"""
class VehicleAgent:
    def __init__(self, 
                 actor: Actor,
                 critic: Critic,
                 actor_target: Actor,
                 critic_target: Critic,
                 actor_optimizer: optim.Adam,
                 critic_optimizer: optim.Adam,
                 throttle_and_brake_noise: OUActionNoise,
                 steer_noise: OUActionNoise,
                 gamma: float,
                 tau: float,
                 batch_size: int,
                 device,
                 buffer_size=1000,
                 buffer_alpha=0.6,
                 buffer_eps=1e-4,
                 buffer_beta=0.4,
                 target_update_freq=10
                 ) -> None:
        
        self.device = device

        # actor-critic components
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.actor_target = actor_target.to(self.device)
        self.critic_target = critic_target.to(self.device)
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.throttle_and_brake_noise = throttle_and_brake_noise
        self.steer_noise = steer_noise
        
        # replay buffer components
        self.batch_size = batch_size
        self.buffer_eps = buffer_eps
        self.buffer_alpha = buffer_alpha
        self.buffer_beta = buffer_beta
        self.prioritized_replay_buffer = PrioritizedReplayBuffer(alpha=buffer_alpha, beta=buffer_beta, storage=ListStorage(buffer_size))

        # other hyperparameters
        self.gamma = gamma # discount factor
        self.tau = tau # temperature
        self.target_update_freq = target_update_freq
        self.update_counter = 0

    # select an action with the actor network
    def select_action(self, state_sequence, vehicle_ego_state):
        self.actor.eval() # sets actor to evaluation mode 

        with torch.no_grad():
            throttle, steer, brake = self.actor.forward(state_sequence, vehicle_ego_state)
        
        # convert to cpu then numpy for .clip() method and add noise for exploration
        throttle = throttle.cpu().numpy() + self.throttle_and_brake_noise()
        steer = steer.cpu().numpy() + self.steer_noise()
        brake = brake.cpu().numpy() + self.throttle_and_brake_noise()
        
        # ensure agent doesn't throttle and brake at the same time
        if throttle > 0.1:
            brake = 0.0
        elif brake > 0.1:
            throttle = 0.0

        # Ensure all actions stay within their respective bounds
        throttle_action = np.clip(throttle, 0, 1)
        steer_action = np.clip(steer, -1, 1)
        brake_action = np.clip(brake, 0, 1)

        if isinstance(throttle_action, np.ndarray):
            throttle_action = throttle_action[0]
        if isinstance(steer_action, np.ndarray):
            steer_action = steer_action[0]
        if isinstance(brake_action, np.ndarray):
            brake_action = brake_action[0]

        final_actions = (throttle_action, steer_action, brake_action)

        self.actor.train() # sets actor to training mode

        return torch.tensor(final_actions, dtype=torch.float32, device=self.device)

    

    # extract the last frame from a sequence tensor
    def get_last_frame(self, state_sequence):
        # state_sequence shape is [batch_size, seq_len, channels, height, width]
        return state_sequence[:, -1]

    # learn from a previously sampled batch of experiences
    def learn(self, states, vehicle_state, actions, rewards, next_states, next_vehicle_state, dones, weights, indices):
        print("Learning...")
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        # print(f"states dim: {states.size()}")
        # Before passing 'states' to the model, ensure the dimensions are correct
        states = states.squeeze(1)  # Remove the singleton dimension at position 1
        next_states = next_states.squeeze(1)

        # extract the last frame from the sequences (for critic only)
        critic_states = self.get_last_frame(states)
        critic_next_states = self.get_last_frame(next_states)

        # ensure the extracted frames have the right dimensions
        # critic_states = critic_states.permute(0, 3, 1, 2)
        # critic_next_states = critic_next_states.permute(0, 3, 1, 2)

        # compute target Q-values using the Bellman equation
        with torch.no_grad():
            throttle, steer, brake = self.actor_target(next_states, next_vehicle_state)
            # Concatenate these actions to form the complete action vector for each instance in the batch
            # Each of throttle, steer, brake should be a tensor of shape [batch_size, 1]
            # Concatenate along dimension 1 to form a [batch_size, 3] tensor
            target_actions = torch.cat((throttle.unsqueeze(1), steer.unsqueeze(1), brake.unsqueeze(1)), dim=1).to(self.device)
            # target_actions = torch.tensor(target_actions, dtype=torch.float32, device=self.device)
            target_Q_values = self.critic_target(critic_next_states, next_vehicle_state, target_actions).squeeze().to(self.device)
            # convert 'dones' to float and compute the expected Q-values
            dones_float = dones.to(torch.float32).to(self.device)  # convert boolean to float
            expected_Q_values = rewards + (self.gamma * target_Q_values * (1 - dones_float)).detach() # Ensure no gradient computation

        # compute current Q-values from the critic network
        current_Q_values = self.critic(critic_states, vehicle_state, actions).squeeze()

        # critic Loss: mean Squared Error between current Q-values and expected Q-values
        # multiply by importance sampling weights to offset bias from priority sampling
        critic_loss = (weights * F.mse_loss(current_Q_values.float(), expected_Q_values.float(), reduction='none')).mean()
        critic_loss.backward()
        self.critic_optimizer.step()

        # compute new priorities (absolute TD errors + a small epsilon)
        new_priorities = (torch.abs(current_Q_values - expected_Q_values) + self.buffer_eps).detach().cpu().numpy()
        self.update_priorities(indices, new_priorities)

        # actor Loss: mean of Q-values output by the critic for current policy's actions
        # negative sign because we want to maximize the critic's output (policy gradient ascent)
        throttle, steer, brake = self.actor(next_states, next_vehicle_state)
        current_actions = torch.cat((throttle.unsqueeze(1), steer.unsqueeze(1), brake.unsqueeze(1)), dim=1).to(self.device)
        # current_actions = torch.tensor(current_actions, dtype=torch.float32, device=self.device)
        actor_loss = -self.critic(critic_states, vehicle_state, current_actions).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft-update the target networks every 'target_update_freq' steps
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.soft_update(self.actor, self.actor_target, self.tau)
            self.soft_update(self.critic, self.critic_target, self.tau)

        print("Finished learning.")

    # soft update for copying parameters of actor-critic to target networks
    def soft_update(self, local_model, target_model, tau):
        """ Soft update model parameters """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    # compute TD-error for storing experiences where the priority is the TD-error
    def compute_td_error_and_Q_values(self, experience_tuple): 
        """
        computes TD-error, the current Q-values, and the expected Q-values
        """
        state_sequence, sensor_state, action, reward, next_state_sequence, next_vehicle_state, dones = experience_tuple

        critic_state = self.get_last_frame(state_sequence)
        critic_next_state = self.get_last_frame(next_state_sequence)

        with torch.no_grad():
            next_actions = self.actor_target(next_state_sequence, next_vehicle_state)
            next_actions = torch.tensor(next_actions, dtype=torch.float32, device=self.device)
            next_Q_values = self.critic_target(critic_next_state, next_vehicle_state, next_actions)
            expected_Q_values = reward + self.gamma * next_Q_values * (1 - dones)

        current_Q_values = self.critic(critic_state, sensor_state, action)

        td_error = torch.abs(current_Q_values - expected_Q_values)

        return td_error, current_Q_values, expected_Q_values

    
    def validate_experience(self, experience):
        expected_length = 7  # since you have seven items in each tuple
        if len(experience) != expected_length:
            # Iterate over each element in the tuple and print its type
            for i, item in enumerate(experience):
                print(f"Element {i} type: {type(item)}")
                # Optionally print the shape if the item is a numpy array or torch tensor
                if hasattr(item, 'shape'):
                    print(f"  Shape: {item.shape}")
            
            raise ValueError(f"Expected tuple of length {expected_length}, got {len(experience)}")

    
    # store experience in replay buffer
    def store_experience(self, experience_tuple, td_error):
        self.validate_experience(experience_tuple)
        # convert priority to a scalar
        priority = (td_error.abs() + self.buffer_eps).pow(self.buffer_alpha).item()

        # store the experience with priority
        index = self.prioritized_replay_buffer.add(experience_tuple)
        self.prioritized_replay_buffer.update_priority(index=index, priority=priority)


    # update priorities in the replay buffer after learning or after storing
    def update_priorities(self, indices, priorities):
        self.prioritized_replay_buffer.update_priority(indices, priorities)

    
    # sample a batch of experiences from the buffer
    def sample_experiences(self):
        print("Sampling experiences...")
        experiences, info = self.prioritized_replay_buffer.sample(self.batch_size, return_info=True)

        # Extracting each component from the tuple
        states, sensor_states, actions, rewards, next_states, next_sensor_states, dones = experiences

        # Check if data is already in tensor form and ensure it's on the correct device
        states = states if isinstance(states, torch.Tensor) else torch.tensor(states, dtype=torch.float32, device=self.device)
        sensor_states = sensor_states if isinstance(sensor_states, torch.Tensor) else torch.tensor(sensor_states, dtype=torch.float32, device=self.device)
        actions = actions if isinstance(actions, torch.Tensor) else torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = rewards if isinstance(rewards, torch.Tensor) else torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = next_states if isinstance(next_states, torch.Tensor) else torch.tensor(next_states, dtype=torch.float32, device=self.device)
        next_sensor_states = next_sensor_states if isinstance(next_sensor_states, torch.Tensor) else torch.tensor(next_sensor_states, dtype=torch.float32, device=self.device)
        dones = dones if isinstance(dones, torch.Tensor) else torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Get importance weights and indices from info
        weights = torch.tensor(info['_weight'], dtype=torch.float32, device=self.device)
        indices = torch.tensor(info['index'], dtype=torch.long, device=self.device)

        return states, sensor_states, actions, rewards, next_states, next_sensor_states, dones, weights, indices

    

    def save_models(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")

    def load_models(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.critic.load_state_dict(torch.load(filename + "_critic"))
