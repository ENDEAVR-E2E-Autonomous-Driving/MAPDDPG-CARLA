import torch
import torch.autograd
import torch.optim as optim
import torch.nn.functional as F
from torchrl.data import ListStorage, PrioritizedReplayBuffer
from mapddpg.networks import *
from mapddpg.networks import Actor, Critic
from mapddpg.ou_noise import OU_noise

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
                 action_noise: OU_noise,
                 gamma: float,
                 tau: float,
                 batch_size: int,
                 buffer_size=1000,
                 buffer_alpha=0.6,
                 buffer_eps=1e-4,
                 buffer_beta=0.4,
                 ) -> None:
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # actor-critic components
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.actor_target = actor_target.to(self.device)
        self.critic_target = critic_target.to(self.device)
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.action_noise = action_noise
        
        # replay buffer components
        self.batch_size = batch_size
        self.buffer_eps = buffer_eps
        self.buffer_alpha = buffer_alpha
        self.buffer_beta = buffer_beta
        self.prioritized_replay_buffer = PrioritizedReplayBuffer(alpha=buffer_alpha, beta=buffer_beta, storage=ListStorage(buffer_size))

        # other hyperparameters
        self.gamma = gamma # discount factor
        self.tau = tau # temperature

    # select an action with the actor network
    def select_action(self, state_sequence, vehicle_ego_state):
        self.actor.eval() # sets actor to evaluation mode 

        with torch.no_grad():
            throttle, steer, brake = self.actor.forward(state_sequence, vehicle_ego_state)
        
        if throttle > 0.1:
            brake = 0.0
        elif brake > 0.1:
            throttle = 0.0
        
        actions = [throttle, steer, brake]

        self.actor.train() # sets actor to training mode
        actions += self.action_noise.sample() # add noise to actions to encourage exploration

        # ensure action is within bounds
        actions[0] = np.clip(actions[0], 0, 1) # throttle
        actions[1] = np.clip(actions[1], -1, 1) # steer
        actions[2] = np.clip(actions[2], 0, 1) # brake

        return actions

    # learn from a previously sampled batch of experiences
    def learn(self, states, vehicle_state, actions, rewards, next_states, next_vehicle_state, dones, weights, indices):
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        # compute target Q-values using the Bellman equation
        with torch.no_grad():
            target_actions = self.actor_target(next_states, next_vehicle_state)
            target_Q_values = self.critic_target(next_states, next_vehicle_state, target_actions)
            expected_Q_values = rewards + self.gamma * target_Q_values * (1 - dones)

        # compute current Q-values from the critic network
        current_Q_values = self.critic(states, vehicle_state, actions)

        # critic Loss: mean Squared Error between current Q-values and expected Q-values
        # multiply by importance sampling weights to offset bias from priority sampling
        critic_loss = (weights * F.mse_loss(current_Q_values, expected_Q_values, reduction='none')).mean()
        critic_loss.backward()
        self.critic_optimizer.step()

        # compute new priorities (absolute TD errors + a small epsilon)
        new_priorities = (torch.abs(current_Q_values - expected_Q_values) + self.buffer_eps).detach().numpy()
        self.update_priorities(indices, new_priorities)

        # actor Loss: mean of Q-values output by the critic for current policy's actions
        # negative sign because we want to maximize the critic's output (policy gradient ascent)
        current_actions = self.actor(states, vehicle_state)
        actor_loss = -self.critic(states, vehicle_state, current_actions).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft-update the target networks
        self.soft_update(self.actor, self.actor_target, self.tau)
        self.soft_update(self.critic, self.critic_target, self.tau)

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

        with torch.no_grad():
            next_actions = self.actor_target(next_state_sequence, next_vehicle_state)
            next_Q_values = self.critic_target(next_state_sequence, next_vehicle_state, next_actions)
            expected_Q_values = reward + self.gamma * next_Q_values * (1 - dones)

        current_Q_values = self.critic(state_sequence, sensor_state, action)

        td_error = torch.abs(current_Q_values - expected_Q_values)

        return td_error, current_Q_values, expected_Q_values
    
    # store experience in replay buffer
    def store_experience(self, experience_tuple, td_error):
        # convert priority to a scalar
        priority = (td_error.abs() + self.buffer_eps).pow(self.buffer_alpha).item()

        # store the experience with priority
        self.prioritized_replay_buffer.add(experience_tuple, priority)


    # update priorities in the replay buffer after learning
    def update_priorities(self, indices, priorities):
        self.prioritized_replay_buffer.update_priority(indices, priorities)

    
    # sample a batch of experiences from the buffer
    def sample_experiences(self):
        experiences, info = self.prioritized_replay_buffer.sample(self.batch_size, return_info=True)

        # get importance weights and indices for the sampled experiences
        weights = torch.tensor(info['_weight'], dtype=torch.float32)
        indices = info['index']

        # unpack individual experience components
        states, sensor_states, actions, rewards, next_states, next_sensor_states, dones = zip(*experiences)

        # convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        sensor_state = torch.tensor(np.array(sensor_states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        next_sensor_state = torch.tensor(np.array(next_sensor_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)

        return states, sensor_states, actions, rewards, next_states, next_sensor_states, dones, weights, indices
    

    def save_models(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic.state_dict(), filename + "_critic")

    def load_models(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.critic.load_state_dict(torch.load(filename + "_critic"))
