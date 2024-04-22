import torch
import torch.autograd
import torch.optim as optim
import torch.nn.functional as F
from networks import *
from torchrl.data import ListStorage, PrioritizedReplayBuffer
from networks import Actor, Critic
from ou_noise import OU_noise

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
                 buffer_beta=0.4) -> None:
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # actor-critic components
        self.actor = actor
        self.critic = critic
        self.actor_target = actor_target
        self.critic_target = critic_target
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
            actions = self.actor.forward(state_sequence, vehicle_ego_state)
        
        self.actor.train() # sets actor to training mode
        actions += self.action_noise.sample() # add noise to actions to encourage exploration

        # ensure action is within bounds
        actions[0] = np.clip(actions[0], 0, 1) # throttle
        actions[1] = np.clip(actions[1], -1, 1) # steer
        actions[2] = np.clip(actions[2], 0, 1) # brake

    # learn from a previously sampled batch of experiences
    def learn(self, states, vehicle_state, actions, rewards, next_states, next_vehicle_state, dones):
        # compute target actions using target actor network
        target_actions = self.actor_target.forward(next_states, next_vehicle_state) 

        # compute target Q values using the target critic network
        target_Q_values = self.critic_target.forward(next_states, next_vehicle_state, target_actions)

        # compute expected Q values
        expected_Q_values = rewards + (self.gamma * target_Q_values * (1 - dones))

        # compute current Q value using critic network
        current_Q_values = self.critic(states, actions)

        # compute critic loss and update critic network
        critic_loss = F.mse_loss(current_Q_values, expected_Q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # compute actor loss and update actor network
        # actor loss is usually the negative mean of the current Q values produced by the critic for the current policy's actions
        current_actions = self.actor.forward(states, vehicle_state)
        actor_loss = -self.critic.forward(states, vehicle_state, current_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft-update the target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    
    # soft-update method for copying parameters of actor-critic to target actor-critic
    def soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 * self.tau) + param.data * self.tau)

    
    # store experience in replay buffer
    def store_experience(self, experience_tuple, td_error):
        # # convert to tensors
        # state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        # action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        # reward = torch.FloatTensor([reward]).unsqueeze(0).to(self.device)
        # done = torch.FloatTensor([done]).unsqueeze(0).to(self.device)

        # # compute current Q vals from critic
        # current_Q_values = self.critic

        # convert priority to a scalar
        priority = (td_error.abs() + self.buffer_eps).pow(self.buffer_alpha).item()

        # store the experience with priority
        self.prioritized_replay_buffer.add(experience_tuple, priority)


    # update priorities in the replay buffer after learning
    def update_priorities(self, indices, priorities):
        self.prioritized_replay_buffer.update_priority(indices, priorities)

    
    # sample a batch of experiences from the buffer
    def sample_experiences(self):
        states, vehicle_state, actions, rewards, next_states, next_vehicle_state, dones, info = self.prioritized_replay_buffer.sample(self.batch_size, return_info=True)

        # get weights and indices for the sampled experiences
        weights = torch.tensor(info['_weight'], dtype=torch.float32)
        indices = info['index']

        return states, vehicle_state, actions, rewards, next_states, next_vehicle_state, dones, weights, indices
    

    def save_models(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic.state_dict(), filename + "_critic")

    def load_models(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.critic.load_state_dict(torch.load(filename + "_critic"))
