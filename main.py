import numpy as np
import datetime
import torch
from environment import environment
from agent import VehicleAgent
from mapddpg.networks import Actor, Critic
from mapddpg.ou_noise import OU_noise
import argparse

if __name__=='__main__':
    """
    Parsing arguments
    """

    ...

    """
    MAPDDPG Training Loop
    """
    # initialize the environment
    env = environment()
    # state = env.reset()

    # initialize networks and agents
    actor = Actor(num_gru_layers=2)
    critic = Critic()
    actor_target = Actor(num_gru_layers=2)
    critic_target = Critic()

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # initialize ornstein-uhlenbeck noise for action exploration
    action_noise = OU_noise(size=3, seed=np.random.randint(100))

    # initialize the agent
    agent = VehicleAgent(
        actor=actor,
        critic=critic,
        actor_target=actor_target,
        critic_target=critic_target,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        action_noise=action_noise,
        gamma=0.99,
        tau=0.001,
        batch_size=64,
        buffer_size=10000
    )

    num_episodes = 1000
    seq_len = 5 # number of sequence frames to capture for actor GRU

    # stores current sequences
    current_sequence = []

    for episode in range(num_episodes):
        state = env.reset() # reset environment at start of each episode
        current_sequence = [state] * (seq_len - 1) # reset sequence for each new episode
        total_reward = 0
        done = False
        step = 0
        num_collisions = 0

        while not done:
            current_sequence.append(state)

            # form a sequence
            state_sequence = np.stack(current_sequence, axis=0)

            # get vehicle sensor info as additional state inputs into actor
            sensor_state = np.concatenate(env.gps_data, env.imu_data)

            # select and execute action
            action = agent.select_action(state_sequence=state_sequence, vehicle_ego_state=sensor_state) # using actor network
            next_state, reward, done, _ = environment.step(action)

            # store transition in the replay buffer
            agent.store_experience()

            total_reward += reward
            step += 1



