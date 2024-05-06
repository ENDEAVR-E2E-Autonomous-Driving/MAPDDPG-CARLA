import numpy as np
import torch
from environment import environment
from agent import VehicleAgent
from mapddpg.networks import Actor, Critic
from mapddpg.ou_noise import OU_noise
import argparse
import json

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # initialize networks and agents
    actor = Actor(num_gru_layers=2).to(device)
    critic = Critic().to(device)
    actor_target = Actor(num_gru_layers=2).to(device)
    critic_target = Critic().to(device)

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
    max_steps = 500

    # lists for plots
    rewards_list = []
    steps_list = []
    collisions_list = []
    lane_deviations_list = []
    max_time_exceeded_list = []
    episodes_list = range(num_episodes)

    # stores current sequences
    current_sequence = []

    for episode in range(num_episodes):
        state = env.reset() # reset environment at start of each episode
        current_sequence = [state] * (seq_len - 1) # reset sequence for each new episode
        total_reward = 0
        done = False
        steps = 0
        num_collisions = 0
        max_time_exceeded = False
        lane_deviated = False
        collision_occurred = False

        while not done:
            current_sequence.append(state)

            # form a sequence
            # state_sequence = np.stack(current_sequence, axis=0)
            state_sequence = torch.tensor(np.stack(current_sequence, axis=0), dtype=torch.float32, device=device)

            # get vehicle sensor info as additional state inputs into actor
            sensor_state = torch.tensor(np.concatenate((env.gps_data, env.imu_data)), dtype=torch.float32, device=device)

            # select and execute action
            action = agent.select_action(state_sequence=state_sequence, vehicle_ego_state=sensor_state) # using actor network
            next_state, reward, done, info = environment.step(action)
            
            max_time_exceeded = info['max_time_exceeded']
            lane_deviated = info['lane_deviated']
            collision_occurred = info['collision_occurred']

            if steps > max_steps:
                done = True

            # update current sequence
            current_sequence.append(next_state)
            if len(current_sequence) > seq_len:
                current_sequence.pop(0)

            # get next vehicle sensor state
            next_sensor_state = torch.tensor(np.concatenate((env.gps_data, env.imu_data)), dtype=torch.float32, device=device)

            next_state_sequence = torch.tensor(np.stack(current_sequence, axis=0), dtype=torch.float32, device=device)

            # store transition in the prioritized replay buffer, td-error is the priority
            experience_tuple = (state_sequence, sensor_state, action, reward, next_state_sequence, next_sensor_state, done)
            td_error, current_Q_vals, expected_Q_vals = agent.compute_td_error_and_Q_values(experience_tuple)
            agent.store_experience(experience_tuple, td_error=td_error)

            # learn if there are enough batches
            if len(agent.prioritized_replay_buffer) > agent.batch_size:
                states, sensor_states, actions, rewards, next_states, next_sensor_states, dones, weights, indices = agent.sample_experiences()
                agent.learn(states, sensor_states, actions, rewards, next_states, next_sensor_states, dones, weights, indices)

            # move to next state
            state = next_state
            # sensor_state = next_sensor_state

            total_reward += reward
            steps += 1

        # save model every 50 episodes
        if episode % 50 == 0:
            filename_prefix = f"{episode}_episodes"
            agent.save_models(filename_prefix)

        collision_occurred = len(env.collision_history) > 0

        rewards_list.append(total_reward)
        steps_list.append(steps)
        collisions_list.append(collision_occurred)

        print(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {steps}, Collision?: {collision_occurred}")


    with open('stats.json', 'w') as f:
        json.dump({})
