import numpy as np
import torch
from environment import environment
from agent import VehicleAgent
from mapddpg.networks import Actor, Critic
from mapddpg.ou_noise import OUActionNoise
import argparse
import json
import time

if __name__=='__main__':
    """
    Parsing arguments
    """

    ...

    """
    MAPDDPG Training Loop
    """

    print("Initializing variables...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training is processed on {device}")

    # initialize the environment
    env = environment(display_img=False)
    # env.reset()

    # initialize networks and agents
    actor = Actor(num_gru_layers=2).to(device)
    critic = Critic().to(device)
    actor_target = Actor(num_gru_layers=2).to(device)
    critic_target = Critic().to(device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # initialize ornstein-uhlenbeck noise for action exploration
    throttle_brake_noise = OUActionNoise(mean=np.array([0.5]), std_dev=np.array([0.2]))
    steer_noise = OUActionNoise(mean=np.array([0]), std_dev=np.array([0.1]))

    # initialize the agent
    agent = VehicleAgent(
        actor=actor,
        critic=critic,
        actor_target=actor_target,
        critic_target=critic_target,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        throttle_and_brake_noise=throttle_brake_noise,
        steer_noise=steer_noise,
        device=device,
        gamma=0.99,
        tau=0.001,
        batch_size=16,
        buffer_size=10000
    )

    num_episodes = 500
    seq_len = 5 # number of sequence frames to capture for actor GRU
    max_steps = 500

    # lists for plots
    rewards_list = []
    steps_list = []
    collisions_list = []
    lane_deviations_list = []
    episode_lengths_list = []
    episodes_list = []

    # stores current sequences
    current_sequence = []

    # total steps: used for learning every N steps across all episodes
    total_steps = 0
    learn_every_n_steps = 100

    print("All components are initialized.")
    print("Starting training loop...")
    for episode in range(num_episodes):
        print("---------------------------------------------------------------------------------------------------")
        print(f"Episode [{episode}]")
        try:
            # environment and agent variables
            state = env.reset() # reset environment at start of each episode
            time.sleep(3) # delay execution to allow initialization of all env variables
            current_sequence = [state] * (seq_len - 1) # reset sequence for each new episode
            throttle_brake_noise.reset() # reset action noise to mean every episode for better learning
            steer_noise.reset()
            agent.update_counter = 0
            done = False

            # metrics
            total_reward = 0
            steps = 0
            num_collisions = 0
            episode_length_time = 0
            total_lane_deviation = 0
            collision_occurred = False

            while not done:
                # form a sequence
                current_sequence.append(state)
                while len(current_sequence) > seq_len: # there will always be one extra state in the sequence after each step
                    current_sequence.pop(0)
                
                # reordering from (batch, h, w, channels) to (batch, channels, h, w)
                state_sequence = torch.tensor(np.stack(current_sequence, axis=0), dtype=torch.float32, device=device).permute(0, 3, 1, 2).unsqueeze(0)

                # get vehicle sensor info as additional state inputs into actor
                sensor_state = torch.tensor(np.concatenate((env.gps_data, env.imu_data)), dtype=torch.float32, device=device)

                # select and execute action
                action = agent.select_action(state_sequence=state_sequence, vehicle_ego_state=sensor_state) # using actor network
                next_state, reward, done, info = env.step(action)
                
                # collecting metrics
                episode_length_time = info['episode_length']
                total_lane_deviation += info['lane_deviation']
                collision_occurred = info['collision_occurred']

                if steps > max_steps:
                    done = True

                # update current sequence
                current_sequence.append(next_state)
                while len(current_sequence) > seq_len:
                    current_sequence.pop(0)
                
                # while len(current_sequence) < seq_len:
                #     current_sequence.append(next_state)

                # get next vehicle sensor state
                next_sensor_state = torch.tensor(np.concatenate((env.gps_data, env.imu_data)), dtype=torch.float32, device=device)

                next_state_sequence = torch.tensor(np.stack(current_sequence, axis=0), dtype=torch.float32, device=device).permute(0, 3, 1, 2).unsqueeze(0)

                # store transition in the prioritized replay buffer, td-error is the priority
                experience_tuple = (state_sequence, sensor_state, action, reward, next_state_sequence, next_sensor_state, done)
                td_error, _, _ = agent.compute_td_error_and_Q_values(experience_tuple)
                agent.store_experience(experience_tuple, td_error=td_error)

                # learn if there are enough batches
                if len(agent.prioritized_replay_buffer) > agent.batch_size and total_steps % learn_every_n_steps == 0:
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                        print(f"CUDA cache cleared at episode {episode}.")

                    states, sensor_states, actions, rewards, next_states, next_sensor_states, dones, weights, indices = agent.sample_experiences()
                    agent.learn(
                        states.to(device), sensor_states.to(device), actions.to(device), rewards.to(device), 
                        next_states.to(device), next_sensor_states.to(device), 
                        dones.to(device), weights.to(device), indices.to(device)
                    )

                # move to next state
                state = next_state
                # sensor_state = next_sensor_state

                total_reward += reward
                steps += 1
                total_steps += 1

            
            collision_occurred = len(env.collision_history) > 0

            rewards_list.append(total_reward)
            steps_list.append(steps)
            collisions_list.append(collision_occurred)
            lane_deviations_list.append(total_lane_deviation)
            episode_lengths_list.append(episode_length_time)
            episodes_list.append(episode)

        
            print(f"Total Reward: {total_reward}, Steps: {steps}, Collision?: {collision_occurred}, Episode Length: {episode_length_time}s, Total Lane Deviation: {total_lane_deviation}m")
            # print("CUDA Memory Usage Summary:")
            # print(torch.cuda.memory_summary())
            print("---------------------------------------------------------------------------------------------------")

            # if episode % 5 == 0:
            # Clear CUDA cache to free unused memory every episode
            if device == 'cuda':
                torch.cuda.empty_cache()
                print(f"CUDA cache cleared at episode {episode}.")

            # save model every 20 episodes
            if episode % 20 == 0:
                filename_prefix = "trained"
                agent.save_models(filename_prefix)

                print(f"Models saved at episode {episode}.")

                with open('stats.json', 'w') as f:
                    json.dump(
                        {
                            "rewards": rewards_list, "steps": steps, 
                               "collisions": collisions_list, "lane_deviations": lane_deviations_list, 
                               "episodes": episodes_list, "episodes_lengths": episode_lengths_list
                        }, f
                    )

                print(f"Metrics data saved at episode {episode}")


        finally:
            env.destroy_all_actors()
            time.sleep(1)
            


    with open('stats.json', 'w') as f:
        json.dump({"rewards": rewards_list, "steps": steps, "collisions": collisions_list, "lane_deviations": lane_deviations_list, "episodes": episodes_list, "episodes_lengths": episode_lengths_list}, f)
