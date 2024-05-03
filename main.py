import numpy as np
import datetime
import torch
from environment import environment
from agent import VehicleAgent
from mapddpg.networks import Actor, Critic
import argparse

if __name__=='__main__':
    """
    Parsing arguments
    """

    # initialize the environment
    env = environment()
    state = env.reset()

    # initialize networks and agents
    actor = Actor(num_gru_layers=2)
    critic = Critic()
    actor_target = Actor(num_gru_layers=2)
    critic_target = Critic()

