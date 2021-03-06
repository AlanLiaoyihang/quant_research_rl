#!/usr/bin/env python3

import gym
import torch
import random
import numpy as np

from torch import optim
from torch import nn
from torch.distributions import Normal

import cherry as ch
from cherry import envs
from cherry import td
from cherry import pg

DISCOUNT = 0.99
EPSILON = 0.05
HIDDEN_SIZE = 32
LEARNING_RATE = 0.001
MAX_STEPS = 500
BATCH_SIZE = 2048
TRACE_DECAY = 0.97
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class Actor(nn.Module):
    def __init__(self, hidden_size, stochastic=True, layer_norm=False):
        super().__init__()
        layers = [nn.Linear(3, hidden_size),
                  nn.Tanh(),
                  nn.Linear(hidden_size, hidden_size),
                  nn.Tanh(),
                  nn.Linear(hidden_size, 1)]
        if layer_norm:
            layers = (layers[:1]
                      + [nn.LayerNorm(hidden_size)]
                      + layers[1:3]
                      + [nn.LayerNorm(hidden_size)]
                      + layers[3:])
        self.policy = nn.Sequential(*layers)
        if stochastic:
            self.policy_log_std = nn.Parameter(torch.tensor([[0.]]))

    def forward(self, state):
        policy = self.policy(state)
        return policy


class Critic(nn.Module):
    def __init__(self, hidden_size, state_action=False, layer_norm=False):
        super().__init__()
        self.state_action = state_action
        layers = [nn.Linear(3 + (1 if state_action else 0), hidden_size),
                  nn.Tanh(),
                  nn.Linear(hidden_size, hidden_size),
                  nn.Tanh(),
                  nn.Linear(hidden_size, 1)]
        if layer_norm:
            layers = (layers[:1]
                      + [nn.LayerNorm(hidden_size)]
                      + layers[1:3]
                      + [nn.LayerNorm(hidden_size)]
                      + layers[3:])
        self.value = nn.Sequential(*layers)

    def forward(self, state, action=None):
        if self.state_action:
            value = self.value(torch.cat([state, action], dim=1))
        else:
            value = self.value(state)
        return value.squeeze(dim=1)


class ActorCritic(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.actor = Actor(hidden_size, stochastic=True)
        self.critic = Critic(hidden_size)

    def forward(self, state):
        policy = Normal(self.actor(state), self.actor.policy_log_std.exp())
        value = self.critic(state)
        action = policy.sample()
        log_prob = policy.log_prob(action)
        return action, {
                'log_prob': log_prob,
                'value': value,
        }


def main(env='Pendulum-v0'):
    agent = ActorCritic(HIDDEN_SIZE)
    actor_optimiser = optim.Adam(agent.actor.parameters(), lr=LEARNING_RATE)
    critic_optimiser = optim.Adam(agent.critic.parameters(), lr=LEARNING_RATE)

    env = gym.make(env)
    env.seed(SEED)
    env = envs.Torch(env)
    env = envs.Logger(env)
    env = envs.Runner(env)
    replay = ch.ExperienceReplay()

    for step in range(1, MAX_STEPS + 1):

        replay += env.run(agent, episodes=1)
        if len(replay) > BATCH_SIZE:
            with torch.no_grad():
                advantages = pg.generalized_advantage(DISCOUNT,
                                                      TRACE_DECAY,
                                                      replay.reward(),
                                                      replay.done(),
                                                      replay.value(),
                                                      torch.zeros(1))#this is vpg difference to dpg, the advantage calculation
                advantages = ch.normalize(advantages, epsilon=1e-8)#normalising the advantage
                returns = td.discount(DISCOUNT,
                                      replay.reward(),
                                      replay.done())
            
            # Policy loss
            log_probs = replay.log_prob()#log loss of policy
            policy_loss = ch.algorithms.a2c.policy_loss(log_probs, advantages)#loss calculation
            actor_optimiser.zero_grad()
            policy_loss.backward()
            actor_optimiser.step()

            # Value loss
            value_loss = ch.algorithms.a2c.state_value_loss(replay.value(),
                                                            returns)
            critic_optimiser.zero_grad()
            value_loss.backward()
            critic_optimiser.step()
            
            replay.empty()


if __name__ == '__main__':
    main()