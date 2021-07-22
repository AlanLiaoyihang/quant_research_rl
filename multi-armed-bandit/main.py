"""
<<<<<<< HEAD:multi-armed-bandit/main.py
Multi-armed Bandit examples
=======
Multi-armed Bandit example
>>>>>>> 707ad261214ef5fecd7104ead5af2f240daccd3d:multi-armed-bandit/main.py
"""

from environment import Environment
from bandits import GaussianBandit
from agent import Agent, GradientAgent
from policy import (EpsilonGreedyPolicy, GreedyPolicy, UCBPolicy)


class EpsilonGreedyExample:
    label = '2.2 - Action-Value Methods'
    bandit = GaussianBandit(10)
    agents = [
        Agent(bandit, GreedyPolicy()),
        Agent(bandit, EpsilonGreedyPolicy(0.01)),
        Agent(bandit, EpsilonGreedyPolicy(0.1)),
    ]


class OptimisticInitialValueExample:
    label = '2.5 - Optimistic Initial Values'
    bandit = GaussianBandit(10)
    agents = [
        Agent(bandit, EpsilonGreedyPolicy(0.1)),
        Agent(bandit, GreedyPolicy(), prior=5)
    ]


class UCBExample:
    label = '2.6 - Upper-Confidence-Bound Action Selection'
    bandit = GaussianBandit(10)
    agents = [
        Agent(bandit, EpsilonGreedyPolicy(0.1)),
        Agent(bandit, UCBPolicy(1))
    ]


if __name__ == '__main__':
    experiments = 500
    trials = 1000

    #example = EpsilonGreedyExample
    #example = OptimisticInitialValueExample
    example = UCBExample

    env = Environment(example.bandit, example.agents, example.label)
    scores, optimal = env.run(trials, experiments)
    env.plot_results(scores, optimal)
