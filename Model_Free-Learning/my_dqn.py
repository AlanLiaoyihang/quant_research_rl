import torch 
import torch.nn as nn
from torch.nn import functional as F

import random
import copy
import numpy as numpy
import gym
import cherry as ch
import numpy as np
from cherry import envs


#parameters settings
ACTION_DISCRETISATION = 5
DISCOUNT = 0.99
EPSILON = 0.05
HIDDEN_SIZE = 32
LR = 0.001
MAX_STEPS = 100000
BATCH_SIZE = 128
REPLAY_SIZE = 100000
TARGET_UPDATE_INTERVAL = 2500
UPDATE_INTERVAL = 1
UPDATE_START = 10000
SEED = 42


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class DQN(nn.Module):
    def __init__(self,input_size,HIDDEN_SIZE,num_actions):
        super().__init__()
        self.num = num_actions
        #constructing the network
        #first fully connected layers
        self.fc1 = nn.Linear(input_size,HIDDEN_SIZE)
        #activation function
        self.af1 =  nn.Tanh()

        #second fully connected layers
        self.fc2 = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
        self.af2 = nn.Tanh()

        #output
        self.last = nn.Linear(HIDDEN_SIZE,num_actions)

        #egreedy
        self.egreedy = ch.nn.EpsilonGreedy(EPSILON)

    
    def forward(self, state):
        output = self.fc1(state)
        output = self.af1(output)
        output = self.fc2(output)
        output = self.af2(output)
        output = self.last(output)

        action = self.egreedy(output)

        return action,output


def create_target_network(network):
    target_network = copy.deepcopy(network)
    for param in target_network.parameters():
        param.requires_grad = False
    return target_network

def _compute_prob_max(q):
    q_array = np.array(q).T
    score = (q_array[:,:,None,None] >= q_array).astype(int)
    prob = score.sum(axis = 3).prod(axis = 2).sum(axis = 1)
    prob = prob.astype(np.float32)
    return prob/np.sum(prob)

def scale(x, out_range=(-1, 1), axis=None):
    domain = np.min(x, axis), np.max(x, axis)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def convert_discrete_to_continuous_action(action):
    return action.to(dtype=torch.float32) - ACTION_DISCRETISATION// 2



#main function
if __name__ == '__main__':
    #load the environments
    env = gym.make('Pendulum-v0')
    env.seed(SEED)
    env = envs.Torch(env)
    env = envs.ActionLambda(env,convert_discrete_to_continuous_action)
    env = envs.Logger(env)
    env = envs.Runner(env)

    replay = ch.ExperienceReplay()
    #initialise agent
    agent = DQN(3,HIDDEN_SIZE, ACTION_DISCRETISATION)
    
    #target network
    target_agent = create_target_network(agent)

    optimiser = torch.optim.Adam(agent.parameters(),lr=LR)

    def get_random_action(state):
        action = torch.tensor([[random.randint(0, ACTION_DISCRETISATION-1)]])
        return action

    def get_action(state):
        return agent(state)[0]

    for step in range(1,MAX_STEPS+1):
        #print("step : {}".format(step))
        with torch.no_grad():
            if step < UPDATE_START:
                replay += env.run(get_random_action,steps = 1)
            else:
                replay += env.run(get_action,steps = 1)

            replay = replay[-REPLAY_SIZE:]


        if step > UPDATE_START and step%UPDATE_INTERVAL == 0:
            #randomly sample
            batch = random.sample(replay, BATCH_SIZE)
            batch = ch.ExperienceReplay(batch)

            target_values = target_agent(batch.next_state())[1].max(dim=1, keepdim=True)[0]
            target_values = batch.reward() + DISCOUNT * (1 - batch.done()) * target_values

            #updating network
            pred_values = agent(batch.state())[1].gather(1, batch.action())
            value_loss = F.mse_loss(pred_values, target_values)
            optimiser.zero_grad()
            value_loss.backward()
            optimiser.step()


        if step > UPDATE_START and step % TARGET_UPDATE_INTERVAL == 0:
                # Update target network
                target_agent = create_target_network(agent) 


    step_stats = env.logger