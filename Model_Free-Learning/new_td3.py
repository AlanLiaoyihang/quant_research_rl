import copy
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F

import random
import gym
import cherry as ch
from cherry import envs

HIDDEN_SIZE = 32
BATCH_SIZE = 256
MAX_STEPS = 1000
REPLAY_SIZE = 10000
NOISE_FACTOR = 0.2
NOISE_CLIP = 0.5
DISCOUNT = 0.95
UPDATE_FREQ = 2
TAU = 0.995

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)

class Actor(nn.Module):
    def __init__(self,input_dim,action_dim,max_action):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, HIDDEN_SIZE)
        self.af1 = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
        self.af2 = nn.ReLU() 
        self.out = nn.Linear(HIDDEN_SIZE,action_dim)

        self.max_action = max_action

    def forward(self,x):
        x = self.fc1(x)
        x = self.af1(x)
        x = self.fc2(x)
        x = self.af2(x)
        state_values = self.out(x)

        return self.max_action * torch.tanh(state_values)

class Critic(nn.Module):
    def __init__(self,input_dim,action_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, HIDDEN_SIZE)
        self.af1 = nn.Tanh() 
        self.fc2 = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
        self.af2 = nn.Tanh() 
        self.q1 = nn.Linear(HIDDEN_SIZE,1)

    def forward(self,state,action):
        q_in = torch.cat([state,action],1)
        Q1 = self.fc1(q_in)
        Q1 = self.af1(Q1)
        Q1 = self.fc2(Q1)
        Q1 = self.af2(Q1)
        Q1 = self.q1(Q1)
        return Q1
        #return Q1#,Q2
    
class TD3(nn.Module):

    def __init__(self,input_dim,action_dim,max_action,discount= 0.95, tau = 0.995, policy_noise = 0.2 ,clip = 0.5, freq = 2):
        super().__init__()

        self.actor = Actor(input_dim, action_dim, max_action)
        
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr= 3e-3)

        self.critic_1 = Critic(input_dim,action_dim)
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2 = Critic(input_dim,action_dim)
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.critic_opt1 = torch.optim.Adam(self.critic_1.parameters(), lr = 3e-3)
        self.critic_opt2 = torch.optim.Adam(self.critic_2.parameters(), lr = 1e-3)

    def get_action(self, state_input):

        return self.actor(state_input)

def get_random_action(state):
    return torch.tensor([[2 * random.random() - 1]])

if __name__ == "__main__":
    #load environment
    env = gym.make('Pendulum-v0')
    env.seed(SEED)
    env = envs.Torch(env)

    env = envs.Logger(env)
    env = envs.Runner(env)

    replay = ch.ExperienceReplay()

    td3_agent = TD3(3,1,1)
    #td3_agent.apply(weights_init)

    #perform the training
    for step in range(1,MAX_STEPS+1):
        replay += env.run(td3_agent.get_action, episodes=1)

        if len(replay) > REPLAY_SIZE:
            replay = replay[-REPLAY_SIZE:]

        if len(replay) > BATCH_SIZE*3:
            for epoch in range(20):
                #consideration of the overestimation, random sampling
                replay_buff = replay.sample(BATCH_SIZE)

                with torch.no_grad():
                    noise = (torch.randn_like(replay_buff.action())*NOISE_FACTOR).clamp(-NOISE_CLIP,NOISE_CLIP)

                    next_action = (td3_agent.actor_target(replay_buff.next_state()) + noise).clamp(-1,1)

                    target_Q1 = td3_agent.critic_1_target(replay_buff.next_state(),next_action.detach())
                    target_Q2 = td3_agent.critic_2_target(replay_buff.next_state(),next_action.detach())

                    target_Q = torch.min(target_Q1,target_Q2).view(-1,1)

                    target_Q = replay_buff.reward() + (1-replay_buff.done())*DISCOUNT*target_Q
                
                temp_Q1 = td3_agent.critic_1(replay_buff.state(),replay_buff.action().detach())
                temp_Q2 = td3_agent.critic_2(replay_buff.state(),replay_buff.action().detach())

                critic_loss1 = F.mse_loss(temp_Q1,target_Q)
                critic_loss2 = F.mse_loss(temp_Q2,target_Q)

                td3_agent.critic_opt1.zero_grad()
                critic_loss1.backward()
                td3_agent.critic_opt1.step()

                
                td3_agent.critic_opt2.zero_grad()
                critic_loss2.backward()
                td3_agent.critic_opt2.step()
                
                #delayed policy updates
                if epoch % UPDATE_FREQ == 0:

                    actor_loss = - td3_agent.critic_1(replay_buff.state(),td3_agent.actor(replay_buff.state()))
                    actor_loss = actor_loss.mean()
                    
                    td3_agent.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    td3_agent.actor_optimizer.step()

                    #updating the target parameters
                    for param, target_param in zip(td3_agent.critic_1.parameters(), td3_agent.critic_1_target.parameters()):
                        target_param.data.copy_((1- TAU) * param.data + TAU * target_param.data)
                    
                    for param, target_param in zip(td3_agent.critic_2.parameters(), td3_agent.critic_2_target.parameters()):
                        target_param.data.copy_((1- TAU) * param.data + TAU * target_param.data)

                    for param, target_param in zip(td3_agent.actor.parameters(), td3_agent.actor_target.parameters()):
                        target_param.data.copy_((1- TAU) * param.data + TAU * target_param.data)
                    
                replay_buff.empty()
