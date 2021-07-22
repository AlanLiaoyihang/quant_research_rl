"""
rewriting the model based network
"""

#header
import copy
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F

import random
import gym
import cherry as ch
from cherry import envs


#parameters
HIDDEN_SIZE = 32
BATCH_SIZE = 256
MAX_STEPS = 2000
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
        q_in = torch.cat([state,action],dim = -1)
        Q1 = self.fc1(q_in)
        Q1 = self.af1(Q1)
        Q1 = self.fc2(Q1)
        Q1 = self.af2(Q1)
        Q1 = self.q1(Q1)
        return Q1
        #return Q1#,Q2

        
class Mb_A3C(nn.Module):
    def __init__(self,obs_dim,action_dim,max_action,num_models):
        super().__init__()
        """
        here we construct the model based actor critic model
        with the model created the probability distribution, feeding
        forward to the actor and the critic to obtain the next state 
        actions and the expected critics. 

        While training ,the models are updated by the maximum likelihood
        againts the actual probability distributions, the actor and the 
        critic are updated the same as the original ddpg.
        """
        self.actor = Actor(obs_dim,action_dim,max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = 1e-3)

        self.critic_1 = Critic(obs_dim,action_dim)
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2 = Critic(obs_dim,action_dim)
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.critic_opt1 = torch.optim.Adam(self.critic_1.parameters(), lr = 3e-3)
        self.critic_opt2 = torch.optim.Adam(self.critic_2.parameters(), lr = 1e-3)

        self.models = []
        self.model_optimizers = []

        for i in range(num_models):
            #appending models into lists
            model = Model(obs_dim+action_dim,obs_dim)
            model.apply(weights_init)
            optimizer = torch.optim.Adam(model.parameters(),lr = 1e-2)
            self.models.append(model)
            self.model_optimizers.append(optimizer)
    
    def get_action(self,state_input):
        return self.actor(state_input)
    

def get_random_action(state):
    return torch.tensor([[2 * random.random() - 1]])



class Model(nn.Module):
    def __init__(self,obs_dim,action_dim):
        """
        constructing the model approximate the real environment
        """
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, HIDDEN_SIZE)
        self.af1 = nn.Tanh()
        self.fc2 = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
        self.af2 = nn.ReLU()

        self.out = nn.Linear(HIDDEN_SIZE,action_dim)

    def forward(self,observation,action):
        x = torch.cat([observation,action], dim = -1)
        x = self.fc1(x)
        x = self.af1(x)
        x = self.fc2(x)
        x = self.af2(x)
        x = self.out(x)
        #next observation
        return x


if __name__ == "__main__":
    """
    in this section we perform the actual training
    """
    #loading environments
    env = gym.make('Pendulum-v0')
    env.seed(SEED)
    env = envs.Torch(env)

    env = envs.Logger(env)
    env = envs.Runner(env)

    replay = ch.ExperienceReplay()
    
    my_model = Mb_A3C(3,1,1,5)

    for step in range(1,MAX_STEPS+1):
        replay += env.run(my_model.get_action,episodes = 1)
        
        if len(replay) > REPLAY_SIZE:
            replay = replay[-REPLAY_SIZE:]
        
        if len(replay) > BATCH_SIZE*3:
            for epoch in range(10):
                replay_buff = replay.sample(BATCH_SIZE)

                #in this we firstly updating the model seperately
                for i in range(len(my_model.models)):
                    #mle training
                    o_next_pred = my_model.models[i](replay_buff.state().detach(), replay_buff.action().detach())
                    loss = F.mse_loss(o_next_pred,replay_buff.next_state())
                    my_model.model_optimizers[i].zero_grad()
                    loss.backward()
                    my_model.model_optimizers[i].step()

                replay_buff.empty()
                
                replay_buff = replay.sample(BATCH_SIZE)

                with torch.no_grad():
                    actions = my_model.actor(replay_buff.state())

                for i in range(len(my_model.models)):
                    if i == 0:
                        sum_o = my_model.models[i](replay_buff.state(),actions)
                    else:
                        sum_o = sum_o + my_model.models[i](replay_buff.state(),actions)
                
                sum_o = sum_o/len(my_model.models)#sum_o is the average next state predictions

                with torch.no_grad():
                    noise =(torch.randn_like(actions.detach())*NOISE_FACTOR).clamp(-NOISE_CLIP,NOISE_CLIP)
                    next_action = (my_model.actor_target(sum_o.detach()) + noise).clamp(-1,1)

                    target_Q1 = my_model.critic_1_target(sum_o.detach(),next_action.detach())
                    target_Q2 = my_model.critic_2_target(sum_o.detach(),next_action.detach())

                    target_Q = torch.min(target_Q1,target_Q2).view(-1,1)

                    target_Q = replay_buff.reward() + (1-replay_buff.done())*DISCOUNT*target_Q

                temp_Q1 = my_model.critic_1(replay_buff.state(),actions.detach())
                temp_Q2 = my_model.critic_2(replay_buff.state(),actions.detach())

                critic_loss1 = F.mse_loss(temp_Q1,target_Q)
                critic_loss2 = F.mse_loss(temp_Q2,target_Q)

                my_model.critic_opt1.zero_grad()
                critic_loss1.backward()
                my_model.critic_opt1.step()

                
                my_model.critic_opt2.zero_grad()
                critic_loss2.backward()
                my_model.critic_opt2.step()

                #delayed policy updates
                if epoch % UPDATE_FREQ == 0:

                    actor_loss = -my_model.critic_1(replay_buff.state(),my_model.actor(replay_buff.state()))
                    actor_loss = actor_loss.mean()

                    my_model.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    my_model.actor_optimizer.step()
                    
                    #updating target models
                    for param, target_param in zip(my_model.critic_1.parameters(), my_model.critic_1_target.parameters()):
                        target_param.data.copy_((1- TAU) * param.data + TAU * target_param.data)

                    for param, target_param in zip(my_model.critic_2.parameters(), my_model.critic_2_target.parameters()):
                        target_param.data.copy_((1- TAU) * param.data + TAU * target_param.data)
                    
                    for param, target_param in zip(my_model.actor.parameters(), my_model.actor_target.parameters()):
                        target_param.data.copy_((1- TAU) * param.data + TAU * target_param.data)
                replay_buff.empty()
