"""
implementations of the model_assisted bootstrapping ddpg
"""

#header
import copy
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F

import random
import numpy as np
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
NUMS = 3


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
        self.af1 = nn.ReLU() 
        self.fc2 = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
        self.af2 = nn.ReLU() 
        self.q1 = nn.Linear(HIDDEN_SIZE,1)

    def forward(self,state,action):
        q_in = torch.cat([state,action],dim = -1)
        Q1 = self.fc1(q_in)
        Q1 = self.af1(Q1)
        Q1 = self.fc2(Q1)
        Q1 = self.af2(Q1)
        Q1 = self.q1(Q1)
        return Q1

class Model(nn.Module):
    def __init__(self,obs_dim,action_dim):
        """
        constructing the model approximate the real environment
        """
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, HIDDEN_SIZE)
        self.af1 = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
        self.af2 = nn.ReLU()

        self.out = nn.Linear(HIDDEN_SIZE,obs_dim)

    def forward(self,observation,action):
        x = torch.cat([observation,action], dim = -1)
        x = self.fc1(x)
        x = self.af1(x)
        x = self.fc2(x)
        x = self.af2(x)
        x = self.out(x)

        return x

class MBDDPG(nn.Module):
    def __init__(self,obs_dim,action_dim,max_action,num_critics,num_models):
        #structure of the mbddpg
        super().__init__()
        
        #constructing actor
        self.actor = Actor(obs_dim,action_dim,max_action)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = 3e-4)
        self.target_actor = copy.deepcopy(self.actor)

        #constructing critic
        self.critics = []
        self.critics_optimizers = []
        for i in range(num_critics):
            critic = Critic(obs_dim,action_dim)
            opt = torch.optim.Adam(critic.parameters(),lr = 1e-3)
            critic.apply(weights_init)
            self.critics.append(critic)
            self.critics_optimizers.append(opt)
        
        self.target_critics = copy.deepcopy(self.critics)

        #second list of critic
        self.critics_2 = []
        self.critics_optimizers2 = []
        for i in range(num_critics):
            critic = Critic(obs_dim,action_dim)
            opt = torch.optim.Adam(critic.parameters(),lr = 3e-3)
            critic.apply(weights_init)
            self.critics_2.append(critic)
            self.critics_optimizers2.append(opt)
        
        self.target_critics2 = copy.deepcopy(self.critics_2)

        self.models = []
        self.models_optimizers = []
        #constructing the transition models
        for i in range(num_models):
            #appending models into lists
            model = Model(obs_dim,action_dim)
            model.apply(weights_init)
            optimizer = torch.optim.Adam(model.parameters(),lr = 1e-2)
            self.models.append(model)
            self.models_optimizers.append(optimizer)
    
    def get_action(self,state_input):
        return self.actor(state_input)

def rollout(replay_buff, model, index):
    """
    this is the function that produce single stage rollouts, to
    generate a set of the imaginary state transitions
    """
    resultant_replay = ch.ExperienceReplay()

    state = replay_buff.state()
    reward = replay_buff.reward()
    action = model.actor(state.detach())
    done = replay_buff.done()

    index = np.random.randint(len(my_model.models))
    pred_state = model.models[index](state,action)
    
    resultant_replay.append(state = state,reward = reward, action = action, done = done, next_state = pred_state)
    return resultant_replay




if __name__ == "__main__":
    #loading environments
    env = gym.make('Pendulum-v0')
    env.seed(SEED)
    env = envs.Torch(env)

    env = envs.Logger(env)
    env = envs.Runner(env)

    replay = ch.ExperienceReplay()

    imagine_replay = ch.ExperienceReplay()

    my_model = MBDDPG(3,1,1,NUMS,NUMS)

    #variance representations
    var_max = 0
    var_ratio = 0

    for step in range(1,MAX_STEPS+1):
        #should firstly generating the density(mask to calculate for the network)
        temp_replay = env.run(my_model.get_action,episodes = 1)

        #unsure about the variance calculations
        head = np.random.randint(NUMS)
        var_e = torch.var(my_model.critics[head](temp_replay.state(),temp_replay.action()))

        if var_e > var_max:
            var_max = var_e

        var_ratio = var_e/var_max

        replay += temp_replay
        if len(replay)> REPLAY_SIZE:
            replay = replay[-REPLAY_SIZE:]

        if len(replay) > BATCH_SIZE*10:
            for epoch in range(20):
                replay_buff = replay.sample(BATCH_SIZE)

                #training the model
                #should considering updated by the combine loss
                for i in range(len(my_model.models)):
                    o_next_pred = my_model.models[i](replay_buff.state().detach(), replay_buff.action().detach())
                    loss = F.mse_loss(o_next_pred,replay_buff.next_state())
                    my_model.models_optimizers[i].zero_grad()
                    loss.backward()
                    my_model.models_optimizers[i].step()
            
            head = np.random.randint(NUMS)
            
            #training the actor and critic based on the real experience
            #assuming the model equally weighted
            for epoch in range(5):
                replay_buff = replay.sample(BATCH_SIZE)
                for i in range(len(my_model.critics)):
                    with torch.no_grad():                
                        next_action = (my_model.target_actor(replay_buff.next_state()))

                        target_Q = my_model.target_critics[i](replay_buff.next_state(),next_action.detach())

                        target_Q = replay_buff.reward() + (1-replay_buff.done())*DISCOUNT*target_Q

                    temp_Q = my_model.critics[i](replay_buff.state().detach(),replay_buff.action().detach())

                    
                    loss = F.mse_loss(temp_Q,target_Q)

                    my_model.critics_optimizers[i].zero_grad()
                    loss.backward()
                    my_model.critics_optimizers[i].step()

                # Update policy by one step of gradient ascent
                actor_loss = -my_model.critics[head](replay_buff.state(),my_model.actor(replay_buff.state()))
                actor_loss = actor_loss.mean()
                #updating target models
                for i in range(len(my_model.critics)):
                    for param, target_param in zip(my_model.critics[i].parameters(), my_model.target_critics[i].parameters()):
                        target_param.data.copy_((1- TAU) * param.data + TAU * target_param.data)

                
                for param, target_param in zip(my_model.actor.parameters(), my_model.target_actor.parameters()):
                    target_param.data.copy_((1- TAU) * param.data + TAU * target_param.data)

                replay_buff.empty()
            
            #training the actor and critic based on the imaginery experience
            #assuming the model equally weighted
            nums = 0
            for epoch in range(95):
                indicator = np.random.rand()
                if indicator < var_ratio:
                    #print(var_ratio)
                    nums += 1
                    replay_buff = replay.sample(BATCH_SIZE)
                    replay_buff = rollout(replay_buff,my_model,head)
                    for i in range(len(my_model.critics)):
                        with torch.no_grad():
                            actions = my_model.actor(replay_buff.state())
                            
                            noise =(torch.randn_like(actions.detach())*NOISE_FACTOR).clamp(-NOISE_CLIP,NOISE_CLIP)
                            next_action = (my_model.target_actor(replay_buff.next_state()) + noise).clamp(-1,1)

                            target_Q = my_model.target_critics[i](replay_buff.next_state(),next_action.detach())

                            target_Q = replay_buff.reward() + (1-replay_buff.done())*DISCOUNT*target_Q

                        temp_Q = my_model.critics[i](replay_buff.state().detach(),replay_buff.action().detach())

                        
                        loss = F.mse_loss(temp_Q,target_Q)

                        my_model.critics_optimizers[i].zero_grad()
                        loss.backward()
                        my_model.critics_optimizers[i].step()
                    
                    actor_loss = -my_model.critics[head](replay_buff.state(),my_model.actor(replay_buff.state()))
                    actor_loss = actor_loss.mean()

                    my_model.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    my_model.actor_optimizer.step()
                    
                    #updating target models
                    for i in range(len(my_model.critics)):
                        for param, target_param in zip(my_model.critics[i].parameters(), my_model.target_critics[i].parameters()):
                            target_param.data.copy_((1- TAU) * param.data + TAU * target_param.data)

                    
                    for param, target_param in zip(my_model.actor.parameters(), my_model.target_actor.parameters()):
                        target_param.data.copy_((1- TAU) * param.data + TAU * target_param.data)

                    replay_buff.empty()
