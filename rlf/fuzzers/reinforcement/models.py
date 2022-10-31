import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(use_cuda)
print(device)
# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.1
MEMORY_CAPACITY = 500
Q_NETWORK_ITERATION = 100

ENV_A_SHAPE = 0

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self, state_dim, action_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 800)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(800,500)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(500,action_dim)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class DQN_ARG():
    """docstring for DQN"""
    def __init__(self, state_dim, action_dim):
        super(DQN_ARG, self).__init__()
        self.eval_net = Net(state_dim, action_dim).to(device) 
        self.target_net = Net(state_dim, action_dim).to(device)
        # self.target_net = copy.deepcopy(self.eval_net)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.learn_step_counter = 0
        self.memory_dict = dict()
        # self.memory_counter = 0
        # self.memory = np.zeros((MEMORY_CAPACITY, self.state_dim * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.episilo = EPISILO

        self.directory = 'model_dqn/'

    def choose_action(self, state, greedy=False):
#         state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device) # get a 1D array
        if np.random.randn() <= EPISILO or greedy:# greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.cpu().numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random policy
            action = np.random.randint(0,self.action_dim)
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        if self.episilo < 0.8:
            self.episilo += (0.8-EPISILO)/1500
        return action

    def store_transition(self, state, action, reward, next_state, done, contract):
        if contract.name not in self.memory_dict:
            self.memory_dict[contract.name] = dict()
            self.memory_dict[contract.name]['memory_counter'] = 0
            self.memory_dict[contract.name]['memory'] = np.zeros((MEMORY_CAPACITY, self.state_dim * 2 + 2 + 1))
        
        transition = np.hstack((state, [action, reward], next_state, done))
        index = self.memory_dict[contract.name]['memory_counter'] % MEMORY_CAPACITY
        self.memory_dict[contract.name]['memory'][index, :] = transition
        self.memory_dict[contract.name]['memory_counter'] += 1


    def learn(self):

        for contract_name in self.memory_dict:
            #update the parameters
            if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
            self.learn_step_counter+=1

            #sample batch from memory
            sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
            batch_memory = self.memory_dict[contract_name]['memory'][sample_index, :]
            batch_state = torch.FloatTensor(batch_memory[:, :self.state_dim]).to(device)
            batch_action = torch.LongTensor(batch_memory[:, self.state_dim:self.state_dim+1].astype(int)).to(device)
            batch_reward = torch.FloatTensor(batch_memory[:, self.state_dim+1:self.state_dim+2]).to(device)
            batch_next_state = torch.FloatTensor(batch_memory[:,-self.state_dim-1:-1]).to(device)
            batch_done = torch.FloatTensor(batch_memory[:,-1:]).to(device)
            #q_eval
            q_eval = self.eval_net(batch_state).gather(1, batch_action)
            q_next = self.target_net(batch_next_state).detach()
            q_target = torch.zeros(BATCH_SIZE,1)
            for i in range(0, BATCH_SIZE):
                if batch_done[i]:
                    q_target[i] = batch_reward[i]
                else:
                    q_target[i] = batch_reward[i] + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)[i]
            loss = self.loss_func(q_eval, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def save(self):
        torch.save(self.eval_net.state_dict(), self.directory + 'eval_arg_net.pth')
        # torch.save(self.optimizer.state_dict(), self.directory + 'optimizer.pth')
        print("====================================")
        print("Model DQN_ARG has been saved...")
        print("====================================")

    def load(self):
        import os
        if 'eval_arg_net.pth' in os.listdir(self.directory):
            self.eval_net.load_state_dict(torch.load(self.directory + 'eval_arg_net.pth'))
            # self.optimizer.load_state_dict(torch.load(self.directory + 'optimizer.pth'))
            print("load model from {}".format(self.directory))
            print("====================================")
            print("model DQN_ARG has been loaded...")
            print("====================================")