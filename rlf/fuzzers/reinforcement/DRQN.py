import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import random
from collections import deque

use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(use_cuda)
print(device)
# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.2
MEMORY_CAPACITY = 500
Q_NETWORK_ITERATION = 1

ENV_A_SHAPE = 0

class ReplayMemory(object):
    def __init__(self, max_epi_num=50, max_epi_len=51):
        # capacity is the maximum number of episodes
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.memory = deque(maxlen=self.max_epi_num)
        self.is_av = False
        self.current_epi = 0
        self.memory.append([])

    def reset(self):
        self.current_epi = 0
        self.memory.clear()
        self.memory.append([])

    def create_new_epi(self):
        self.memory.append([])
        self.current_epi = self.current_epi + 1
        if self.current_epi > self.max_epi_num - 1:
            self.current_epi = self.max_epi_num - 1

    def remember(self, state, action, reward, *arg):
        if len(self.memory[self.current_epi]) < self.max_epi_len:
            self.memory[self.current_epi].append([state, action, reward, *arg])

    def sample(self):
        epi_index = random.randint(0, len(self.memory)-2)
        if self.is_available():
            return self.memory[epi_index]
        else:
            return []

    def size(self):
        return len(self.memory)

    def is_available(self):
        self.is_av = True
        if len(self.memory) <= 1:
            self.is_av = False
        return self.is_av

    def print_info(self):
        for i in range(len(self.memory)):
            print('epi', i, 'length', len(self.memory[i]))
            print(print([(m[3], m[1],m[2]) for m in self.memory[i]]))


class Net(nn.Module):
    """docstring for Net"""
    def __init__(self, state_dim, action_dim):
        super(Net, self).__init__()
        self.lstm_i_dim = 16    # input dimension of LSTM
        self.lstm_h_dim = 16     # output dimension of LSTM
        self.lstm_N_layer = 1   # number of layers of LSTM
        self.N_action = action_dim
        self.fc1 = nn.Linear(state_dim, 200)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(200,self.lstm_i_dim)
        self.fc2.weight.data.normal_(0,0.1)
        self.lstm = nn.LSTM(input_size=self.lstm_i_dim, hidden_size=self.lstm_h_dim, num_layers=self.lstm_N_layer)
        self.out = nn.Linear(self.lstm_h_dim, action_dim)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x, hidden):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h2 = h2.unsqueeze(1)
        # print(h2)
        h3, new_hidden = self.lstm(h2, hidden)
        action_prob = self.out(h3)
        return action_prob, new_hidden

class DRQN():
    """docstring for DQN"""
    def __init__(self, state_dim, action_dim, max_epi_num=50, max_eps_len=100):
        super(DRQN, self).__init__()
        self.eval_net = Net(state_dim, action_dim).to(device) 
        self.target_net = Net(state_dim, action_dim).to(device)
        # self.target_net = copy.deepcopy(self.eval_net)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_eps_len
        self.gamma = GAMMA
        self.buffer = ReplayMemory(max_epi_num=self.max_epi_num, max_epi_len=self.max_epi_len)
        self.episilo = EPISILO

        self.learn_step_counter = 0
        self.memory_dict = dict()
        # self.memory_counter = 0
        # self.memory = np.zeros((MEMORY_CAPACITY, self.state_dim * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def store_transition(self, state, action, reward, *arg):
        self.buffer.remember(state, action, reward, *arg)

    def choose_action(self, obs, choices, limit_action, hidden=None, episole=0.2, agent_action_count_array=np.zeros(7)):
        obs = torch.unsqueeze(torch.FloatTensor(obs).to(device), 0)
        # print(choices)
        # print(episole)
        if hidden == None:
            hidden = (Variable(torch.zeros(1, 1, self.eval_net.lstm_i_dim).float().to(device)), Variable(torch.zeros(1, 1, self.eval_net.lstm_i_dim).float().to(device)))
        # print(obs.device, hidden[0].device)
        if random.random() > episole:
            # print('agent')
            # print(obs, hidden)
            action_value, new_hidden = self.eval_net.forward(obs, hidden)
            # print('action_value',action_value)
            with torch.no_grad():
                action_value = action_value[0,0].cpu().numpy() - limit_action
                # print('action_value',action_value)
                action = np.argmax(action_value)
            # print('action_value',action_value)
            # input('stop')
            agent_action_count_array[action] += 1
            # print(action)
        else:
            # print('random')
            q, new_hidden = self.eval_net.forward(obs, hidden)
            action = np.random.choice(choices)
        # print(action, new_hidden)
        return action, new_hidden

    def learn(self):
        if self.buffer.is_available():
            # print('learn')
            if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
            self.learn_step_counter+=1

            for _ in range(2):
                memo = self.buffer.sample()
                # print("learn ",len(memo))
                obs_list = []
                action_list = []
                reward_list = []
                for i in range(len(memo)):
                    obs_list.append(memo[i][0])
                    action_list.append(memo[i][1])
                    reward_list.append(memo[i][2])
                obs_list = torch.FloatTensor(np.array(obs_list)).to(device)
                hidden = (Variable(torch.zeros(1, 1, self.eval_net.lstm_i_dim).float().to(device)), Variable(torch.zeros(1, 1, self.eval_net.lstm_i_dim).float().to(device)))
                q_next, _ = self.target_net.forward(obs_list, hidden)
                q_eval, _ = self.eval_net.forward(obs_list,hidden)
                q_target = q_eval.clone()
                for t in range(len(memo) - 1):
                    max_next_q = torch.max(q_next[t+1,0,:]).clone().detach()
                    q_target[t, 0, action_list[t]] = reward_list[t] + self.gamma * max_next_q
                T = len(memo) - 1
                q_target[T,0,action_list[T]] = reward_list[T]
                loss = self.loss_func(q_eval, q_target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save(self, model_path):
        torch.save(self.eval_net.state_dict(), f'{model_path}/eval_net.pth')
        # torch.save(self.optimizer.state_dict(), self.directory + 'optimizer.pth')
        print("====================================")
        print("Model DQN has been saved...")
        print("====================================")

    def load(self, model_path):
        import os
        if 'eval_net.pth' in os.listdir(f'{model_path}'):
            if torch.cuda.is_available():
                print('load from cuda')
                self.eval_net.load_state_dict(torch.load(f'{model_path}/eval_net.pth'))
            else:
                print('load from cpu')
                self.eval_net.load_state_dict(torch.load(f'{model_path}/eval_net.pth', map_location=torch.device('cpu')))
            # self.optimizer.load_state_dict(torch.load(self.directory + 'optimizer.pth'))
            # self.eval_net.eval()
            print("load model from {}".format(f'{model_path}/eval_net.pth'))
            print("====================================")
            print("model DQN has been loaded...")
            print("====================================")
