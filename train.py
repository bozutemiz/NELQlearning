from agent import RLAgent
from environment import Environment
from config import config1

from collections import deque
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

import numpy as np

def train(agent, env, actions, optimizer):
  EPS_START = 1.
  EPS_END = .1
  EPS_DECAY_START=1000.
  EPS_DECAY_END=50000.

  update_frequency = 4
  target_update_frequency = 1000
  batch_size = 16
  training_steps = 0
  epsilon = 1.
  replay = deque(maxlen=10000)
  discount_factor = .9


  while(True):
    if training_steps < EPS_DECAY_START:
      epsilon = EPS_START
    elif training_steps > EPS_DECAY_END:
      epsilon = EPS_END
    else:
      epsilon = EPS_START - (EPS_START - EPS_END) * (training_steps - EPS_DECAY_START) / (EPS_DECAY_END - EPS_DECAY_START)

    s1 = agent.get_state()
    print agent.prev_states
    print s1.shape
    action, reward = env.step(agent)
    s2 = agent.get_state()
    print s2.shape
    replay.append((s1, action, reward, s2))
    if training_steps % update_frequency == 0:
      if batch_size < len(replay):
        sample = random.sample(replay, batch_size)
        s1, action, reward, s2 = zip(*sample)
        s1 = np.array(s1)
        reward = np.array(reward)
        s2 = np.array(s2)
        action = np.array(action)
        s1 = Variable(torch.from_numpy(s1).float())
        q1 = agent.policy(s1)
        q1 = q1[torch.arange(0, action.size).long(), torch.LongTensor(action)]

        s2 = Variable(torch.from_numpy(s2).float())
        q2 = agent.target(s2).data
        q2, _ = torch.max(q2, 1)

        reward = torch.FloatTensor(reward)
        y = Variable(reward + (discount_factor * q2))

        huber = nn.SmoothL1Loss()
        loss = huber(q1, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if training_steps % target_update_frequency == 0:
          agent.target.load_state_dict(agent.policy.state_dict())
          print(loss)

    training_steps += 1

def main():
  env = Environment(config1)
  agent = RLAgent(env)
  optimizer = optim.SGD(agent.policy.parameters(), lr=.1)
  train(agent, env, [0,1,2,3], optimizer)

if __name__ == '__main__':
  main()
