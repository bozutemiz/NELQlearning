from agent import RLAgent, RLCoupledAgent
from environment import Environment
from config import config2, agent_config, train_config
from plot import plot_reward
import nel

from collections import deque
import random
from six.moves import cPickle
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import os


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


def compute_td_loss(batch_size, agent, replay_buffer, gamma, optimizer):
    # Sample a random minibatch from the replay history.
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    #print('state:', state)
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    qv_target = Variable(torch.FloatTensor(1, batch_size).zero_())

    # TODO: need to add randomnization for next q-value chosen 
    q_values = agent.Qpolicy(state)
    q_values_target = agent.Qtarget(next_state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = q_values_target.max(1)[0] #introduce randomnization
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    max_q_value = q_values_target.max(1)[0]
    v_values = agent.Vpolicy(state)
    v_values_target = agent.Vtarget(next_state)
    next_v_value = v_values_target[0]
    expected_v_value = reward + gamma * next_v_value * (1 - done)

    # loss = F.smooth_l1_loss(q_value,  Variable(expected_q_value.data))
    Qloss = F.mse_loss(q_value,  Variable(expected_q_value.data))
    Vloss = F.mse_loss(v_values,  Variable(expected_v_value.data))

    difference = max_q_value.sub(v_values.transpose(0, 1))
    QVloss = F.mse_loss(difference,  qv_target)

    totloss = Qloss + Vloss + QVloss

    # print('before Q')
    # cnt = 0
    # for param in agent.Qpolicy.parameters():
    # 	if cnt == 0:
    # 		# print(param.data)
    # 	cnt += 1

    # # print('before V')
    # cnt = 0
    # for param in agent.Vpolicy.parameters():
    # 	if cnt == 0:
    # 		# print(param.data)
    # 	cnt += 1

    # print('cnt:',cnt)	
    optimizer.zero_grad()
    totloss.backward()
    optimizer.step()

    # print('after Q')
    # cnt = 0
    # for param in agent.Qpolicy.parameters():
    # 	if cnt == 0:
    # 		print(param.data)
    # 	cnt += 1

    # # print('after V')
    # cnt = 0
    # for param in agent.Vpolicy.parameters():
    # 	if cnt == 0:
    # 		print(param.data)
    # 	cnt += 1

    #Voptimizer.zero_grad()
    #totVloss.backward()
    #Voptimizer.step()

    return Qloss, Vloss, QVloss, totloss
    #return totloss


def plot_setup():
    # plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(151)
    ax2 = fig.add_subplot(152)
    ax3 = fig.add_subplot(153)
    ax4 = fig.add_subplot(154)
    ax5 = fig.add_subplot(155)
    p1, = ax1.plot([])
    p2, = ax2.plot([])
    p3, = ax3.plot([])
    p4, = ax4.plot([])
    p5, = ax5.plot([])
    ax2.set_title('Q loss')
    ax3.set_title('V loss')
    ax4.set_title('QV loss')
    ax5.set_title('Total loss')
    print("SETUP")
    fig.canvas.draw()

    def update(frame_idx, rewards, Qlosses, Vlosses, QVlosses, totlosses):
    #def update(frame_idx, rewards, losses):
        p1.set_xdata(range(len(rewards)))
        p1.set_ydata(rewards)

        ax1.set_title('frame %s. reward: %s' %
                      (frame_idx, np.mean([rewards[i] for i in range(-10, 0)])))
        p2.set_xdata(range(len(Qlosses)))
        p2.set_ydata(Qlosses)
        p3.set_xdata(range(len(Vlosses)))
        p3.set_ydata(Vlosses)
        p4.set_xdata(range(len(QVlosses)))
        p4.set_ydata(QVlosses)
        p5.set_xdata(range(len(totlosses)))
        p5.set_ydata(totlosses)
        ax1.set_xlim([0, len(rewards)])
        ax1.set_ylim([min(rewards), max(rewards) + 10])
        ax2.set_xlim([0, len(Qlosses)])
        ax2.set_ylim([min(Qlosses), max(Qlosses)])
        ax3.set_xlim([0, len(Vlosses)])
        ax3.set_ylim([min(Vlosses), max(Vlosses)])
        ax4.set_xlim([0, len(QVlosses)])
        ax4.set_ylim([min(QVlosses), max(QVlosses)])
        ax5.set_xlim([0, len(totlosses)])
        ax5.set_ylim([min(totlosses), max(totlosses)])
        # print(max(QVlosses))
        ax2.set_yscale('log')
        ax3.set_yscale('log')
        ax4.set_yscale('log')
        ax5.set_yscale('log')
        plt.draw()
        plt.pause(0.0001)

    def save(fname):
        fig.savefig(fname)

    return update, save


# def plot(frame_idx, rewards, Qlosses, Vlosses, QVlosses):
def plot(frame_idx, rewards, losses):
    # clear_output(True)
    fig = plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.title('frame %s. reward: %s' %
              (frame_idx, np.mean([rewards[i] for i in range(-10, 0)])))
    plt.plot(rewards)
    plt.subplot(122)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


def get_epsilon(i, EPS_START, EPS_END, EPS_DECAY_START, EPS_DECAY_END):
    if i < EPS_DECAY_START:
        epsilon = EPS_START
    elif i > EPS_DECAY_END:
        epsilon = EPS_END
    else:
        epsilon = EPS_START - (EPS_START - EPS_END) * (i -
                                                       EPS_DECAY_START) / (EPS_DECAY_END - EPS_DECAY_START)
    return epsilon


def save_training_run(Qlosses, Vlosses, QVlosses, totlosses, rewards, agent, save_fn, model_path, plot_path):
#def save_training_run(losses, rewards, agent, save_fn, model_path, plot_path):
    with open('outputs/train_stats.pkl', 'wb') as f:
        #cPickle.dump((losses, rewards), f)
        cPickle.dump((Qlosses, Vlosses, QVlosses, totlosses, rewards), f)

    agent.save(filepath=model_path)

    save_fn(plot_path)


def train(agent, env, actions, optimizer):
    EPS_START = 1.
    EPS_END = .1
    EPS_DECAY_START = 1000.
    EPS_DECAY_END = 50000.

    def eps_func(i):
        return get_epsilon(i, EPS_START, EPS_END, EPS_DECAY_START, EPS_DECAY_END)
    num_steps_save_training_run = train_config['num_steps_save_training_run']
    policy_update_frequency = train_config['policy_update_frequency']
    target_update_frequency = train_config['target_update_frequency']
    eval_frequency = train_config['eval_frequency']
    batch_size = train_config['batch_size']
    training_steps = 0
    replay = ReplayBuffer(train_config['replay_buffer_capacity'])
    discount_factor = train_config['discount_factor']
    eval_reward = []
    eval_steps = train_config['eval_steps']
    max_steps = train_config['max_steps']
    tr_reward = 0
    
    agent.update_target()
    Qlosses = []
    Vlosses = []
    QVlosses = []
    Totlosses = []

    all_rewards = deque(maxlen=100)
    rewards = []
    plt_fn, save_fn = plot_setup()
    painter = None
    #painter_tr = nel.MapVisualizer(env.simulator, config2, (-30, -30), (150, 150))
    prev_Qweights = agent.Qpolicy.fc3.weight
    prev_Vweights = agent.Vpolicy.fc3.weight
    for training_steps in range(max_steps):
        # Update current exploration parameter epsilon, which is discounted
        # with time.
        epsilon = eps_func(training_steps)

        add_to_replay = len(agent.prev_states) >= 1

        # Get current state.
        s1 = agent.get_state()

        # Make a step.
        action, reward = agent.step(epsilon)

        # Update state according to step.
        s2 = agent.get_state()

        # Accumulate all rewards.
        tr_reward += reward
        all_rewards.append(reward)
        rewards.append(np.sum(all_rewards))

        # Add to memory current state, action it took, reward and new state.
        if add_to_replay:
            # enum issue in server machine
            replay.push(s1, action.value, reward, s2, False)

        # Update the network parameter every update_frequency steps.
        if training_steps % policy_update_frequency == 0:
            if batch_size < len(replay):
                # Compute loss and update parameters.
                Qloss, Vloss, QVloss, totloss = compute_td_loss(
                #totloss = compute_td_loss(
                    batch_size, agent, replay, discount_factor, optimizer)
                #Vloss = compute_td_Vloss(
                #    batch_size, agent, replay, discount_factor, Voptimizer)
                Qlosses.append(Qloss.data[0])
                Vlosses.append(Vloss.data[0])
                QVlosses.append(QVloss.data[0])
                Totlosses.append(totloss.data[0])

        if training_steps % 200 == 0 and training_steps > 0:
            print('step = ', training_steps)
            print("Qloss = ", Qloss.data[0])
            print("Vloss = ", Vloss.data[0])
            print("QVloss = ", QVloss.data[0])
            print("totloss = ", totloss.data[0])
            print("train reward = ", tr_reward)
            print('')
            if training_steps < 100000:
                #plt_fn(training_steps, rewards, Totlosses)
                if training_steps % 2000 == 0 and training_steps > 0:
                	plt_fn(training_steps, rewards, Qlosses, Vlosses, QVlosses, Totlosses)
            elif training_steps % 50000 == 0:
                plt_fn(training_steps, rewards, Qlosses, Vlosses, QVlosses, Totlosses)
                #plt_fn(training_steps, rewards, Totlosses)


        if training_steps % target_update_frequency == 0:
            agent.update_target()

        model_path = 'outputs/models/NELQ_' + str(training_steps)
        p_path = 'outputs/plots/NELQ_plot_' + str(training_steps) + '.png'

        if training_steps % num_steps_save_training_run == 0:
            save_training_run(Qlosses, Vlosses, QVlosses, Totlosses, rewards, agent, save_fn, model_path, p_path)
            #save_training_run(Totlosses, rewards, agent, save_fn, model_path, p_path)

    position = agent.position()
    painter = nel.MapVisualizer(env.simulator, config2, (
        position[0] - 70, position[1] - 70), (position[0] + 70, position[1] + 70))
    for _ in range(100):
        s1 = agent.get_state()
        action, reward = agent.step()
        painter.draw()

    with open('outputs/eval_reward.pkl', 'w') as f:
        cPickle.dump(eval_reward, f)

    save_training_run(Qlosses, Vlosses, QVlosses, Totlosses, rewards, agent, save_fn, model_path, p_path)
    #save_training_run(Totlosses, rewards, agent, save_fn, model_path, p_path)
    # print(eval_reward)


# cumulative reward for training and test

def setup_output_dir():
    m_dir = 'outputs/models'
    p_dir = 'outputs/plots'

    if not os.path.exists(m_dir):
        os.makedirs(m_dir)
    if not os.path.exists(p_dir):
        os.makedirs(p_dir)

def main():
    env = Environment(config2)
    from agent import actions
    state_size = (config2.vision_range*2 + 1)**2 * config2.color_num_dims + config2.scent_num_dims + len(actions)
    #state_size = (config2.vision_range*2 + 1)**2 * config2.color_num_dims + config2.scent_num_dims
    agent = RLCoupledAgent(env, state_size=state_size)

    # TODO: need to initialize the weights correctly
    Optimizer = optim.Adam(list(agent.Qpolicy.parameters()) + list(agent.Vpolicy.parameters()),lr=agent_config['learning_rate'])
    #Qoptimizer = optim.Adam(agent.Qpolicy.parameters(),
    #    lr=agent_config['learning_rate'])
    #optim.Adam(agent.Vpolicy.parameters(),
    #    lr=agent_config['learning_rate'])

    setup_output_dir()
    train(agent, env, [0, 1, 2, 3], Optimizer)


if __name__ == '__main__':
    main()
