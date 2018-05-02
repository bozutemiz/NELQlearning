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


def compute_td_loss(batch_size, agent, replay_buffer, gamma, optimizer, target_update_frequency, training_steps):
    # Sample a random minibatch from the replay history.
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    qv_target = Variable(torch.FloatTensor(1, batch_size).zero_())

    q_values = agent.Qpolicy(state)
    q_values_target = agent.Qtarget(next_state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = q_values_target.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    max_q_value = q_values.max(1)[0]
    v_values = agent.Vpolicy(state)
    v_values_target = agent.Vtarget(next_state)
    next_v_value = v_values_target[0]
    expected_v_value = reward + gamma * next_v_value * (1 - done)

   
    Qloss = F.mse_loss(q_value,  Variable(expected_q_value.data))
    Vloss = F.mse_loss(v_values,  Variable(expected_v_value.data))

    difference = max_q_value.sub(v_values.transpose(0, 1))
    totloss = Qloss + Vloss
    
    QVloss = F.mse_loss(difference,  qv_target)
    if training_steps % target_update_frequency == 0:
        # QVloss = F.mse_loss(difference,  qv_target)
        totloss += QVloss
    
    #totloss = Qloss + Vloss +QVloss

	# backprop total loss
    optimizer.zero_grad()
    totloss.backward()
    optimizer.step()

    return Qloss, Vloss, QVloss, totloss
    # return Qloss, Vloss, QVloss, totloss


def plot_setup():
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
    fig.canvas.draw()

    def update(frame_idx, rewards, Qlosses, Vlosses, QVlosses, totlosses):
    
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

        ax2.set_yscale('log')
        ax3.set_yscale('log')
        ax4.set_yscale('log')
        ax5.set_yscale('log')
        plt.draw()
        plt.pause(0.0001)

    def save(fname):
        fig.savefig(fname)

    return update, save


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


def save_training_run(Qlosses, Vlosses, QVlosses, totlosses, rewards, agent, save_fn, model_path, plot_path, target_update_frequency, n):
    fname = 'outputs/' + str(target_update_frequency) + '_' + str(n) + '_train_stats.pkl'
    with open(fname, 'wb') as f:
        cPickle.dump((Qlosses, Vlosses, QVlosses, totlosses, rewards), f)

    agent.save(filepath=model_path)

    #save_fn(plot_path)


def train(agent, env, actions, optimizer, target_update_frequency, n):
    EPS_START = 1.
    EPS_END = .1
    EPS_DECAY_START = 1000.
    EPS_DECAY_END = 50000.

    def eps_func(i):
        return get_epsilon(i, EPS_START, EPS_END, EPS_DECAY_START, EPS_DECAY_END)
    num_steps_save_training_run = train_config['num_steps_save_training_run']
    policy_update_frequency = train_config['policy_update_frequency']
    #target_update_frequency = #train_config['target_update_frequency']
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
        NoneType = type(None)

        # Add to memory current state, action it took, reward and new state.
        if add_to_replay:
            # enum issue in server machine
            replay.push(s1, action.value, reward, s2, False)

        # Update the network parameter every update_frequency steps.
        if training_steps % policy_update_frequency == 0:
            if batch_size < len(replay):
                # Compute loss and update parameters.
                Qloss, Vloss, QVloss, totloss = compute_td_loss(
                    batch_size, agent, replay, discount_factor, optimizer, 
                    target_update_frequency, training_steps)
                
                Qlosses.append(Qloss.data[0])
                Vlosses.append(Vloss.data[0])
                if not isinstance(QVloss, NoneType):
                    QVlosses.append(QVloss.data[0])
                Totlosses.append(totloss.data[0])

        # Update target with policy network weights
        if training_steps % 1000 == 0:
        # if training_steps % target_update_frequency == 0:
            agent.update_target()

        
        #     plt_fn(training_steps, rewards, Qlosses, Vlosses, QVlosses, Totlosses)
        if training_steps % 5000 == 0 and training_steps > 0:
        #     if training_steps < 100000:
            print("training step:", training_steps)
        #         if training_steps % 2000 == 0 and training_steps > 0:
        #         	plt_fn(training_steps, rewards, Qlosses, Vlosses, QVlosses, Totlosses)
        #     elif training_steps % 50000 == 0:
        #         plt_fn(training_steps, rewards, Qlosses, Vlosses, QVlosses, Totlosses)


        model_path = 'outputs/models/' + str(target_update_frequency) + '_' + str(n) + '_NELQ_' + str(training_steps)
        p_path = 'outputs/plots/'+ str(target_update_frequency) + '_' + str(n) + '_NELQ_plot_' + str(training_steps) + '.png'

        if training_steps % num_steps_save_training_run == 0:
            save_training_run(Qlosses, Vlosses, QVlosses, Totlosses, rewards, agent, save_fn, model_path, p_path, target_update_frequency, n)

    position = agent.position()
    painter = nel.MapVisualizer(env.simulator, config2, (
        position[0] - 70, position[1] - 70), (position[0] + 70, position[1] + 70))
    for _ in range(100):
        s1 = agent.get_state()
        action, reward = agent.step()
        painter.draw()

    fname = 'outputs/' + str(target_update_frequency) + '_' + str(n) + '_eval_reward.pkl'
    with open(fname, 'w') as f:
        cPickle.dump(eval_reward, f)

    p_path = 'outputs/plots/'+ str(target_update_frequency) + '_' + str(n) + '_NELQ_plot_' + str(max_steps) + '.png'
    plt_fn(training_steps, rewards, Qlosses, Vlosses, QVlosses, Totlosses)
    save_training_run(Qlosses, Vlosses, QVlosses, Totlosses, rewards, agent, save_fn, model_path, p_path, target_update_frequency, n)
    save_fn(p_path)

# cumulative reward for training and test
def setup_output_dir(target_update_frequency, n):
    m_dir = 'outputs/models/' + str(target_update_frequency) + '_' + str(n) + "-run for 10,000" 
    p_dir = 'outputs/plots/' + str(target_update_frequency) + '_' + str(n) + "-run for 10,000"

    if not os.path.exists(m_dir):
        os.makedirs(m_dir)
    if not os.path.exists(p_dir):
        os.makedirs(p_dir)

def main(target_update_frequency, n):
    env = Environment(config2)
    from agent import actions
    state_size = (config2.vision_range*2 + 1)**2 * config2.color_num_dims + config2.scent_num_dims + len(actions)
    agent = RLCoupledAgent(env, state_size=state_size)

    # TODO: need to initialize the weights correctly
    Optimizer = optim.Adam(list(agent.Qpolicy.parameters()) + list(agent.Vpolicy.parameters()),lr=agent_config['learning_rate'])

    setup_output_dir(target_update_frequency, n)
    train(agent, env, [0, 1, 2, 3], Optimizer, target_update_frequency, n)


# if __name__ == '__main__':
#     main()
