from agent import RLAgent, RLCoupledAgent
from environment import Environment
#from config import config2, agent_config, train_config
from config import agent_config, train_config
from plot import plot_reward
import nel

from collections import deque
import random
from six.moves import cPickle
import copy
import math

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


def compute_td_loss(batch_size, agent, replay_buffer, gamma, optimizer, index, is_qv_onpol=True):
    # Sample a random minibatch from the replay history.
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    #print('state:', state)
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    qv_target = Variable(torch.FloatTensor(1, batch_size).zero_())

    q_values = agent.Qpolicy(state)
    q_values_t = agent.Qtarget(state)

    #asddd
    #max_q_value = q_values_t.max(1)[0]


    q_values_target = agent.Qtarget(next_state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = q_values_target.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    v_values = agent.Vpolicy(state)
    v_values_t = agent.Vtarget(state)
    v_values_target = agent.Vtarget(next_state)
    next_v_value = v_values_target[0]
    expected_v_value = reward + gamma * next_v_value * (1 - done)

    #print("v_values: ", v_values[0:2])
    #print("q_values: ", q_values[0:2])

    #asdd
    difference = max_q_value.sub(v_values_t.transpose(0, 1))

    # loss = F.smooth_l1_loss(q_value,  Variable(expected_q_value.data))
    Qloss = F.mse_loss(q_value,  Variable(expected_q_value.data))
    Vloss = F.mse_loss(v_values,  Variable(expected_v_value.data))

    QVloss = Variable(torch.FloatTensor(1).zero_())

    #if index > 1000:
    if index >= 0:

        if is_qv_onpol:
            max_q_value = q_values.max(1)[0]
            difference = max_q_value.sub(v_values.transpose(0, 1))
        else:
            max_q_value = q_values_t.max(1)[0]
            difference = max_q_value.sub(v_values_t.transpose(0, 1))
    	


    	#difference = max_q_value.sub(v_values_t.transpose(0, 1))
    #print("q_max: ", max_q_value[0:2])
    #print("difference: ", difference[0][0:2])
    #	w = 0.1 * math.log(index)

    #	if index % 200 == 0:
    #		print('w :', w)


    	QVloss = F.mse_loss(difference,  qv_target)



    	totloss = Qloss + Vloss + QVloss

    else:
    	totloss = Qloss + Vloss

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
    fig = plt.figure(figsize=(100, 5))
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


def save_training_run(Qlosses, Vlosses, QVlosses, totlosses, rewards, agent, save_fn, output_dir, model_path, plot_path):
#def save_training_run(losses, rewards, agent, save_fn, model_path, plot_path):
    with open(output_dir + 'train_stats.pkl', 'wb') as f:
        #cPickle.dump((losses, rewards), f)
        cPickle.dump((Qlosses, Vlosses, QVlosses, totlosses, rewards), f)

    agent.save(filepath=model_path)

    save_fn(plot_path)


def train(agent, env, actions, optimizer, output_dir, m_dir, p_dir, conf, is_qv_onpol=True):
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
 #   eval_reward = []
    eval_steps = train_config['eval_steps']
    max_steps = train_config['max_steps']
    tr_reward = 0
    
    agent.update_target()
    Qlosses = []
    Vlosses = []
    QVlosses = []
    Totlosses = []

    all_rewards = deque(maxlen=100)
    rewards_100 = []
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
        rewards.append(reward)
        all_rewards.append(reward)
        rewards_100.append(np.sum(all_rewards))

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
                    batch_size, agent, replay, discount_factor, optimizer, training_steps, is_qv_onpol=is_qv_onpol)
                #Vloss = compute_td_Vloss(
                #    batch_size, agent, replay, discount_factor, Voptimizer)
                Qlosses.append(Qloss.data[0])
                Vlosses.append(Vloss.data[0])
                QVlosses.append(QVloss.data[0])
                Totlosses.append(totloss.data[0])

        if training_steps % 10000 == 0 and training_steps > 0:
            print('step = ', training_steps)
            print("Qloss = ", Qloss.data[0])
            print("Vloss = ", Vloss.data[0])
            print("QVloss = ", QVloss.data[0])
            print("totloss = ", totloss.data[0])
            print("train reward = ", tr_reward)
            print('')
            if training_steps < 50000:
                #plt_fn(training_steps, rewards_100, Totlosses)
                if training_steps % 2000 == 0 and training_steps > 0:
                	plt_fn(training_steps, rewards_100, Qlosses, Vlosses, QVlosses, Totlosses)
            elif training_steps % 50000 == 0:
                plt_fn(training_steps, rewards_100, Qlosses, Vlosses, QVlosses, Totlosses)
                #plt_fn(training_steps, rewards_100, Totlosses)


        if training_steps % target_update_frequency == 0:
            agent.update_target()

        #model_path = 'outputs/models/NELQ_' + str(training_steps)
        model_path = m_dir + '/NELQ_' + str(training_steps)
        #p_path = 'outputs/plots/NELQ_plot_' + str(training_steps) + '.png'
        p_path = p_dir + '/NELQ_plot_' + str(training_steps) + '.png'

        if training_steps % num_steps_save_training_run == 0:
            save_training_run(Qlosses, Vlosses, QVlosses, Totlosses, rewards, agent, save_fn, output_dir, model_path, p_path)
            #save_training_run(Totlosses, rewards, agent, save_fn, model_path, p_path)

    position = agent.position()
    painter = nel.MapVisualizer(env.simulator, conf, (
        position[0] - 70, position[1] - 70), (position[0] + 70, position[1] + 70))
    for _ in range(100):
        s1 = agent.get_state()
        action, reward = agent.step()
        painter.draw()

#    with open('outputs/eval_reward.pkl', 'w') as f:
#        cPickle.dump(eval_reward, f)

    save_training_run(Qlosses, Vlosses, QVlosses, Totlosses, rewards, agent, save_fn, output_dir, model_path, p_path)
    #save_training_run(Totlosses, rewards, agent, save_fn, model_path, p_path)
    # print(eval_reward)


# cumulative reward for training and test

def setup_output_dir(m_dir, p_dir):

    if not os.path.exists(m_dir):
        os.makedirs(m_dir)
    if not os.path.exists(p_dir):
        os.makedirs(p_dir)

def main():


    for i in range(10):

        # Initialize the seeds
        random.seed(i)
        np.random.seed(i)
        torch.manual_seed(i)

        # stuff in the config file is moved here and removed from import and config.py
        items = []
        items.append(nel.Item("banana", [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], False))
        items.append(nel.Item("onion", [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], False))
        items.append(nel.Item("jellybean", [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], True))

        # specify the intensity and interaction function parameters
        intensity_fn_args = [-3.3, -3.7, -3.0]
        interaction_fn_args = [len(items)]
        interaction_fn_args.extend([10.0, 100.0, 0.0, -6.0])     # parameters for interaction between item 0 and item 0
        interaction_fn_args.extend([100.0, 0.0, -6.0, -6.0])     # parameters for interaction between item 0 and item 1
        interaction_fn_args.extend([10.0, 100.0, 1.0, -100.0])   # parameters for interaction between item 0 and item 2
        interaction_fn_args.extend([100.0, 0.0, -6.0, -6.0])     # parameters for interaction between item 1 and item 0
        interaction_fn_args.extend([10.0, 0.0, -2.0, 0.0])         # parameters for interaction between item 1 and item 1
        interaction_fn_args.extend([100.0, 0.0, -100.0, -100.0]) # parameters for interaction between item 1 and item 2
        interaction_fn_args.extend([10.0, 100.0, 1.0, -100.0])   # parameters for interaction between item 2 and item 0
        interaction_fn_args.extend([100.0, 0.0, -100.0, -100.0]) # parameters for interaction between item 2 and item 1
        interaction_fn_args.extend([10.0, 100.0, 0.0, -6.0])     # parameters for interaction between item 2 and item 2

        config2 = nel.SimulatorConfig(seed=i,
            max_steps_per_movement=1, vision_range=5,
            patch_size=32, gibbs_num_iter=10, items=items,
            agent_color=[1.0, 0.5, 0.5],
            collision_policy=nel.MovementConflictPolicy.FIRST_COME_FIRST_SERVED,
            decay_param=0.4, diffusion_param=0.14,
            deleted_item_lifetime=2000,
            intensity_fn=nel.IntensityFunction.CONSTANT,
            intensity_fn_args=intensity_fn_args,
            interaction_fn=nel.InteractionFunction.PIECEWISE_BOX,
            interaction_fn_args=interaction_fn_args)

        isqvonpol = False
    	output_dir = 'QVloss_offpol_seed_3/outputs_' + str(i) + '/'
    	m_dir = output_dir + 'models'
    	p_dir = output_dir + 'plots'

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

    	setup_output_dir(m_dir,p_dir)
    	train(agent, env, [0, 1, 2, 3], Optimizer,output_dir,m_dir,p_dir, config2, is_qv_onpol=isqvonpol)


if __name__ == '__main__':
    main()
