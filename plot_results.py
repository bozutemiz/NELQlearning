import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from six.moves import cPickle
from collections import deque
import numpy as np

def read_and_calculate_score(f_path, ind, is_rew100=False):

	data = cPickle.load( open( f_path, "rb" ) )
	rews = data[ind]

	if is_rew100:
		rews = get_rewards_per_step(rews)


	print('tot_rew :', np.sum(rews))
	print('ave_rew :', np.mean(rews))
	print('ave_rew_f100k :', np.mean(rews[0:100000]))


	return calculate_mean_rew_per_time(rews)



def calculate_mean_rew_per_time(rewards):
	mean_rews = []
	tot_rews = 0
	for i in range(len(rewards)):
		tot_rews += rewards[i]
		mean_rews.append(tot_rews/(i+1.0))
	return mean_rews


def tot_reward_n_step(rewards, n=100):
	all_rewards = deque(maxlen=n)
	cum_rewards = []

	for i in range(len(rewards)):
		all_rewards.append(rewards[i])
		asd = np.sum(all_rewards)
		cum_rewards.append(asd)

	return cum_rewards

def get_rewards_per_step(data):

	rews = [0]
	for i in range(len(data)):
		if i < 100 and i > 0:
			rews.append(data[i] - data[i-1])
		if i >= 100:
			deleted_val = rews[i-100]
			curr_val = data[i]
			prev_val = data[i-1]
			if deleted_val == 0:
				if curr_val - prev_val == 0:
					rews.append(0)
				else:
					rews.append(1)
			else:
				if curr_val - prev_val == 0:
					rews.append(1)
				else:
					rews.append(0)

	return rews


def main():

	#running_mean_rew_qvl_onpol = []
	running_mean_rew_qvl_onpol_20m = []
	running_mean_rew_qvl_onpol_seed = []
	#running_mean_rew_qvl_onpol_vis10 = []
	#running_mean_rew_qnet = []
	#running_mean_rew_qnet_vis10 = []
	running_mean_rew_qnet_20m = []
	running_mean_rew_qnet_seed = []
	#running_mean_rew_qnet_seed_2 = []
	#running_mean_rew_qvl_offpol = []
	running_mean_rew_qvl_offpol_seed = []

	#running_mean_rew_qvl_onpol_aft1k_w1 = []
	running_mean_rew_qvl_onpol_aft1k_w1_seed = []
	#running_mean_rew_qvl_onpol_aft10k_w1 = []
	running_mean_rew_qvl_onpol_aft10k_w1_seed = []
	#running_mean_rew_qvl_onpol_aft100k_w1 = []
	running_mean_rew_qvl_onpol_aft100k_w1_seed = []

	running_mean_rew_qnet_eps25k = []
	running_mean_rew_qnet_eps100k = []

	running_mean_rew_qnet_exprep5k = []

	
	running_mean_n100 = []
	running_mean_n1k = []
	running_mean_n10k = []

	running_mean_n100_seed = []
	running_mean_n1k_seed = []
	running_mean_n10k_seed = []


	running_mean_ln1 = []
	running_mean_ln2 = []

	file_dir1 = "Results Seed/"
	file_name = "train_stats.pkl"

	#for j in range(5):
	
	#	file_dir2 = "baseline_exprep_5k/"
	#	file_dir3 = "outputs_" + str(j)# + "_5m"
	
	#	file_dir = file_dir2 + file_dir3 + "/"
	#	ind = 1
		
	#	running_mean_rew_qnet_exprep5k.append(read_and_calculate_score(file_dir+file_name, ind))
	

	#for j in range(10):
	#	file_dir2 = "QVLoss_onpol/"
	#	file_dir3 = "outputs_" + str(j)
		
	#	file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
	#	ind = 4
		
	#	running_mean_rew_qvl_onpol.append(read_and_calculate_score(file_dir+file_name, ind)[0:500001])

	for j in range(10):
		file_dir2 = "QVloss_onpol_seed/"
		file_dir3 = "outputs_" + str(j)
		
		#file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
		file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
		ind = 4
		
		running_mean_rew_qvl_onpol_seed.append(read_and_calculate_score(file_dir+file_name, ind))

	for j in range(1):
		file_dir2 = "QVloss_onpol_20m/"
		file_dir3 = "outputs_" + str(j)
		
		#file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
		file_dir = file_dir2 + file_dir3 + "/"
		ind = 4
		
		running_mean_rew_qvl_onpol_20m.append(read_and_calculate_score(file_dir+file_name, ind)[0:10000001])


	#for j in range(10):

	#	file_dir2 = "QVLoss_offpol/"
	#	file_dir3 = "outputs_" + str(j)
		
	#	file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
	#	ind = 4
		
	#	running_mean_rew_qvl_offpol.append(read_and_calculate_score(file_dir+file_name, ind))

	for j in range(1):

		#file_dir2 = "Results Seed/QVloss_offpol_seed/"
		file_dir2 = "QVloss_offpol_seed_2/"
		file_dir3 = "outputs_" + str(j)
		
		file_dir = file_dir2 + file_dir3 + "/"
		ind = 4
		
		running_mean_rew_qvl_offpol_seed.append(read_and_calculate_score(file_dir+file_name, ind))


	#for j in range(10):

	#	file_dir2 = "QVloss applied after 1k steps with a weight of 1/"
	#	file_dir3 = "outputs_" + str(j)
		
	#	file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
	#	ind = 4
		
	#	running_mean_rew_qvl_onpol_aft1k_w1.append(read_and_calculate_score(file_dir+file_name, ind))

	for j in range(10):

		file_dir2 = "QVloss applied after 1k steps with a weight of 1 seed/"
		file_dir3 = "outputs_" + str(j)
		
		file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
		ind = 4
		
		running_mean_rew_qvl_onpol_aft1k_w1_seed.append(read_and_calculate_score(file_dir+file_name, ind))

	#for j in range(10):

	#	file_dir2 = "QVloss applied after 10k steps with a weight of 1/"
	#	file_dir3 = "outputs_" + str(j)
		
	#	file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
	#	ind = 4
		
	#	running_mean_rew_qvl_onpol_aft10k_w1.append(read_and_calculate_score(file_dir+file_name, ind))

	for j in range(10):

		file_dir2 = "QVloss applied after 10k steps with a weight of 1 seed/"
		file_dir3 = "outputs_" + str(j)
		
		file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
		ind = 4
		
		running_mean_rew_qvl_onpol_aft10k_w1_seed.append(read_and_calculate_score(file_dir+file_name, ind))

	#for j in range(10):

	#	file_dir2 = "QVloss applied after 100k steps with a weight of 1/"
	#	file_dir3 = "outputs_" + str(j)
		
	#	file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
	#	ind = 4
		
	#	running_mean_rew_qvl_onpol_aft100k_w1.append(read_and_calculate_score(file_dir+file_name, ind))

	for j in range(10):

		file_dir2 = "QVloss applied after 100k steps with a weight of 1 seed/"
		file_dir3 = "outputs_" + str(j)
		
		file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
		ind = 4
		
		running_mean_rew_qvl_onpol_aft100k_w1_seed.append(read_and_calculate_score(file_dir+file_name, ind))


	#for j in range(10):

	#	file_dir2 = "Results/n=100 steps/"
	#	file_dir3 = "outputs_" + str(j)
		
	#	file_dir = file_dir2 + file_dir3 + "/"
	#	ind = 4
		
	#	running_mean_n100.append(read_and_calculate_score(file_dir+file_name, ind,is_rew100=True))


	for j in range(10):

		file_dir2 = "n=100 steps seed/"
		file_dir3 = "outputs_" + str(j)
		
		file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
		ind = 4
		
		running_mean_n100_seed.append(read_and_calculate_score(file_dir+file_name, ind,is_rew100=True))



	#for j in range(10):

	#	file_dir2 = "Results/n=1000 steps/"
	#	file_dir3 = "outputs_" + str(j)
		
	#	file_dir = file_dir2 + file_dir3 + "/"
	#	ind = 4
		
	#	running_mean_n1k.append(read_and_calculate_score(file_dir+file_name, ind,is_rew100=True))


	for j in range(10):

		file_dir2 = "n=1000 steps seed/"
		file_dir3 = "outputs_" + str(j)
		
		file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
		ind = 4
		
		running_mean_n1k_seed.append(read_and_calculate_score(file_dir+file_name, ind,is_rew100=True))


	#for j in range(10):

	#	file_dir2 = "Results/n=10,000 steps/"
	#	file_dir3 = "outputs_" + str(j)
		
	#	file_dir = file_dir2 + file_dir3 + "/"
	#	ind = 4
		
	#	running_mean_n10k.append(read_and_calculate_score(file_dir+file_name, ind,is_rew100=True))


	for j in range(10):

		file_dir2 = "n=10,000 steps seed/"
		file_dir3 = "outputs_" + str(j)
		
		file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
		ind = 4
		
		running_mean_n10k_seed.append(read_and_calculate_score(file_dir+file_name, ind,is_rew100=True))



	#for j in range(10):
	
	#	file_dir2 = "baseline/"
	#	file_dir3 = "outputs_" + str(j)# + "_5m"
		
	#	file_dir = file_dir2 + file_dir3 + "/"
	#	ind = 1
		
	#	running_mean_rew_qnet.append(read_and_calculate_score(file_dir+file_name, ind))

	for j in range(10):
	
		file_dir2 = "baseline_seed/"
		file_dir3 = "outputs_" + str(j)# + "_5m"
		
		file_dir = file_dir2 + file_dir3 + "/"
		ind = 1
		
		running_mean_rew_qnet_seed.append(read_and_calculate_score(file_dir+file_name, ind))


	for j in range(1):
	
		file_dir2 = "baseline/"
		file_dir3 = "outputs_" + str(j) + "_20m"
		
		file_dir = file_dir2 + file_dir3 + "/"
		ind = 1
		
		running_mean_rew_qnet_20m.append(read_and_calculate_score(file_dir+file_name, ind))

	#for j in range(10):
	
	#	file_dir2 = "baseline and plot/"
	#	file_dir3 = "baseline_500k/"
		
	#	file_dir = file_dir2 + file_dir3
	#	ind = 1

	#	f_path = file_dir + "rews_baseline_" + str(j) + ".pkl"
	#	running_mean_rew_qnet_seed_2.append(cPickle.load( open( f_path, "rb" ) ))


	for j in range(5):
	
		file_dir2 = "baseline_eps_25k/"
		file_dir3 = "outputs_" + str(j)# + "_5m"
		
		file_dir = file_dir2 + file_dir3 + "/"
		ind = 1
		
		running_mean_rew_qnet_eps25k.append(read_and_calculate_score(file_dir+file_name, ind))


	for j in range(5):
	
		file_dir2 = "baseline_eps_100k/"
		file_dir3 = "outputs_" + str(j)# + "_5m"
		
		file_dir = file_dir2 + file_dir3 + "/"
		ind = 1
		
		running_mean_rew_qnet_eps100k.append(read_and_calculate_score(file_dir+file_name, ind))


	#for j in range(10):
	#
	#	file_dir2 = "Qlearning baseline vision=10/"
	#	file_dir3 = "outputs_" + str(j)
	#	
	#	file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
	#	ind = 1
	#	
	#	running_mean_rew_qnet_vis10.append(read_and_calculate_score(file_dir+file_name, ind))

	#for j in range(10):
	#
	#	file_dir2 = "n=1000 steps/"
	#	file_dir3 = "outputs_" + str(j)
	#	
	#	file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
	#	ind = 4
	#	
	#	running_mean_n1000.append(read_and_calculate_score(file_dir+file_name, ind, is_rew100=True))

	#for j in range(10):
	#
	#	file_dir2 = "on policy 0_1ln(i) schedule/"
	#	file_dir3 = "outputs_" + str(j)
	#	
	#	file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
	#	ind = 4
	#	
	#	running_mean_ln1.append(read_and_calculate_score(file_dir+file_name, ind))

	#for j in range(10):

	#	file_dir2 = "on policy 0_1ln(i) schedule2/"
	#	file_dir3 = "outputs_" + str(j)
		
	#	file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
	#	ind = 4
		
	#	running_mean_ln2.append(read_and_calculate_score(file_dir+file_name, ind))




	#print(len(running_mean_rew_qvl_onpol))
	#print(np.size(np.sum(running_mean_rew_qvl_onpol,axis=0)/5))

	#mean_qvl_onpol = np.sum(running_mean_rew_qvl_onpol,axis=0)/len(running_mean_rew_qvl_onpol)
	#mean_qvl_onpol = np.mean(running_mean_rew_qvl_onpol,axis=0)
	#std_qvl_onpol = np.std(running_mean_rew_qvl_onpol,axis=0)

	mean_qvl_onpol_seed = np.mean(running_mean_rew_qvl_onpol_seed,axis=0)
	std_qvl_onpol_seed = np.std(running_mean_rew_qvl_onpol_seed,axis=0)

	mean_qvl_onpol_20m = np.mean(running_mean_rew_qvl_onpol_20m,axis=0)
	std_qvl_onpol_20m = np.std(running_mean_rew_qvl_onpol_20m,axis=0)

	#mean_qvl_onpol_vis10 = np.mean(running_mean_rew_qvl_onpol_vis10,axis=0)
	#std_qvl_onpol_vis10 = np.std(running_mean_rew_qvl_onpol_vis10,axis=0)
	#print(std_qvl_onpol)
	#mean_qnet = np.mean(running_mean_rew_qnet,axis=0)
	#std_qnet = np.std(running_mean_rew_qnet,axis=0)

	mean_qnet_seed = np.mean(running_mean_rew_qnet_seed,axis=0)
	std_qnet_seed = np.std(running_mean_rew_qnet_seed,axis=0)

	mean_qnet_20m = np.mean(running_mean_rew_qnet_20m,axis=0)
	std_qnet_20m = np.std(running_mean_rew_qnet_20m,axis=0)

	#mean_qnet_seed_2 = np.mean(running_mean_rew_qnet_seed_2,axis=0)
	#std_qnet_seed_2 = np.std(running_mean_rew_qnet_seed_2,axis=0)

	#mean_qnet_vis10 = np.mean(running_mean_rew_qnet_vis10,axis=0)
	#std_qnet_vis10 = np.std(running_mean_rew_qnet_vis10,axis=0)

	#mean_qvl_offpol = np.mean(running_mean_rew_qvl_offpol,axis=0)
	#std_qvl_offpol = np.std(running_mean_rew_qvl_offpol,axis=0)

	mean_qvl_offpol_seed = np.mean(running_mean_rew_qvl_offpol_seed,axis=0)
	std_qvl_offpol_seed = np.std(running_mean_rew_qvl_offpol_seed,axis=0)

	#mean_qvl_onpol_after1k_w1 = np.mean(running_mean_rew_qvl_onpol_aft1k_w1,axis=0)
	#std_qvl_onpol_after1k_w1 = np.std(running_mean_rew_qvl_onpol_aft1k_w1,axis=0)

	mean_qvl_onpol_after1k_w1_seed = np.mean(running_mean_rew_qvl_onpol_aft1k_w1_seed,axis=0)
	std_qvl_onpol_after1k_w1_seed = np.std(running_mean_rew_qvl_onpol_aft1k_w1_seed,axis=0)

	#mean_qvl_onpol_after10k_w1 = np.mean(running_mean_rew_qvl_onpol_aft10k_w1,axis=0)
	#std_qvl_onpol_after10k_w1 = np.std(running_mean_rew_qvl_onpol_aft10k_w1,axis=0)

	mean_qvl_onpol_after10k_w1_seed = np.mean(running_mean_rew_qvl_onpol_aft10k_w1_seed,axis=0)
	std_qvl_onpol_after10k_w1_seed = np.std(running_mean_rew_qvl_onpol_aft10k_w1_seed,axis=0)

	#mean_qvl_onpol_after100k_w1 = np.mean(running_mean_rew_qvl_onpol_aft100k_w1,axis=0)
	#std_qvl_onpol_after100k_w1 = np.std(running_mean_rew_qvl_onpol_aft100k_w1,axis=0)

	mean_qvl_onpol_after100k_w1_seed = np.mean(running_mean_rew_qvl_onpol_aft100k_w1_seed,axis=0)
	std_qvl_onpol_after100k_w1_seed = np.std(running_mean_rew_qvl_onpol_aft100k_w1_seed,axis=0)

	mean_n100 = np.mean(running_mean_n100,axis=0)
	std_n100 = np.std(running_mean_n100,axis=0)

	mean_n100_seed = np.mean(running_mean_n100_seed,axis=0)
	std_n100_seed = np.std(running_mean_n100_seed,axis=0)

	mean_n1k = np.mean(running_mean_n1k,axis=0)
	std_n1k = np.std(running_mean_n1k,axis=0)

	mean_n1k_seed = np.mean(running_mean_n1k_seed,axis=0)
	std_n1k_seed = np.std(running_mean_n1k_seed,axis=0)

	mean_n10k = np.mean(running_mean_n10k,axis=0)
	std_n10k = np.std(running_mean_n10k,axis=0)

	mean_n10k_seed = np.mean(running_mean_n10k_seed,axis=0)
	std_n10k_seed = np.std(running_mean_n10k_seed,axis=0)

	#mean_ln1 = np.mean(running_mean_ln1,axis=0)
	#std_ln1 = np.std(running_mean_ln1,axis=0)
	
	#mean_ln2 = np.mean(running_mean_ln2,axis=0)
	#std_ln2 = np.std(running_mean_ln2,axis=0)

	mean_qnet_eps25k = np.mean(running_mean_rew_qnet_eps25k,axis=0)
	std_qnet_eps25k = np.std(running_mean_rew_qnet_eps25k,axis=0)

	mean_qnet_eps100k = np.mean(running_mean_rew_qnet_eps100k,axis=0)
	std_qnet_eps100k = np.std(running_mean_rew_qnet_eps100k,axis=0)

	#mean_qnet_exprep5k = np.mean(running_mean_rew_qnet_exprep5k,axis=0)
	#std_qnet_exprep5k = np.std(running_mean_rew_qnet_exprep5k,axis=0)
	
	
	#print(qqq)

	x = range(1,len(mean_qvl_onpol_seed)+1)
	print(len(x))
	x2 = range(1,len(mean_qnet_eps25k)+1)
	print(len(x2))
	x3 = range(1,len(mean_qnet_eps100k)+1)
	print(len(x3))

	x4 = range(1,len(mean_qnet_20m)+1)
	x5 = range(1,len(mean_qvl_onpol_20m)+1)


	alp = 0.3
	x_lims = [0,510000]

	# Figure 1
	fig, ax = plt.subplots()
	ax.plot(mean_qnet_seed,linewidth=5, color='red', label='Q-net')
	ax.fill_between(x, mean_qnet_seed - std_qnet_seed, mean_qnet_seed + std_qnet_seed,color='red',alpha=alp)
	#ax.plot(mean_qvl_onpol,linewidth=5, color='blue', label='QV')
	#ax.fill_between(x, mean_qvl_onpol - std_qvl_onpol, mean_qvl_onpol + std_qvl_onpol,color='blue',alpha=alp)
	ax.plot(mean_qvl_onpol_seed,linewidth=5, color='blue', label='QV')
	ax.fill_between(x, mean_qvl_onpol_seed - std_qvl_onpol_seed, mean_qvl_onpol_seed + std_qvl_onpol_seed,color='blue',alpha=alp)
	#ax.plot(mean_qvl_offpol,linewidth=5, color='green', label='QV-t')
	#ax.fill_between(x, mean_qvl_offpol - std_qvl_offpol, mean_qvl_offpol + std_qvl_offpol,color='green',alpha=alp)
	#ax.plot(mean_qvl_offpol_seed,linewidth=5, color='green', label='QV-t')
	#ax.fill_between(x, mean_qvl_offpol_seed - std_qvl_offpol_seed, mean_qvl_offpol_seed + std_qvl_offpol_seed,color='green',alpha=alp)

	ax.set_ylim([0,0.08])
	ax.set_xlim(x_lims)
	#ax.set_title('Baseline vs Q-V-loss and Q-V-loss target')
	ax.set(xlabel='Steps in the environment', ylabel='R')
	ax.legend(loc='upper right', fontsize='x-large', ncol=3)
	ax.grid()

	# Figure 2
	fig2, ax2 = plt.subplots()
	ax2.plot(mean_qnet_seed,linewidth=5, color='red', label='Q-net')
	ax2.fill_between(x, mean_qnet_seed - std_qnet_seed, mean_qnet_seed + std_qnet_seed,color='red',alpha=alp)
	ax2.plot(mean_qvl_onpol_seed,linewidth=5, color='blue', label='QV')
	ax2.fill_between(x, mean_qvl_onpol_seed - std_qvl_onpol_seed, mean_qvl_onpol_seed + std_qvl_onpol_seed,color='blue',alpha=alp)
	#ax2.plot(mean_qvl_onpol_after1k_w1,linewidth=5, color='green', label='QV-S1')
	#ax2.fill_between(x, mean_qvl_onpol_after1k_w1 - std_qvl_onpol_after1k_w1, mean_qvl_onpol_after1k_w1 + std_qvl_onpol_after1k_w1,color='green',alpha=alp)
	ax2.plot(mean_qvl_onpol_after1k_w1_seed,linewidth=5, color='green', label='QV-S1')
	ax2.fill_between(x, mean_qvl_onpol_after1k_w1_seed - std_qvl_onpol_after1k_w1_seed, mean_qvl_onpol_after1k_w1_seed + std_qvl_onpol_after1k_w1_seed,color='green',alpha=alp)
	#ax2.plot(mean_qvl_onpol_after10k_w1,linewidth=5, color='magenta', label='QV-S2')
	#ax2.fill_between(x, mean_qvl_onpol_after10k_w1 - std_qvl_onpol_after10k_w1, mean_qvl_onpol_after10k_w1 + std_qvl_onpol_after10k_w1,color='magenta',alpha=alp)
	ax2.plot(mean_qvl_onpol_after10k_w1_seed,linewidth=5, color='orange', label='QV-S2')
	ax2.fill_between(x, mean_qvl_onpol_after10k_w1_seed - std_qvl_onpol_after10k_w1_seed, mean_qvl_onpol_after10k_w1_seed + std_qvl_onpol_after10k_w1_seed,color='orange',alpha=alp)
	#ax2.plot(mean_qvl_onpol_after100k_w1,linewidth=5, color='orange', label='QV-S3')
	#ax2.fill_between(x, mean_qvl_onpol_after100k_w1 - std_qvl_onpol_after100k_w1, mean_qvl_onpol_after100k_w1 + std_qvl_onpol_after100k_w1,color='orange',alpha=alp)
	ax2.plot(mean_qvl_onpol_after100k_w1_seed,linewidth=5, color='black', label='QV-S3')
	ax2.fill_between(x, mean_qvl_onpol_after100k_w1_seed - std_qvl_onpol_after100k_w1_seed, mean_qvl_onpol_after100k_w1_seed + std_qvl_onpol_after100k_w1_seed,color='black',alpha=alp)


	ax2.set_ylim([0,0.08])
	ax2.set_xlim(x_lims)
	#ax2.set_title('Baseline vs Q-V-loss and Q-V-loss target')
	ax2.set(xlabel='Steps in the environment', ylabel='R')
	ax2.legend(loc='upper right', fontsize='x-large', ncol=3)
	ax2.grid()


	# Figure 3
	fig3, ax3 = plt.subplots()
	ax3.plot(mean_qnet_seed,linewidth=5, color='red', label='Q-net')
	ax3.fill_between(x, mean_qnet_seed - std_qnet_seed, mean_qnet_seed + std_qnet_seed,color='red',alpha=alp)
	ax3.plot(mean_qvl_onpol_seed,linewidth=5, color='blue', label='QV')
	ax3.fill_between(x, mean_qvl_onpol_seed - std_qvl_onpol_seed, mean_qvl_onpol_seed + std_qvl_onpol_seed,color='blue',alpha=alp)
	#ax3.plot(mean_n100,linewidth=5, color='green', label='QV-n1')
	#ax3.fill_between(x, mean_n100 - std_n100, mean_n100 + std_n100,color='green',alpha=alp)
	#ax3.plot(mean_n1k,linewidth=5, color='black', label='QV-n2')
	#ax3.fill_between(x, mean_n1k - std_n1k, mean_n1k + std_n1k,color='black',alpha=alp)
	#ax3.plot(mean_n10k,linewidth=5, color='orange', label='QV-n3')
	#ax3.fill_between(x, mean_n10k - std_n10k, mean_n10k + std_n10k,color='orange',alpha=alp)
	ax3.plot(mean_n100_seed,linewidth=5, color='green', label='QV-n1')
	ax3.fill_between(x, mean_n100_seed - std_n100_seed, mean_n100_seed + std_n100_seed,color='green',alpha=alp)
	ax3.plot(mean_n1k_seed,linewidth=5, color='black', label='QV-n2')
	ax3.fill_between(x, mean_n1k_seed - std_n1k_seed, mean_n1k_seed + std_n1k_seed,color='black',alpha=alp)
	ax3.plot(mean_n10k_seed,linewidth=5, color='orange', label='QV-n3')
	ax3.fill_between(x, mean_n10k_seed - std_n10k_seed, mean_n10k_seed + std_n10k_seed,color='orange',alpha=alp)
	
	ax3.set_ylim([0,0.08])
	ax3.set_xlim(x_lims)
	#ax2.set_title('Baseline vs Q-V-loss and Q-V-loss target')
	ax3.set(xlabel='Steps in the environment', ylabel='R')
	ax3.legend(loc='upper right', fontsize='x-large', ncol=3)
	ax3.grid()

	# Figure 4 - Baseline only
	fig4, ax4 = plt.subplots()
	#ax4.plot(mean_qnet,linewidth=5, color='red', label='Q-net old')
	#ax4.fill_between(x, mean_qnet - std_qnet, mean_qnet + std_qnet,color='red',alpha=alp)
	ax4.plot(mean_qnet_seed,linewidth=5, color='red', label='Q-net Baseline')
	ax4.fill_between(x, mean_qnet_seed - std_qnet_seed, mean_qnet_seed + std_qnet_seed,color='red',alpha=alp)
	#ax4.plot(mean_qnet_seed_2,linewidth=5, color='magenta', label='Q-net seed 2')
	#ax4.fill_between(x, mean_qnet_seed_2 - std_qnet_seed_2, mean_qnet_seed_2 + std_qnet_seed_2,color='magenta',alpha=alp)
	ax4.set_ylim([0,0.055])
	ax4.set_xlim(x_lims)
	ax4.set_title('Baseline Q-Learning Results')
	ax4.set(xlabel='Steps in the environment', ylabel='R')
	ax4.legend(loc='upper right', fontsize='x-large', ncol=3)
	ax4.grid()

	# Figure 5
	fig5, ax5 = plt.subplots()
	ax5.plot(mean_qnet_20m,linewidth=5, color='red', label='Q-net')
	#ax5.fill_between(x, mean_qnet_20m - std_qnet_seed, mean_qnet_seed + std_qnet_seed,color='red',alpha=alp)
	#ax.plot(mean_qvl_onpol,linewidth=5, color='blue', label='QV')
	#ax.fill_between(x, mean_qvl_onpol - std_qvl_onpol, mean_qvl_onpol + std_qvl_onpol,color='blue',alpha=alp)
	ax5.plot(mean_qvl_onpol_20m,linewidth=5, color='blue', label='QV')
	#ax5.fill_between(x, mean_qvl_onpol_seed - std_qvl_onpol_seed, mean_qvl_onpol_seed + std_qvl_onpol_seed,color='blue',alpha=alp)
	#ax.plot(mean_qvl_offpol,linewidth=5, color='green', label='QV-t')
	#ax.fill_between(x, mean_qvl_offpol - std_qvl_offpol, mean_qvl_offpol + std_qvl_offpol,color='green',alpha=alp)
	#ax.plot(mean_qvl_offpol_seed,linewidth=5, color='green', label='QV-t')
	#ax.fill_between(x, mean_qvl_offpol_seed - std_qvl_offpol_seed, mean_qvl_offpol_seed + std_qvl_offpol_seed,color='green',alpha=alp)

	ax5.set_ylim([0,0.08])
	#ax5.set_xlim(x_lims)
	#ax.set_title('Baseline vs Q-V-loss and Q-V-loss target')
	ax5.set(xlabel='Steps in the environment', ylabel='R')
	ax5.legend(loc='upper right', fontsize='x-large', ncol=3)
	ax5.grid()

	# Figure 6
	fig6, ax6 = plt.subplots()
	ax6.plot(mean_qnet_seed,linewidth=5, color='red', label='Q-net')
	ax6.fill_between(x, mean_qnet_seed - std_qnet_seed, mean_qnet_seed + std_qnet_seed,color='red',alpha=alp)
	ax6.plot(mean_qvl_onpol_seed,linewidth=5, color='blue', label='QV')
	ax6.fill_between(x, mean_qvl_onpol_seed - std_qvl_onpol_seed, mean_qvl_onpol_seed + std_qvl_onpol_seed,color='blue',alpha=alp)
	#ax2.plot(mean_qvl_onpol_after1k_w1,linewidth=5, color='green', label='QV-S1')
	#ax2.fill_between(x, mean_qvl_onpol_after1k_w1 - std_qvl_onpol_after1k_w1, mean_qvl_onpol_after1k_w1 + std_qvl_onpol_after1k_w1,color='green',alpha=alp)
	ax6.plot(mean_qvl_onpol_after1k_w1_seed,linewidth=5, color='green', label='QV-S1')
	ax6.fill_between(x, mean_qvl_onpol_after1k_w1_seed - std_qvl_onpol_after1k_w1_seed, mean_qvl_onpol_after1k_w1_seed + std_qvl_onpol_after1k_w1_seed,color='green',alpha=alp)
	#ax2.plot(mean_qvl_onpol_after10k_w1,linewidth=5, color='magenta', label='QV-S2')
	#ax2.fill_between(x, mean_qvl_onpol_after10k_w1 - std_qvl_onpol_after10k_w1, mean_qvl_onpol_after10k_w1 + std_qvl_onpol_after10k_w1,color='magenta',alpha=alp)
	ax6.plot(mean_qvl_onpol_after10k_w1_seed,linewidth=5, color='orange', label='QV-S2')
	ax6.fill_between(x, mean_qvl_onpol_after10k_w1_seed - std_qvl_onpol_after10k_w1_seed, mean_qvl_onpol_after10k_w1_seed + std_qvl_onpol_after10k_w1_seed,color='orange',alpha=alp)
	#ax2.plot(mean_qvl_onpol_after100k_w1,linewidth=5, color='orange', label='QV-S3')
	#ax2.fill_between(x, mean_qvl_onpol_after100k_w1 - std_qvl_onpol_after100k_w1, mean_qvl_onpol_after100k_w1 + std_qvl_onpol_after100k_w1,color='orange',alpha=alp)
	ax6.plot(mean_qvl_onpol_after100k_w1_seed,linewidth=5, color='black', label='QV-S3')
	ax6.fill_between(x, mean_qvl_onpol_after100k_w1_seed - std_qvl_onpol_after100k_w1_seed, mean_qvl_onpol_after100k_w1_seed + std_qvl_onpol_after100k_w1_seed,color='black',alpha=alp)


	ax6.set_ylim([0.035,0.06])
	ax6.set_xlim([100000,200000])
	#ax2.set_title('Baseline vs Q-V-loss and Q-V-loss target')
	ax6.set(xlabel='Steps in the environment', ylabel='R')
	ax6.legend(loc='upper right', fontsize='x-large', ncol=3)
	ax6.grid()

	# Figure 5 - Baseline bump
	#fig5, ax5 = plt.subplots()
	#ax5.plot(mean_qnet,linewidth=5, color='red', label='Q-net eps_final 50k')
	#ax5.fill_between(x, mean_qnet - std_qnet, mean_qnet + std_qnet,color='red',alpha=alp)
	#ax5.plot(mean_qnet_eps25k,linewidth=5, color='blue', label='Q-net eps_final 25k')
	#ax5.fill_between(x2, mean_qnet_eps25k - std_qnet_eps25k, mean_qnet_eps25k + std_qnet_eps25k,color='blue',alpha=alp)
	#ax5.plot(mean_qnet_eps100k,linewidth=5, color='green', label='Q-net eps_final 100k')
	#ax5.fill_between(x3, mean_qnet_eps100k - std_qnet_eps100k, mean_qnet_eps100k + std_qnet_eps100k,color='green',alpha=alp)
	#ax5.plot(mean_qnet_exprep5k,linewidth=5, color='black', label='Q-net exp_rep 5k')
	#ax5.fill_between(x2, mean_qnet_exprep5k - std_qnet_exprep5k, mean_qnet_exprep5k + std_qnet_exprep5k,color='black',alpha=alp)
	#ax5.plot(range(1,len(running_mean_rew_qnet_eps25k[0])+1),running_mean_rew_qnet_eps25k[0],linewidth=5, color='red', label='Q-net eps25k')
	#ax5.set_ylim([0,0.01])
	#ax5.set_xlim([0,50000])
	#ax.set_title('Baseline vs Q-V-loss and Q-V-loss target')
	#ax5.set(xlabel='Steps in the environment', ylabel='R')
	#ax5.legend(loc='upper right', fontsize='x-large', ncol=3)
	#ax5.grid()

	# Figure 6 - QV- QVt bump
	#fig6, ax6 = plt.subplots()
	#ax6.plot(mean_qnet,linewidth=5, color='red', label='Q-net')
	#ax6.fill_between(x, mean_qnet - std_qnet, mean_qnet + std_qnet,color='red',alpha=alp)
	#ax6.plot(mean_qvl_onpol,linewidth=5, color='blue', label='QV')
	#ax6.fill_between(x, mean_qvl_onpol - std_qvl_onpol, mean_qvl_onpol + std_qvl_onpol,color='blue',alpha=alp)
	#ax6.plot(mean_qvl_offpol,linewidth=5, color='green', label='QV-t')
	#ax6.fill_between(x, mean_qvl_offpol - std_qvl_offpol, mean_qvl_offpol + std_qvl_offpol,color='green',alpha=alp)
	#ax6.set_ylim([0,0.01])
	#ax6.set_xlim([0,50000])
	#ax.set_title('Baseline vs Q-V-loss and Q-V-loss target')
	#ax6.set(xlabel='Steps in the environment', ylabel='R')
	#ax6.legend(loc='lower right', fontsize='x-large', ncol=3)
	#ax6.grid()

	# Figure 7 - QV-k bump
	#fig7, ax7 = plt.subplots()
	#ax7.plot(mean_qnet,linewidth=5, color='red', label='Q-net')
	#ax7.fill_between(x, mean_qnet - std_qnet, mean_qnet + std_qnet,color='red',alpha=alp)
	#ax7.plot(mean_qvl_onpol,linewidth=5, color='blue', label='QV')
	#ax7.fill_between(x, mean_qvl_onpol - std_qvl_onpol, mean_qvl_onpol + std_qvl_onpol,color='blue',alpha=alp)
	#ax7.plot(mean_qvl_onpol_after1k_w1,linewidth=5, color='green', label='QV-S1')
	#ax7.fill_between(x, mean_qvl_onpol_after1k_w1 - std_qvl_onpol_after1k_w1, mean_qvl_onpol_after1k_w1 + std_qvl_onpol_after1k_w1,color='green',alpha=alp)
	#ax7.plot(mean_qvl_onpol_after10k_w1,linewidth=5, color='magenta', label='QV-S2')
	#ax7.fill_between(x, mean_qvl_onpol_after10k_w1 - std_qvl_onpol_after10k_w1, mean_qvl_onpol_after10k_w1 + std_qvl_onpol_after10k_w1,color='magenta',alpha=alp)
	#ax7.plot(mean_qvl_onpol_after100k_w1,linewidth=5, color='orange', label='QV-S3')
	#ax7.fill_between(x, mean_qvl_onpol_after100k_w1 - std_qvl_onpol_after100k_w1, mean_qvl_onpol_after100k_w1 + std_qvl_onpol_after100k_w1,color='orange',alpha=alp)

	#ax7.set_ylim([0,0.01])
	#ax7.set_xlim([0,50000])
	#ax.set_title('Baseline vs Q-V-loss and Q-V-loss target')
	#ax7.set(xlabel='Steps in the environment', ylabel='R')
	#ax7.legend(loc='lower right', fontsize='x-large', ncol=3)
	#ax7.grid()

	# Figure 8 - QV-n bump
	#fig8, ax8 = plt.subplots()
	#ax8.plot(mean_qnet,linewidth=5, color='red', label='Q-net')
	#ax8.fill_between(x, mean_qnet - std_qnet, mean_qnet + std_qnet,color='red',alpha=alp)
	#ax8.plot(mean_qvl_onpol,linewidth=5, color='blue', label='QV')
	#ax8.fill_between(x, mean_qvl_onpol - std_qvl_onpol, mean_qvl_onpol + std_qvl_onpol,color='blue',alpha=alp)
	#ax8.plot(mean_n100,linewidth=5, color='green', label='QV-n1')
	#ax8.fill_between(x, mean_n100 - std_n100, mean_n100 + std_n100,color='green',alpha=alp)
	#ax8.plot(mean_n1k,linewidth=5, color='black', label='QV-n2')
	#ax8.fill_between(x, mean_n1k - std_n1k, mean_n1k + std_n1k,color='black',alpha=alp)
	#ax8.plot(mean_n10k,linewidth=5, color='orange', label='QV-n3')
	#ax8.fill_between(x, mean_n10k - std_n10k, mean_n10k + std_n10k,color='orange',alpha=alp)

	#ax8.set_ylim([0,0.01])
	#ax8.set_xlim([0,50000])
	#ax.set_title('Baseline vs Q-V-loss and Q-V-loss target')
	#ax8.set(xlabel='Steps in the environment', ylabel='R')
	#ax8.legend(loc='lower right', fontsize='x-large', ncol=3)
	#ax8.grid()


	#Figure 9 - Individual baselines
	fig9, ax9 = plt.subplots()
	ax9.plot(running_mean_rew_qnet_seed[0],linewidth=5, color='red', label='1')
	ax9.plot(running_mean_rew_qnet_seed[1],linewidth=5, color='black', label='2')
	ax9.plot(running_mean_rew_qnet_seed[2],linewidth=5, color='green', label='3')
	ax9.plot(running_mean_rew_qnet_seed[3],linewidth=5, color='blue', label='4')
	ax9.plot(running_mean_rew_qnet_seed[4],linewidth=5, color='orange', label='5')
	ax9.plot(running_mean_rew_qnet_seed[5],linewidth=5, color='magenta', label='6')
	ax9.plot(running_mean_rew_qnet_seed[6],linewidth=5, color='pink', label='7')
	ax9.plot(running_mean_rew_qnet_seed[7],linewidth=5, color='cyan', label='8')
	ax9.plot(running_mean_rew_qnet_seed[8],linewidth=5, color='yellow', label='9')
	ax9.plot(running_mean_rew_qnet_seed[9],linewidth=5, color='grey', label='10')


	ax9.set_ylim([0,0.01])
	ax9.set_xlim([0,50000])
	#ax.set_title('Baseline vs Q-V-loss and Q-V-loss target')
	ax9.set(xlabel='Steps in the environment', ylabel='R')
	ax9.legend(loc='lower right', fontsize='x-large', ncol=4)
	ax9.grid()
		

	#ax.plot(running_mean_rew_qvl_onpol[0])
	#ax.plot(running_mean_rew_qvl_onpol[1])
	#ax.plot(running_mean_rew_qvl_onpol[2])
	#ax.plot(running_mean_rew_qvl_onpol[3])
	#ax.plot(running_mean_rew_qvl_onpol[4])

	#ax.plot(running_mean_rew_qnet[0])
	#ax.plot(running_mean_rew_qnet[1])
	#ax.plot(running_mean_rew_qnet[2])
	#ax.plot(running_mean_rew_qnet[3])
	#ax.plot(running_mean_rew_qnet[4])



	plt.plot()
	fig.savefig("coupled_results_qv_qvt.png")
	fig2.savefig("coupled_results_aft.png")
	fig3.savefig("coupled_results_n.png")
	fig4.savefig("baselines_results.png")
	fig5.savefig("coupled_vs_qnet_20m.png")
	fig6.savefig("coupled_results_aft_uplose.png")
	#fig7.savefig("qv_k_bump.png")
	#fig8.savefig("qv_n_bump.png")
	fig9.savefig("baseline_bumps_individual.png")
	#fig.savefig("coupled_results_v2.eps")
	#fig.savefig("baselines_results.png")
	plt.show()


if __name__ == '__main__':
    main()
