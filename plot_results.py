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

	fig, ax = plt.subplots()
	running_mean_rew_qvl_onpol = []
	running_mean_rew_qvl_onpol_vis10 = []
	running_mean_rew_qnet = []
	running_mean_rew_qnet_vis10 = []
	running_mean_rew_qvl_offpol = []

	running_mean_rew_qvl_onpol_aft1k_w1 = []
	running_mean_rew_qvl_onpol_aft10k_w1 = []
	running_mean_rew_qvl_onpol_aft100k_w1 = []

	
	running_mean_n100 = []
	running_mean_n1k = []
	running_mean_n10k = []
	running_mean_ln1 = []
	running_mean_ln2 = []

	file_dir1 = "Results/"
	file_name = "train_stats.pkl"
	

	for j in range(5):
		file_dir2 = "QVLoss_onpol/"
		file_dir3 = "outputs_" + str(j)
		
		file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
		ind = 4
		
		running_mean_rew_qvl_onpol.append(read_and_calculate_score(file_dir+file_name, ind)[0:500001])


	for j in range(5):

		file_dir2 = "QVLoss_offpol/"
		file_dir3 = "outputs_" + str(j)
		
		file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
		ind = 4
		
		running_mean_rew_qvl_offpol.append(read_and_calculate_score(file_dir+file_name, ind))


	for j in range(10):

		file_dir2 = "QVloss applied after 1k steps with a weight of 1/"
		file_dir3 = "outputs_" + str(j)
		
		file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
		ind = 4
		
		running_mean_rew_qvl_onpol_aft1k_w1.append(read_and_calculate_score(file_dir+file_name, ind))

	for j in range(5):

		file_dir2 = "QVloss applied after 10k steps with a weight of 1/"
		file_dir3 = "outputs_" + str(j)
		
		file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
		ind = 4
		
		running_mean_rew_qvl_onpol_aft10k_w1.append(read_and_calculate_score(file_dir+file_name, ind))

	for j in range(10):

		file_dir2 = "QVloss applied after 100k steps with a weight of 1/"
		file_dir3 = "outputs_" + str(j)
		
		file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
		ind = 4
		
		running_mean_rew_qvl_onpol_aft100k_w1.append(read_and_calculate_score(file_dir+file_name, ind))


	for j in range(10):

		file_dir2 = "n=100 steps/"
		file_dir3 = "outputs_" + str(j)
		
		file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
		ind = 4
		
		running_mean_n100.append(read_and_calculate_score(file_dir+file_name, ind,is_rew100=True))

	for j in range(10):

		file_dir2 = "n=1000 steps/"
		file_dir3 = "outputs_" + str(j)
		
		file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
		ind = 4
		
		running_mean_n1k.append(read_and_calculate_score(file_dir+file_name, ind,is_rew100=True))


	for j in range(10):

		file_dir2 = "n=10,000 steps/"
		file_dir3 = "outputs_" + str(j)
		
		file_dir = file_dir1 + file_dir2 + file_dir3 + "/"
		ind = 4
		
		running_mean_n10k.append(read_and_calculate_score(file_dir+file_name, ind,is_rew100=True))



	for j in range(10):
	
		file_dir2 = "baseline/"
		file_dir3 = "outputs_" + str(j)# + "_5m"
		
		file_dir = file_dir2 + file_dir3 + "/"
		ind = 1
		
		running_mean_rew_qnet.append(read_and_calculate_score(file_dir+file_name, ind))

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
	mean_qvl_onpol = np.mean(running_mean_rew_qvl_onpol,axis=0)
	std_qvl_onpol = np.std(running_mean_rew_qvl_onpol,axis=0)

	#mean_qvl_onpol_vis10 = np.mean(running_mean_rew_qvl_onpol_vis10,axis=0)
	#std_qvl_onpol_vis10 = np.std(running_mean_rew_qvl_onpol_vis10,axis=0)
	#print(std_qvl_onpol)
	mean_qnet = np.mean(running_mean_rew_qnet,axis=0)
	std_qnet = np.std(running_mean_rew_qnet,axis=0)

	#mean_qnet_vis10 = np.mean(running_mean_rew_qnet_vis10,axis=0)
	#std_qnet_vis10 = np.std(running_mean_rew_qnet_vis10,axis=0)

	mean_qvl_offpol = np.mean(running_mean_rew_qvl_offpol,axis=0)
	std_qvl_offpol = np.std(running_mean_rew_qvl_offpol,axis=0)

	mean_qvl_onpol_after1k_w1 = np.mean(running_mean_rew_qvl_onpol_aft1k_w1,axis=0)
	std_qvl_onpol_after1k_w1 = np.std(running_mean_rew_qvl_onpol_aft1k_w1,axis=0)

	mean_qvl_onpol_after10k_w1 = np.mean(running_mean_rew_qvl_onpol_aft10k_w1,axis=0)
	std_qvl_onpol_after10k_w1 = np.std(running_mean_rew_qvl_onpol_aft10k_w1,axis=0)

	mean_qvl_onpol_after100k_w1 = np.mean(running_mean_rew_qvl_onpol_aft100k_w1,axis=0)
	std_qvl_onpol_after100k_w1 = np.std(running_mean_rew_qvl_onpol_aft100k_w1,axis=0)

	mean_n100 = np.mean(running_mean_n100,axis=0)
	std_n100 = np.std(running_mean_n100,axis=0)

	mean_n1k = np.mean(running_mean_n1k,axis=0)
	std_n1k = np.std(running_mean_n1k,axis=0)

	mean_n10k = np.mean(running_mean_n10k,axis=0)
	std_n10k = np.std(running_mean_n10k,axis=0)

	#mean_ln1 = np.mean(running_mean_ln1,axis=0)
	#std_ln1 = np.std(running_mean_ln1,axis=0)
	
	#mean_ln2 = np.mean(running_mean_ln2,axis=0)
	#std_ln2 = np.std(running_mean_ln2,axis=0)
	
	
	#print(qqq)

	x = range(1,len(mean_qvl_onpol)+1)
	print(len(x))
	#x2 = range(1,len(mean_qnet)+1)
	#print(len(x2))
	#ax.plot(mean_qnet,linewidth=5, color='red', label='Vision range = 5')
	#ax.fill_between(x, mean_qnet - std_qnet, mean_qnet + std_qnet,color='red',alpha=0.4)
	#ax.plot(mean_qnet_vis10,linewidth=5, color='magenta', label='Vision range = 10')
	#ax.fill_between(x, mean_qnet_vis10 - std_qnet_vis10, mean_qnet_vis10 + std_qnet_vis10,color='magenta',alpha=0.4)

	#ax.plot(mean_qnet,linewidth=5, color='red', label='Baseline Vision range = 5')
	#ax.fill_between(x, mean_qnet - std_qnet, mean_qnet + std_qnet,color='red',alpha=0.4)
	#ax.plot(mean_qnet_vis10,linewidth=5, color='magenta', label='Baseline Vision range = 10')
	#ax.fill_between(x, mean_qnet_vis10 - std_qnet_vis10, mean_qnet_vis10 + std_qnet_vis10,color='magenta',alpha=0.4)
	#ax.plot(mean_qvl_onpol,linewidth=5, color='blue', label='Q-V Loss on Policy')
	#ax.fill_between(x, mean_qvl_onpol - std_qvl_onpol, mean_qvl_onpol + std_qvl_onpol,color='blue',alpha=0.4)
	#ax.plot(mean_qvl_onpol_vis10,linewidth=5, color='cyan', label='Q-V Loss on Policy Vis = 10')
	#ax.fill_between(x, mean_qvl_onpol_vis10 - std_qvl_onpol_vis10, mean_qvl_onpol_vis10 + std_qvl_onpol_vis10,color='cyan',alpha=0.4)
	#ax.plot(mean_qvl_onpol_after10k_w1,linewidth=5, color='orange', label='Q-V Loss on Policy schedule 1')
	#ax.fill_between(x, mean_qvl_onpol_after10k_w1 - std_qvl_onpol_after10k_w1, mean_qvl_onpol_after10k_w1 + std_qvl_onpol_after10k_w1,color='orange',alpha=0.4)
	#ax.plot(mean_qvl_offpol,linewidth=5, color='green', label='Q-V Loss off Policy')
	#ax.fill_between(x, mean_qvl_offpol - std_qvl_offpol, mean_qvl_offpol + std_qvl_offpol,color='green',alpha=0.4)

	ax.plot(mean_qnet,linewidth=5, color='red', label='Q-net')
	ax.fill_between(x, mean_qnet - std_qnet, mean_qnet + std_qnet,color='red',alpha=0.4)
	ax.plot(mean_qvl_onpol,linewidth=5, color='blue', label='QV')
	ax.fill_between(x, mean_qvl_onpol - std_qvl_onpol, mean_qvl_onpol + std_qvl_onpol,color='blue',alpha=0.4)
	#ax.plot(mean_qvl_onpol_after1k_w1,linewidth=5, color='green', label='QV-S1')
	#ax.fill_between(x, mean_qvl_onpol_after1k_w1 - std_qvl_onpol_after1k_w1, mean_qvl_onpol_after1k_w1 + std_qvl_onpol_after1k_w1,color='green',alpha=0.4)
	#ax.plot(mean_qvl_onpol_after10k_w1,linewidth=5, color='magenta', label='QV-S2')
	#ax.fill_between(x, mean_qvl_onpol_after10k_w1 - std_qvl_onpol_after10k_w1, mean_qvl_onpol_after10k_w1 + std_qvl_onpol_after10k_w1,color='magenta',alpha=0.4)
	#ax.plot(mean_qvl_onpol_after100k_w1,linewidth=5, color='orange', label='QV-S3')
	#ax.fill_between(x, mean_qvl_onpol_after100k_w1 - std_qvl_onpol_after100k_w1, mean_qvl_onpol_after100k_w1 + std_qvl_onpol_after100k_w1,color='orange',alpha=0.4)
	
	ax.plot(mean_qvl_offpol,linewidth=5, color='green', label='QV-t')
	ax.fill_between(x, mean_qvl_offpol - std_qvl_offpol, mean_qvl_offpol + std_qvl_offpol,color='green',alpha=0.4)
	#ax.plot(mean_ln1,linewidth=5, color='cyan', label='QV-S2')
	#ax.fill_between(x, mean_ln1 - std_ln1, mean_ln1 + std_ln1,color='cyan',alpha=0.4)
	#ax.plot(mean_n100,linewidth=5, color='green', label='QV-n1')
	#ax.fill_between(x, mean_n100 - std_n100, mean_n100 + std_n100,color='green',alpha=0.4)
	#ax.plot(mean_n1k,linewidth=5, color='black', label='QV-n2')
	#ax.fill_between(x, mean_n1k - std_n1k, mean_n1k + std_n1k,color='black',alpha=0.4)
	#ax.plot(mean_n10k,linewidth=5, color='orange', label='QV-n3')
	#ax.fill_between(x, mean_n10k - std_n10k, mean_n10k + std_n10k,color='orange',alpha=0.4)
	#ax.plot(mean_ln2,linewidth=5, color='black', label='QV-S4')
	#ax.fill_between(x, mean_ln2 - std_ln2, mean_ln2 + std_ln2,color='black',alpha=0.4)
		

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



	#ax.set_ylim([0,0.055])
	ax.set_ylim([0,0.08])
	#ax.set_title('Coupled Q-Learning Results')
	ax.set_title('Baseline vs Q-V-loss and Q-V-loss target')
	ax.set(xlabel='Steps in the environment', ylabel='R')
	ax.legend(loc='upper right', fontsize='x-large', ncol=3)
	ax.grid()

	plt.plot()
	fig.savefig("coupled_results_qv_qvt.png")
	#fig.savefig("coupled_results_v2.eps")
	#fig.savefig("baselines_results.png")
	plt.show()


if __name__ == '__main__':
    main()
