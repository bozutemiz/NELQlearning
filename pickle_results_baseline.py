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


	#file_dir1 = "Results/"
	file_dir1 = ""
	file_name = "train_stats.pkl"

	output_dir = "baseline and plot/baseline_500k/"


	for j in range(10):
	
		file_dir2 = "baseline_seed/"
		file_dir3 = "outputs_" + str(j)
		
		file_dir = file_dir2 + file_dir3 + "/"
		ind = 1

		running_mean_rew_qnet = read_and_calculate_score(file_dir+file_name, ind)
		
		with open(output_dir + "rews_baseline_" + str(j) + ".pkl", 'wb') as f:
			cPickle.dump(running_mean_rew_qnet, f)
		#running_mean_rew_qnet.append(read_and_calculate_score(file_dir+file_name, ind))




if __name__ == '__main__':
    main()
