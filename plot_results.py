import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from six.moves import cPickle
from collections import deque
import numpy as np

def tot_reward_n_step(rewards, n=100):
	all_rewards = deque(maxlen=n)
	cum_rewards = []

	for i in range(len(rewards)):
		all_rewards.append(rewards[i])
		asd = np.sum(all_rewards)
		cum_rewards.append(asd)

	return cum_rewards

def main():

	# for j in range(5):

	#file_dir2 = "Qnet run " + str(j+1)
	file_dir2 = "" #"QVloss on policy run " + str(j+1)
	#file_dir2 = "QVloss off policy run " + str(j+1)

	file_dir1 = "outputs/"
	
	file_dir = file_dir1 + file_dir2 + "/"
	#file_dir = "outputs/"
	file_name = "train_stats.pkl"
	ind = 4

	data = cPickle.load( open( file_dir+file_name, "rb" ) )

	rews = [0]
	for i in range(len(data[ind])):
		if i < 100 and i > 0:
			rews.append(data[ind][i] - data[ind][i-1])
		if i >= 100:
			deleted_val = rews[i-100]
			curr_val = data[ind][i]
			prev_val = data[ind][i-1]
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
	
	#all_rewards = deque(maxlen=100)
	#rewards = []

	#for i in range(len(rews)):
	#	all_rewards.append(rews[i])
	#	asd = np.sum(all_rewards)
	#	rewards.append(asd)

	print('tot_rew :', np.sum(rews))
	tot_rew_10 = tot_reward_n_step(rews, n=10)
	tot_rew_1000 = tot_reward_n_step(rews, n=1000)
	print('max_rew10 :', np.max(tot_rew_10))
	print('max_rew100 :', np.max(data[ind]))
	print('max_rew1000 :', np.max(tot_rew_1000))
	print('ave_rew :', np.mean(rews))
	print('ave_rew10 :', np.mean(tot_rew_10))
	print('ave_rew100 :', np.mean(data[ind]))
	print('ave_rew1000 :', np.mean(tot_rew_1000))
	print('ave_rew_f100k :', np.mean(rews[0:100000]))
	print('ave_rew10_f100k :', np.mean(tot_rew_10[0:100000]))
	print('ave_rew100_f100k :', np.mean(data[ind][0:100000]))
	print('ave_rew1000_f100k :', np.mean(tot_rew_1000[0:100000]))
		

		#diff = []
		#for i in range(len(rews)):
		#	diff.append(data[4][i]-rewards[i])


		#print('diff: ',np.sum(diff))
		
	fig, ax = plt.subplots()
	ax.plot(data[ind])
	ax.set_ylim([0,25])
	ax.set_title(file_dir2)
	plt.plot()
	fig.savefig(file_dir2 + ".png")
		#plt.show()


if __name__ == '__main__':
    main()
