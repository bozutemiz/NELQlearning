import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from six.moves import cPickle
from collections import deque
import numpy as np

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
	running_mean_rew_qnet = []
	running_mean_rew_qvl_offpol = []
	

	for j in range(5):
		file_dir2 = "QVloss on policy run " + str(j+1)
		file_dir1 = "outputs/"
		file_dir = file_dir1 + file_dir2 + "/"
		file_name = "train_stats.pkl"
		ind = 4

		data = cPickle.load( open( file_dir+file_name, "rb" ) )
		rews = get_rewards_per_step(data[ind])
		
		#all_rewards = deque(maxlen=100)
		#rewards = []

		#for i in range(len(rews)):
		#	all_rewards.append(rews[i])
		#	asd = np.sum(all_rewards)
		#	rewards.append(asd)

		print('tot_rew :', np.sum(rews))
		#tot_rew_10 = tot_reward_n_step(rews, n=10)
		#tot_rew_1000 = tot_reward_n_step(rews, n=1000)
		#print('max_rew10 :', np.max(tot_rew_10))
		print('max_rew100 :', np.max(data[ind]))
		#print('max_rew1000 :', np.max(tot_rew_1000))
		print('ave_rew :', np.mean(rews))
		#print('ave_rew10 :', np.mean(tot_rew_10))
		print('ave_rew100 :', np.mean(data[ind]))
		#print('ave_rew1000 :', np.mean(tot_rew_1000))
		print('ave_rew_f100k :', np.mean(rews[0:100000]))
		#print('ave_rew10_f100k :', np.mean(tot_rew_10[0:100000]))
		print('ave_rew100_f100k :', np.mean(data[ind][0:100000]))
		#print('ave_rew1000_f100k :', np.mean(tot_rew_1000[0:100000]))
		
		running_mean_rew_qvl_onpol.append(calculate_mean_rew_per_time(rews))
		#print(running_mean_rew)

		#diff = []
		#for i in range(len(rews)):
		#	diff.append(data[4][i]-rewards[i])


		#print('diff: ',np.sum(diff))
		
		#fig, ax = plt.subplots()
		

		#ax.set_ylim([0,0.06])
		#ax.set_title(file_dir2)
		#plt.plot()
		#fig.savefig(file_dir2 + ".png")
		#plt.show()

	for j in range(5):

		file_dir2 = "Qnet run " + str(j+1)
		file_dir1 = "outputs/"
		file_dir = file_dir1 + file_dir2 + "/"
		file_name = "train_stats.pkl"
		ind = 1

		data = cPickle.load( open( file_dir+file_name, "rb" ) )
		rews = get_rewards_per_step(data[ind])
		
		#all_rewards = deque(maxlen=100)
		#rewards = []

		#for i in range(len(rews)):
		#	all_rewards.append(rews[i])
		#	asd = np.sum(all_rewards)
		#	rewards.append(asd)

		print('tot_rew :', np.sum(rews))
		#tot_rew_10 = tot_reward_n_step(rews, n=10)
		#tot_rew_1000 = tot_reward_n_step(rews, n=1000)
		#print('max_rew10 :', np.max(tot_rew_10))
		print('max_rew100 :', np.max(data[ind]))
		#print('max_rew1000 :', np.max(tot_rew_1000))
		print('ave_rew :', np.mean(rews))
		#print('ave_rew10 :', np.mean(tot_rew_10))
		print('ave_rew100 :', np.mean(data[ind]))
		#print('ave_rew1000 :', np.mean(tot_rew_1000))
		print('ave_rew_f100k :', np.mean(rews[0:100000]))
		#print('ave_rew10_f100k :', np.mean(tot_rew_10[0:100000]))
		print('ave_rew100_f100k :', np.mean(data[ind][0:100000]))
		#print('ave_rew1000_f100k :', np.mean(tot_rew_1000[0:100000]))
		
		running_mean_rew_qnet.append(calculate_mean_rew_per_time(rews))

	for j in range(5):
		file_dir2 = "QVloss off policy run " + str(j+1)
		file_dir1 = "outputs/"
		file_dir = file_dir1 + file_dir2 + "/"
		file_name = "train_stats.pkl"
		ind = 4

		data = cPickle.load( open( file_dir+file_name, "rb" ) )
		rews = get_rewards_per_step(data[ind])
		
		#all_rewards = deque(maxlen=100)
		#rewards = []

		#for i in range(len(rews)):
		#	all_rewards.append(rews[i])
		#	asd = np.sum(all_rewards)
		#	rewards.append(asd)

		print('tot_rew :', np.sum(rews))
		#tot_rew_10 = tot_reward_n_step(rews, n=10)
		#tot_rew_1000 = tot_reward_n_step(rews, n=1000)
		#print('max_rew10 :', np.max(tot_rew_10))
		print('max_rew100 :', np.max(data[ind]))
		#print('max_rew1000 :', np.max(tot_rew_1000))
		print('ave_rew :', np.mean(rews))
		#print('ave_rew10 :', np.mean(tot_rew_10))
		print('ave_rew100 :', np.mean(data[ind]))
		#print('ave_rew1000 :', np.mean(tot_rew_1000))
		print('ave_rew_f100k :', np.mean(rews[0:100000]))
		#print('ave_rew10_f100k :', np.mean(tot_rew_10[0:100000]))
		print('ave_rew100_f100k :', np.mean(data[ind][0:100000]))
		#print('ave_rew1000_f100k :', np.mean(tot_rew_1000[0:100000]))
		
		running_mean_rew_qvl_offpol.append(calculate_mean_rew_per_time(rews))
		#print(running_mean_rew)

		#diff = []
		#for i in range(len(rews)):
		#	diff.append(data[4][i]-rewards[i])


		#print('diff: ',np.sum(diff))
		
		#fig, ax = plt.subplots()
		

		#ax.set_ylim([0,0.06])
		#ax.set_title(file_dir2)
		#plt.plot()
		#fig.savefig(file_dir2 + ".png")
		#plt.show()


	#print(len(running_mean_rew_qvl_onpol))
	#print(np.size(np.sum(running_mean_rew_qvl_onpol,axis=0)/5))

	#mean_qvl_onpol = np.sum(running_mean_rew_qvl_onpol,axis=0)/len(running_mean_rew_qvl_onpol)
	mean_qvl_onpol = np.mean(running_mean_rew_qvl_onpol,axis=0)
	std_qvl_onpol = np.std(running_mean_rew_qvl_onpol,axis=0)
	#print(std_qvl_onpol)
	mean_qnet = np.mean(running_mean_rew_qnet,axis=0)
	std_qnet = np.std(running_mean_rew_qnet,axis=0)

	mean_qvl_offpol = np.mean(running_mean_rew_qvl_offpol,axis=0)
	std_qvl_offpol = np.std(running_mean_rew_qvl_offpol,axis=0)
	
	
	#print(qqq)

	x = range(1,len(mean_qvl_onpol)+1)
	print(len(x))
	ax.plot(mean_qvl_onpol,linewidth=5, color='blue')
	ax.fill_between(x, mean_qvl_onpol - std_qvl_onpol, mean_qvl_onpol + std_qvl_onpol,color='blue',alpha=0.4)
	ax.plot(mean_qnet,linewidth=5, color='red')
	ax.fill_between(x, mean_qnet - std_qnet, mean_qnet + std_qnet,color='red',alpha=0.4)
	ax.plot(mean_qvl_offpol,linewidth=5, color='green')
	ax.fill_between(x, mean_qvl_offpol - std_qvl_offpol, mean_qvl_offpol + std_qvl_offpol,color='green',alpha=0.4)
		

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



	ax.set_ylim([0,0.06])
	ax.set_title(file_dir2)
	plt.plot()
	#fig.savefig(file_dir2 + ".png")
	plt.show()


if __name__ == '__main__':
    main()
