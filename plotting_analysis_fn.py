import train_coupled

def plotting_analysis_fn():
	n = 10
	opt = [100, 10000]
	for target_update_frequency in opt:
		# target_update_frequency = (10**k) * 100
		print('*****************************************')
		print('-----------------------------------------')
		print('*****************************************')
		print('-----------------------------------------')
		print('Starting 10 runs for n='+str(target_update_frequency)) 
		for i in range(n):
			print(str(i+1)+'th run started...')
			train_coupled.main(target_update_frequency, i+1)
			print(str(i+1)+'th run complete!')
		print('Finished 10 runs for n='+str(target_update_frequency))
		print('*****************************************')
		print('-----------------------------------------')
		print('*****************************************')
		print('-----------------------------------------')

plotting_analysis_fn()
