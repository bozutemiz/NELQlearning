import train_coupled

def plotting_analysis_fn():
	n = 10
	for i in range(2, n):
		print(str(i+1)+'th run started...')
		train_coupled.main(i+1)
		print(str(i+1)+'th run complete!')
	print(str(n)+' runs finished!')

plotting_analysis_fn()

