import train_coupled

def plotting_analysis_fn():
	n = 5
	for i in range(n):
		print(str(n+i+1)+'th run started...')
		train_coupled.main(n+i+1)
		print(str(n+i+1)+'th run complete!')
	print(str(n+n)+' runs finished!')

plotting_analysis_fn()

