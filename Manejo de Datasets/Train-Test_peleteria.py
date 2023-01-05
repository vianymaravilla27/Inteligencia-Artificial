import pandas as pd
from sklearn.model_selection import train_test_split
	
if __name__=='__main__':
	
	#Lee el corpus original del archivo de entrada y lo pasa a un DataFrame
	df = pd.read_csv('peleteria.csv', sep=',', engine='python')
	x = df.drop('y',axis=1).values    					#corpus sin etiquetas 
	y = df['y'].values 									#etiquetas
	
	print('\n df', df)	
	print('\n Corpus')
	print('\n', *x)
	print ('----------------------')
	print('\n Etiquetas')
	print('\n', *y)
	print ('----------------------')	            
	
	x_e, x_p, y_e, y_p = train_test_split(x, y, test_size=0.5, shuffle = False)	
	
	print('\n Conjunto de entrenamiento')		
	print ('\n x_e ', *x_e)
	print ('\n y_e ', y_e)
	print ('----------------------')
	
	
	print('\n Conjunto de prueba')	
	print ('\n x_p', *x_p)
	print ('\n y_p', y_p)
	print ('----------------------')

	
	
	
	
	

	
