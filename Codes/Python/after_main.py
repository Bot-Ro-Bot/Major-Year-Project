import tensorflow as tf 
from tensorflow import keras
from data.processing import check_file

#method that constantly checks the folder containing the file of Openbci.
def ch():
	checkresults = []
	while(checkresults == []):
		checkresults = check_file('/home/*/Documents/OpenBCI_GUI/Recordings/', file_extension = '*.txt')
		print(checkresults)

#call the model 
model = keras.models.load_model('model')

#checking for the file from openbci.
ch()

#TODO : extract data points from the file
#TODO : pass the extracted points to the trained model
#TODO : predict and display the articulation.
