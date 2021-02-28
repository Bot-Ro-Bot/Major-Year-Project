
#run this script to after saving the trained model.
import tensorflow as tf 
import numpy as np
from tensorflow import keras
from data.processing import check_file, extractSeg_from_file
from data.emgprocessings import filter_data

#method that constantly checks the folder containing the file of Openbci in /home/<user>/Documents/OpenBCI_GUI/Recordings/<file.txt>
def get_recording_file():
	checkresults = []
	while(checkresults == []):		#constantly observe the folder for file
		checkresults = check_file('/home/*/Documents/OpenBCI_GUI/Recordings/', file_extension = '*.txt')
		print(checkresults)
	return checkresults

#call the model 
model = keras.models.load_model('model')

#checking for the file from openbci.
rec_file= get_recording_file()
print(rec_file)
# extract data points from the file
instance_data, instance_label =  extractSeg_from_file(rec_file[0])

instance_data = np.array(instance_data)
instance_label= np.array(instance_label)
print(instance_data.shape)

#filtering the data.
for i in range(instance_data.shape[0]):
	for j in range(instance_data[i].shape[-1]):
		instance_data[i][:,j] = filter_data(instance_data[i][:,j])

#length normalization 
# zero padding (and equating the sample size as the trained data size)
temp = []
dataset_data_max_len = 935				#maximum data size in the trained data.(TODO : automate this later)
maximum_length = dataset_data_max_len	#max(list(map(len, instance_data)))
print("the maximum_length is : ", maximum_length)
for i in range(len(instance_data)):
	if len(instance_data[i]) <= dataset_data_max_len :		#zero pad the data if it is less than the trained. 
		pad_width = maximum_length - len(instance_data[i])
		pad_before_n, pad_after_n = (int(np.floor(pad_width/2)) , int(np.ceil(pad_width/2)))
		# data[i] = np.pad(data[i], ((pad_before_n, pad_after_n), (0, 0)) , constant_values = (0.0, 0.0))
		temp.append(np.pad(instance_data[i], ((pad_before_n, pad_after_n), (0, 0)) , constant_values = (0.0, 0.0)))
	else : 	#take the first to the dataset_data_max_len of the uttered data.. (need further more methods...)
		trim_width = len(instance_data[i]) - maximum_length 
		temp.append(instance_data[i][ int(np.floor(trim_width/2)) :  -1 * int(np.ceil(trim_width/2))])
instance_data = np.array(temp)

# Feature extraction
temp = []
for i in range(len(instance_data)):
	temp.append(get_emg_features(instance_data[i]))
data_feat = np.array(temp)

X= data_feat 	#just for naming

#scaling data
from sklearn.preprocesing import StandardScaler
scalar = StandardScaler()
X = (scalar.fit_transform(X.reshape(X.shape[0], -1), Y)).reshape(X.shape[0], X.shape[1], -1)

dataset_labels = ['add', 'call', 'go', 'later', 'left',
					'reply', 'right', 'stop', 'subtract', 'you']

#pass the extracted points to the trained model
model_predictions = model.predict_classes(X)
#predict and display the articulation.
print([dataset_labels[model_prediction] for model_prediction in model_predictions])
print(len(model_predictions))
#'''