#system
import numpy as np 
import glob

#created
import processing

path = '../../dataset/RL/mentally' 	#later, we will change to data dir. which is only for data 
# path = '/home/rimesh/Files/major/codes/AlterEgo-master/Core'
sampling_frequency = 250	#	Hz

#load data
def dataset(**kwargs):
	data_dir = path
	filepaths = glob.glob(path + "/**/*.txt", recursive= True)
	# filepaths = filepaths[:3]
	return [processing.process(1, [file], labels = file.split('/')[-2],**kwargs) for file in filepaths]

total_data = dataset(channels = range(0, 8), surrounding=210)


from scipy import signal
def filter_data(data):
	#this follows the arnav process

	#signal processing
	#	- bpf, notch filter (50 Hz)
	
	#applying high pass filter - 0.5, used to remove frequencies lower than 0.5Hz
	filter_order = 1
	# critical_frequencies = [15, 50] #in Hz
	critical_frequency = 0.5 	# in Hz
	FILTER = 'highpass'				#'bandpass'
	output = 'sos'
	#design butterworth bandpass filter
	sos = signal.butter(filter_order, critical_frequency, FILTER, fs = sampling_frequency, output= output)
	filtered = signal.sosfilt(sos, data)

	#normalize -(normalizing to a mean amplitude of zero (still need to cross check this))
	data = data - np.mean(data, axis = 0)

	#3rd order notch (butterworth// need to implement still) - power line noise , 50 Hz and its harmonics.
	#applying notch filter
	notch_times = 3
	notch_frequency = 50 	#Hz
	quality_factor = 30 	# 			-- no reason just copied.
	
	#design notch filter 
	# b, a = signal.iirnotch(notch_frequency, quality_factor, sampling_frequency)
	# filtered = signal.lfilter(b, a, filtered)
	
	freqs = list(map(int , list(map(round, np.arange(1, sampling_frequency/(2. * notch_frequency))* notch_frequency ))  ))
	for _ in range(notch_times):
		for f in reversed(freqs):
			b, a = signal.iirnotch(f, quality_factor, sampling_frequency)
			filtered = signal.lfilter(b, a, filtered)


	#TODO: removing heartbeat artifacts...
	#applying ricker
	# widths = np.arange(1, 50)
	# cwtmatr = signal.cwt(data,signal.ricker, widths)
	# data = data - cwtmatr n


	#applying bandpass filter, 0.5 - 8 Hz
	filter_order = 1
	# critical_frequencies = [15, 50] #in Hz
	critical_frequencies =[ 0.5, 8] 	# in Hz
	FILTER = 'bandpass'				#'bandpass'
	output = 'sos'

	#design butterworth bandpass filter
	sos = signal.butter(filter_order, critical_frequencies, FILTER, fs = sampling_frequency, output= output)
	filtered = signal.sosfilt(sos, data)
			

	return filtered

data = []
label = []
for i in range(len(total_data)):
	for j in range(len(total_data[i])):				#choosing the file //as per the index it was necessary
		for k in range(len(total_data[i][j])):		#chossing the data block in the file
			for l in range(len(total_data[i][j][k][0][0,:])):
				total_data[i][j][k][0][:,l] = filter_data(total_data[i][j][k][0][:,l])	#filters all the channel data //please recheck it...
			data.append(total_data[i][j][k][0])		#recording the data 
			label.append(total_data[i][j][k][1])	#recording the label
num_label = len(list(set(label)))

'''
#data[data_number][sample_number, Channel_number]
#label[data_number]

data - variable to access the sampled data.
label - label of the respective recorded data.

data_number - sum of all the recorded files, a single file contains 10-20(nearly some might contain greater than 20(absent minded during recording.)) instance of records.
`				thus it reference to the total number of data available.
sample_number - number of samples recorded <rows> (variable in length due to differene in recording time)
Channel_number - number of channels i.e. 8 <column>
'''

#visulaize the signal
import matplotlib.pyplot as plt 
plt.plot(data[0])
plt.show()

#applying ricker - on test
# widths = np.arange(1, 50)
# cwtmatr = signal.cwt(data[0][:,0],signal.ricker, widths)
# plt.imshow(cwtmatr)
# plt.show()

#zero padding 
maximum_length = max(list(map(len, data)))
print("the maximum_length is : ", maximum_length)
for i in range(len(data)):
	pad_width = maximum_length - len(data[i])
	pad_before_n, pad_after_n = (int(np.floor(pad_width/2)) , int(np.ceil(pad_width/2)))
	data[i] = np.pad(data[i], ((pad_before_n, pad_after_n), (0, 0)) , constant_values = (0.0, 0.0))

#skipping feature extraction (for now)
data = np.array(data)
label = np.array(label)

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
label = labelencoder_y.fit_transform(label)

from sklearn.model_selection import StratifiedShuffleSplit
def train_test_split(X, Y, verbose = False):
	split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.1, random_state = 42)
	train_id, test_id = next(split.split(X, Y))
	if verbose:
		print("train set shape: ", X[train_id].shape)
		print("test set shape: 	", X[test_id].shape)
	return 	X[train_id], Y[train_id], X[test_id], Y[test_id]

X_train, Y_train, X_test, Y_test = train_test_split(data, label)

#training model
#1D-cnn
import tensorflow as tf 
from tensorflow import keras
def CNN_Classifier(X_train, Y_train, X_test, Y_test):
	Y_train = tf.keras.utils.to_categorical(Y_train, num_classes = num_label)
	Y_test = tf.keras.utils.to_categorical(Y_test, num_classes = num_label)
	CNN_model = keras.Sequential()
	CNN_model.add(keras.layers.Conv1D(100, kernel_size = 3, input_shape = X_train.shape[1:], activation = "relu"))
	CNN_model.add(keras.layers.MaxPool1D(pool_size=2))
	CNN_model.add(keras.layers.Conv1D(100,kernel_size=3,activation="relu"))
	CNN_model.add(keras.layers.MaxPool1D(pool_size=2))
	CNN_model.add(keras.layers.Flatten())
	CNN_model.add(keras.layers.Dense(100,activation="relu"))
	CNN_model.add(keras.layers.Dense(num_label,activation="softmax"))

	opt = keras.optimizers.Adam(lr = 0.00001)

	CNN_model.compile(optimizer = opt, loss = keras.losses.categorical_crossentropy, metrics=['accuracy'])
	print(CNN_model.summary())

	history = CNN_model.fit(X_train, Y_train, epoches = 50, batch_size = 50, verbose = 1)

	CNN_prediction = CNN_model.predict_classes(X_train)
	max_val_acc = max(history.history['accuracy'])
	print(max_val_acc)

	return CNN_model.evaluate(X_test, Y_test)[1]


print("CNN :",CNN_Classifier(X_train, Y_train, X_test, Y_test))

# TODO 
#feature extraction
#use the model made
#train and test

#'''


'''
source :
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html

'''