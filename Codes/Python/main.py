#system
import numpy as np 
import glob
import matplotlib.pyplot as plt 
import pickle
import pandas as pd

from scipy import signal

#created
from data.processing import *
from data.emgprocessings import *

root = '../../'			#if incomaptible to other then reference with os , os.getcwd and its methods. 
# path = './mouthed' 	#later, we will change to data dir. which is only for data 
sampling_frequency = 250	#	Hz

# DATA EXTRACTION

#implement this to check if there is any missing or deleted pickle file which is unknown.
checkresults = check_pickle(root+'dataset/[R|S|U]*/')

# uncomment below to get(or update) the missing(or deleted) pickel files 

#implement this with appropripate  unix glob patterns to get(or update) the pickle file from checkresults.
# method 1 : extract and store data only from the missing obtain from the above checkresults. 
# if checkresults != []:
# 	for checkresult in checkresults :
# 		extractSegInPickle(checkresult, verbose = True)

# method 2 : implement this with appropripate  unix glob patterns to get(or update) the pickle file you desired or proceed to extract and store from all data.
# if checkresults != []:
	# extractSegInPickle(root+'dataset/*/*/', channels = range(0, 8), surrounding=210)

#use  unix glob patterns to import the pickle data and label
data, label = loadSegOfPickle(root+'dataset/[R|S|U]*/me*/')

'''
#data[data_number][sample_number, Channel_number]
#label[data_number]

data - variable to access the sampled data.
label - label of the respective recorded data.

data_number - sum of all the recorded files, a single file contains 10-20(nearly some might contain greater than 20(absent minded during recording.)) instance of records.
				thus it reference to the total number of data available.
sample_number - number of samples recorded <rows> (variable in length due to differene in recording time)
Channel_number - number of channels i.e. 8 <column>
'''

#plot for the data distribution.
labs = list(set(label))
counts = [ label.count(i) for i in labs ]
x_pos = np.arange(len(labs))
y_pos = np.arange(len(labs))#np.arange(0,max(counts), 5)

print(labs)
print(counts)

plt.bar(y_pos, counts, align = 'center', alpha = 0.5)
plt.xticks(x_pos, labs)
plt.ylabel('counts')
plt.title('data distribution')
plt.show()
# end of distribution plot

# END OF DATA EXTRACTION

def drop_data(X,Y,MIN=100,MAX=900):
    index = []
    j = 0
    for i in range(len(X)):
        if(len(X[i][:,0])<MIN or len(X[i][:,0])>MAX):
            j = j + 1
            print(j, i , len(X[i][:,0]))
            continue
        index.append(i)
    return [X[n] for n in index],[Y[n] for n in index]

data, label = drop_data(data, label)

data = np.array(data)
label = np.array(label)
#contains the sets of labels being used.
label_sets = list(sorted(set(label)))
#2-d array containing list of start and end index of the label_sets in the label
#rows for the label_sets.
#columns : [name, start_index, end_index]
label_start_end_indicies = [ [label_set, np.where(label == label_set)[0][0], np.where(label == label_set)[0][-1]] for label_set in label_sets  ]

# filtering the data
for i in range(data.shape[0]):
	for j in range(data[i].shape[-1]):
		data[i][:,j] = filter_data(data[i][:,j])
data = np.array(data)

# #aile plot ko lai
# for i in range(10):
# 	for j in range(data[i].shape[-1]):
# 		data[i][:,j] = filter_data(data[i][:,j])
# test = np.array(data)


# plt.subplot(2, 1, 1)
# plt.plot(test[0][:, 0])
# plt.title("Raw Signal")
# plt.ylabel("Amplitude")
# gmf = get_emg_features(test[0])
# plt.subplot(2, 1, 2)
# plt.plot(gmf[:, 0])
# plt.title("Double Nine Point Average")
# plt.xlabel("Samples")
# plt.ylabel("Amplitude")
# plt.show()



# zero padding 
temp = []
maximum_length = 900 	#max(list(map(len, data)))
print("the maximum_length is : ", maximum_length)
for i in range(len(data)):
	pad_width = maximum_length - len(data[i])
	pad_before_n, pad_after_n = (int(np.floor(pad_width/2)) , int(np.ceil(pad_width/2)))
	# data[i] = np.pad(data[i], ((pad_before_n, pad_after_n), (0, 0)) , constant_values = (0.0, 0.0))
	temp.append(np.pad(data[i], ((pad_before_n, pad_after_n), (0, 0)) , constant_values = (0.0, 0.0)))
data = np.array(temp)

# Feature extraction
temp = []
for i in range(len(data)):
	temp.append(get_emg_features(data[i]))

a = np.array(temp)

di = { 'data' : a , 'label' : label}
print(di.keys)

pickle.dump(di , open('ALL_feature_temporal_me.pickle', 'wb'))
'''
# b = np.zeros(a.shape[0], a.shape[1], a.shape[2]/8, 8)
# for i in range(a.shape[0]):
# 	for j in range( a.shape[2]/8):
# 		for k in range(8):
# 			b[i, :, , k] = a[i, :, j*k]
# graphit(data[0])

# TRAINING MODEL
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
label = labelencoder_y.fit_transform(label)

num_label = len(set(label))
print('[+] num_label = ', num_label)

from sklearn.model_selection import StratifiedShuffleSplit
def train_test_split(X, Y, verbose = False):
	split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.1, random_state = 42)
	train_id, test_id = next(split.split(X, Y))
	if verbose:
		print("train set shape: ", X[train_id].shape)
		print("test set shape: 	", X[test_id].shape)
	return 	X[train_id], Y[train_id], X[test_id], Y[test_id]

X_train, Y_train, X_test, Y_test = train_test_split(data, label)

Y_train_unencoded = Y_train
Y_test_unencoded = Y_test

print("X_train.shape", X_train.shape)
print("X_train.shape[]", X_train[0].shape)

from sklearn.metrics import confusion_matrix
def confusion_matrix_plt(y_value , pred_value, labels):
	print(y_value)
	print(pred_value)
	plt.matshow(confusion_matrix(y_value, pred_value,  normalize = 'true'), cmap = 'gray')
	plt.xticks(range(0, len(labels)), labels)
	plt.yticks(range(0, len(labels)), labels)
	plt.show()

#training model
#1D-cnn
import tensorflow as tf 
from tensorflow import keras
def CNN_Classifier(X_train, Y_train, X_test, Y_test):
	Y_train = tf.keras.utils.to_categorical(Y_train, num_classes = num_label)
	Y_test = tf.keras.utils.to_categorical(Y_test, num_classes = num_label)
	CNN_model = keras.Sequential()
	CNN_model.add(keras.layers.Conv1D(100, kernel_size = 12, input_shape = X_train.shape[1:], activation = "relu"))
	CNN_model.add(keras.layers.MaxPool1D(pool_size=2))
	CNN_model.add(keras.layers.Conv1D(100,kernel_size=6,activation="relu"))
	CNN_model.add(keras.layers.MaxPool1D(pool_size=2))
	CNN_model.add(keras.layers.Flatten())
	CNN_model.add(keras.layers.Dense(100,activation="relu"))
	CNN_model.add(keras.layers.Dense(num_label,activation="softmax"))

	opt = keras.optimizers.Adam(lr = 0.0001)

	CNN_model.compile(optimizer = opt, loss = keras.losses.categorical_crossentropy, metrics=['accuracy'])
	print(CNN_model.summary())

	history = CNN_model.fit(X_train, Y_train, epochs = 5, batch_size = 50, validation_data =(X_test, Y_test) ,verbose = 1)

	CNN_prediction = CNN_model.predict_classes(X_test)
	print(CNN_prediction)
	confusion_matrix_plt(Y_test_unencoded, CNN_prediction, label_sets)
	# save model
	CNN_model.save('model_CNN_SR_Mo_rawadsf')

	# max_val_acc = max(history.history['accuracy'])
	# print(max_val_acc) #['loss', 'acc']

	print(list(history.history.keys()))
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model acc')
	plt.legend([ 'training_acc', 'validation_acc'])
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.show()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.legend([ 'training_loss', 'validation_loss'])
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.show()

	return CNN_model.evaluate(X_test, Y_test)[1]

print("CNN :",CNN_Classifier(X_train, Y_train, X_test, Y_test))

def CNN_2d_Classifier(X_train, Y_train, X_test, Y_test):
	Y_train = tf.keras.utils.to_categorical(Y_train, num_classes = num_label)
	Y_test = tf.keras.utils.to_categorical(Y_test, num_classes = num_label)
	CNN_model = keras.Sequential()
	CNN_model.add(keras.layers.Conv2D(100, kernel_size = (12, 12), input_shape = X_train.shape[1:], activation = "relu"))
	CNN_model.add(keras.layers.MaxPool2D(pool_size=(2, 2) ))
	CNN_model.add(keras.layers.Conv2D(100,kernel_size=(6, 6 ),activation="relu"))
	CNN_model.add(keras.layers.MaxPool1D(pool_size=(2,2) ))
	CNN_model.add(keras.layers.Flatten())
	CNN_model.add(keras.layers.Dense(100,activation="relu"))
	CNN_model.add(keras.layers.Dense(num_label,activation="softmax"))

	opt = keras.optimizers.Adam(lr = 0.0001)

	CNN_model.compile(optimizer = opt, loss = keras.losses.categorical_crossentropy, metrics=['accuracy'])
	print(CNN_model.summary())

	history = CNN_model.fit(X_train, Y_train, epochs = 15, batch_size = 50, validation_data =(X_test, Y_test) ,verbose = 1)

	CNN_prediction = CNN_model.predict_classes(X_train)
	
	confusion_matrix_plt(Y_test, CNN_prediction, label_sets)

	#save model
	CNN_model.save('hh')

	# max_val_acc = max(history.history['accuracy'])
	# print(max_val_acc) #['loss', 'acc']

	print(list(history.history.keys()))
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model acc')
	plt.legend([ 'training_acc','validation_acc'])
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.show()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.legend(['training_loss','validation_loss'])
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.show()

	return CNN_model.evaluate(X_test, Y_test)[1]

# TODO 
#feature extraction
#use the model made
#train and test

#terminal test
# model = keras.models.load_model('model')
# checkresults = check_pickle('~/Document/OpenBCI_GUI/Recordings', file_extension = '*.txt')

#'''

'''
source :
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html
https://www.codegrepper.com/code-examples/python/moving+average+filter+in+python
https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
https://www.tensorflow.org/guide/keras/save_and_serialize
'''