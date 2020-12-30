#system
import numpy as np 
import glob

#created
from data.processing import *

root = '../../'			#if incomaptible to other then reference with os , os.getcwd and its methods. 
# path = './mouthed' 	#later, we will change to data dir. which is only for data 
sampling_frequency = 250	#	Hz

#load data
def dataset(**kwargs):
	data_dir = path
	filepaths = glob.glob(path + "/**/*.txt", recursive= True)
	# filepaths = filepaths[:3]
	return [processing.process(1, [file], labels = file.split('/')[-2],**kwargs) for file in filepaths]

# total_data = dataset(channels = range(0, 8), surrounding=210)

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


#implement this if there is any missing or deleted pickle file which is unknown.
checkresults = check_pickle(root+'dataset/US/')
# if checkresults != []:
# 	for checkresult in checkresults :
# 		extractSegInPickle(checkresult, verbose = True)

#implement this with appropripate REGEX to get/update the pickle file you desired.
# extractSegInPickle(root+'dataset/US/mentally/', channels = range(0, 8), surrounding=210)

#use REGEX to import the pickle data and label
# data, label = loadSegOfPickle(root+'dataset/US/me*/')

# test = []
# for i in range(8):
# 	test.append(filter_data(data[0][:,i]))

# import matplotlib.pyplot as plt 
# plt.plot(test)
# plt.show()

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


#zero padding 
# maximum_length = max(list(map(len, data)))
# print("the maximum_length is : ", maximum_length)
# for i in range(len(data)):
# 	pad_width = maximum_length - len(data[i])
# 	pad_before_n, pad_after_n = (int(np.floor(pad_width/2)) , int(np.ceil(pad_width/2)))
# 	data[i] = np.pad(data[i], ((pad_before_n, pad_after_n), (0, 0)) , constant_values = (0.0, 0.0))

# #skipping feature extraction (for now)
# data = np.array(data)
# label = np.array(label)

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