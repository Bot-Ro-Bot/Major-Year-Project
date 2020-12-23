#system
import numpy as np 
import glob

#created
import processing

path = '.' 	#later, we will change to data dir. which is only for data 
sampling_frequency = 250	#	Hz

#load data
def dataset(**kwargs):
	data_dir = path
	filepaths = glob.glob(path + "/**/*.txt", recursive= True)
	filepaths = filepaths[:3]
	# kwargs = 
	return [processing.process(1, [file], labels = file.split('/')[-2],**kwargs) for file in filepaths]

total_data = dataset(channels = range(0, 8), surrounding=210)


def filtered_data(data):
	#signal processing
	#	- bpf, notch filter (50 Hz)
	from scipy import signal
	
	#applying bandpass filter
	filter_order = 10
	critical_frequencies = [15, 50] #in Hz
	FILTER = 'bandpass'
	output = 'sos'

	#design butterworth bandpass filter
	sos = signal.butter(filter_order, critical_frequencies, FILTER, fs = sampling_frequency, output= output)
	filtered = signal.sosfilt(sos, data)

	#applying notch filter
	notch_frequency = 50 	#Hz
	quality_factor = 30 	# 			-- no reason just copied.
	
	#design notch filter 
	b, a = signal.iirnotch(notch_frequency, quality_factor, sampling_frequency)
	filtered = signal.lfilter(b, a, filtered)
	
	return filtered

data = []
label = []
for i in range(len(total_data)):					#gives the number of file
	for j in range(len(total_data[i])):				#choosing the file //as per the index it was necessary
		for k in range(len(total_data[i][j])):		#chossing the data block in the file
			data.append(total_data[i][j][k][0])		#recording the data 
			label.append(total_data[i][j][k][1])	#recording the label

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

#visulaize the signals
import matplotlib.pyplot as plt 
plt.plot(data[0][:,0])
plt.show()



from scipy import signal

#applying bandpass filter
filter_order = 10
critical_frequencies = [15, 50] #in Hz
FILTER = 'bandpass'
output = 'sos'

#design butterworth bandpass filter
sos = signal.butter(filter_order, critical_frequencies, FILTER, fs = sampling_frequency, output= output)
filtered = signal.sosfilt(sos, data[0][:,0])
plt.plot(filtered)
plt.show()
#applying notch filter
notch_frequency = 50 	#Hz
quality_factor = 30 	# 			-- no reason just copied.

#design notch filter 
b, a = signal.iirnotch(notch_frequency, quality_factor, sampling_frequency)
filtered = signal.lfilter(b, a, filtered)
plt.plot(filtered)
plt.show()

# TODO 
#feature extraction
#use the model made
#train and test


'''
source :
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html

'''