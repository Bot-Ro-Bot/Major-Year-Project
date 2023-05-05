import numpy as np 
import glob

def process(num_classes,filepaths, labels = None, include_surrounding= True,
			sample_rate = 250, channels= range(0, 8), surrounding = 210):
	print("the lable is : ", labels)
	def get_sequence_groups(filepath):
		print("processing filepath"+str(filepath))
		f = open(filepath, 'r')
		contents = map(lambda x : x.strip(), f.readlines())
		#the file starts with '%' and some instruction before data and removing these data 
		frames_original = list(filter(lambda x : x and x[0] != '%', contents))[1:]
		#the data row contains channels info digital trigger and accelerometer info separated by comma
		frames_original = 	list(map(lambda s : list(map( lambda ss: ss.strip(), s.split(','))), frames_original))
		
		# (8 channels) + digital triggers
		# the digital trigger is in a[16], used to indicate the utterance // while testing for eric and arnav file change it to 12
		frames = list(map(lambda a: list(map(float, a[1:9])) + [float(a[16])] , frames_original))
		# last column consisted of time-stamp fo the recording(best to check file and file formatting of OPENBCI)
		time_stamps = list(map(lambda a: a[-1], frames_original))
		frames = np.array(frames)

		speaking = False 
		start_index = 0
		num = -1
		sequence_groups = [[] for _ in range(num_classes)]
		padding = surrounding * sample_rate//250 if include_surrounding else 0
		#separates the triggered data points...
		for i in range(len(frames)):
			if not speaking and bool(frames[i][-1]):
				# print('start speaking')
				speaking = True
				start_index = i 
			if speaking and not bool(frames[i][-1]):
				# print('stop speaking')
				speaking = False
				if np.all(map(bool, frames[i-200:i, -1])):
					num+= 1 
					# print(num)
					if list(frames[start_index- padding: i + padding, channels]) != [] :
						sequence_groups[num % num_classes].append([frames[start_index- padding: i + padding, channels], labels])
					else :
						print("[*] from get_sequence_groups : found NULL, ignoring ") 


		if bool(frames[-1][-1]):
			sequence_groups[num % num_classes].append(frames[start_index- padding:, channels])

		sequence_groups = np.array(sequence_groups)

		print("sequence_groups : ", sequence_groups.shape)

		return sequence_groups

	def join_sequence_groups(*g):
		groups = [[] for _ in range(len(g[0]))]
		for i in range(len(g)):
			for j in range(len(g[i])):
				groups[j] += list(g[i][j])
	
		return np.array(groups)

	sequence_groups = list(map(get_sequence_groups, filepaths))
	return join_sequence_groups(*sequence_groups)

#Major-Year-Project/dataset/<person>/<type>/<words>/files.txt
#Major-Year-Project/dataset/<person>/<type>.pickle

verbose = True

def check_pickle(path):
	'''
	#path must be  := root+ <dataset> + '<person>/'

	checks all the sub-directory for unextracted pickle file.
	returns the paths of the directory that does not have pickle file.
	'''
	pickle_to_extract = []

	dirs = glob.glob(path)
	for dir in dirs:
		if verbose:
			print("[*] Checking For pickle")
			print("[*] In Directory : ", dir)
		sub_dirs = glob.glob(dir + '*/')
		for sub_dir in sub_dirs:
			if verbose:
				print("[*]\tIn Sub Directory : ", sub_dir)
			picklefile = glob.glob(sub_dir+'*.pickle')
			# print(picklefile)
			if picklefile != []:
				if verbose:
					print("[+] Pickle File found.")
			else :
				if verbose:
					print('[-] Pickle File not found in : ',sub_dir)
				pickle_to_extract.append(sub_dir)

	return pickle_to_extract
	#returns list of ['../../dataset/<person>/<types>/']
	# types refer to : Mentally & Mouthed

import pickle
def extractSegInPickle(path, **kwargs):
	# path must be  := root + '<person>/<types>/'
	# types refer to : Mentally & Mouthed
	r_data = []
	r_label = []
	print("this is glob : ",glob.glob(path))
	for p in glob.glob(path):
		def dataset(**kwargs):
			#1st '*'' is for Words and 2nd '*' is the name of the file.
			filepaths = glob.glob(p+'*/*.txt', recursive = True)
			# filepaths = filepaths[:3]		#just to limit the output while testing.
			print(filepaths)
			return [process(1, [file], labels = file.split('/')[-2],**kwargs) for file in filepaths]

		total_data = dataset(**kwargs)

		picklefile_name = str(p).strip('[ ]').split('/') 
		picklefile_name = picklefile_name[-3]+'_'+picklefile_name[-2]+'.pickle'
		
		data = []
		label = []

		for i in range(len(total_data)):
			for j in range(len(total_data[i])):				#choosing the file //as per the index it was necessary
				for k in range(len(total_data[i][j])):		#chossing the data block in the file
					data.append(total_data[i][j][k][0])		#recording the data 
					label.append(total_data[i][j][k][1])	#recording the label
		
		with open(p+  'data_'+picklefile_name, 'wb') as f:
			if verbose:
				print('[*] Writing to file :', f)
			pickle.dump(data, f)

		with open( p + 'label_'+ picklefile_name, 'wb') as f:
			pickle.dump(label, f)

		r_data.append(data)
		r_label.append(label)
	return r_data, r_label

def loadSegOfPickle(path, **kwargs):
	# path must be  := root + '<person>/<types>/'
	temp_data = []
	temp_label = []
	for path in glob.glob(path):
		pickle_files  = glob.glob(path+'**/*.pickle', recursive= True)
		for pickle_file in pickle_files:
			with open(pickle_file, 'rb') as file:
				if 'data_' in pickle_file:
					temp_data.append(pickle.load(file))
					if verbose:
						print("[+] added data")
				elif 'label_' in pickle_file:
					temp_label.append(pickle.load(file))
					if verbose:
						print("[+] added label")

	data = []
	label = []
	for i in range(len(temp_data)):
		for j in range(len(temp_data[i])):
			data.append(temp_data[i][j])
			label.append(temp_label[i][j])

	return data, label


def check_file(path, file_extension = '*.txt', verbose = False):
	'''
	checks all the sub-directory for file with the provided extension.
	returns the paths of the directory that have file.
	'''
	found_files = []
	print('Finding file')
	dirs = glob.glob(path)
	for dir in dirs:
		if verbose:
			print("[*] Checking For "+file_extension )
			print("[*] In Directory : ", dir)
		sub_dirs = glob.glob(dir + '*/')
		for sub_dir in sub_dirs:
			if verbose:
				print("[*]\tIn Sub Directory : ", sub_dir)
			file = glob.glob(sub_dir+file_extension)
			# print(picklefile)
			if file != []:
				if verbose:
					print("[+] " +  file_extension+" File found.")
				found_files.append(sub_dir)
			else :
				if verbose:
					print('[-] ' + file_extension +' File not found in : '+ sub_dir)
	return found_files


# similar to extractSegInPickle
# things between extractSegInPickle and extractSeg_from_file could be implemented in a combined way 
def extractSeg_from_file(path, **kwargs):
	# TODO : need to add extensions but will do for now 
	r_data = []
	r_label = []
	print("this is glob : ",glob.glob(str(path)))
	for p in glob.glob(path):
		def dataset(**kwargs):
			#search all the files (with extension .txt ) in the pointed dir. #note it will not search any sub directories within
			filepaths = glob.glob(p+'*.txt', recursive = True)
			# filepaths = filepaths[:3]		#just to limit the output while testing.
			print(filepaths)
			return [process(1, [file], labels = file.split('/')[-2],**kwargs) for file in filepaths]

		total_data = dataset(**kwargs)

		data = []
		label = []
		#just to sort data out
		for i in range(len(total_data)):
			for j in range(len(total_data[i])):				#choosing the file //as per the index it was necessary
				for k in range(len(total_data[i][j])):		#chossing the data block in the file
					data.append(total_data[i][j][k][0])		#recording the data 
					label.append(total_data[i][j][k][1])	#recording the label
		
		r_data.append(data)
		r_label.append(label)

	#just to sort data out
	data = []
	label = []
	for i in range(len(r_data)):
		for j in range(len(r_data[i])):
			data.append(r_data[i][j])
			label.append(r_label[i][j])

	return data, label