filepath = './mentally/right/OpenBCI-RAW-2020-11-29_13-22-49.txt'
num_classes = 1
surrounding = 150
include_surrounding = True
sample_rate = 250
channels = range(0, 8)

f = open(filepath, 'r')
contents = map(lambda x : x.strip(), f.readlines())
frames_original = list(filter(lambda x : x and x[0] != '%', contents))[1:]
frames_original = 	list(map(lambda s : list(map( lambda ss: ss.strip(), s.split(','))), frames_original))
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
		print('start speaking')
		speaking = True
		start_index = i 
	if speaking and not bool(frames[i][-1]):
		print('stop speaking')
		speaking = False
		if np.all(map(bool, frames[i-50:i, -1])):
			print(num)
			num+= 1 
			sequence_groups[num % num_classes].append(list(frames[start_index- padding: i + padding, channels]))

if bool(frames[-1][-1]):
	sequence_groups[num % num_classes].append(list(frames[start_index- padding:, channels]))


#join group 
g = np.array(sequence_groups)

groups = [[] for _ in range(len(g[0]))]
for i in range(len(g)):
	for j in range(len(g[0])):
		groups[j] += list(g[i][j])
	#return np.array(groups)