#change file name later 

import numpy as np 
import librosa 
import matplotlib.pyplot as plt 

def emg_correct_mean(x):
	'''
	corrects the emg mean 
	As baseline emg values have an offset from zero. 
	'''
	x = x - np.mean(x, axis = 0)
	return x 

def emg_rectify(x):
	x = abs(x)
	return x 

def double_average(x):
	f = np.ones(9)/9.0
	v = np.convolve(x, f, mode='same')
	w = np.convolve(v, f, mode='same')
	return w 

def get_emg_features(emg_data, debug= False):
	xs = emg_data - emg_data.mean(axis=0, keepdims = True)
	frame_feature = []
	for i in range(emg_data.shape[1]):
		x = xs[:,i]
		# print("raw value x.shape ",x.shape)
		w = double_average(x)
		# print("double average w.shape ",w.shape)
		p = x - w
		# print("p.shape ",p.shape)
		r = np.abs(p)
		# print("r.shape ",r.shape)

		w_h = librosa.util.frame(w, frame_length= 16, hop_length= 6).mean(axis= 0)
		# print("w_h.shape ",w_h.shape)
		
		p_w = librosa.feature.rms(w, frame_length= 16, hop_length= 6, center= False)
		p_w = np.squeeze(p_w, 0)
		# print("p_w.shape ",p_w.shape)

		p_r = librosa.feature.rms(r, frame_length= 16, hop_length= 6, center= False)
		p_r = np.squeeze(p_r, 0)
		# print("p_r.shape ",p_r.shape)
		
		z_p = librosa.feature.zero_crossing_rate(p, frame_length= 16, hop_length= 6, center= False)
		z_p = np.squeeze(z_p, 0)
		# print("zero crossing z_p.shape ",z_p.shape)
		
		r_h = librosa.util.frame(r, frame_length= 16, hop_length= 6).mean(axis= 0)
		# print("recrifie high frequecy r_h.shape ",r_h.shape)

		s = abs(librosa.stft(np.ascontiguousarray(x), n_fft= 16, hop_length= 6, center= False))
		# print("short time fourirer transform s.shape ",s.shape)

		if 	debug:
			plt.subplot(7, 1, 1)
			plt.plot(x)
			plt.subplot(7, 1, 2)
			plt.plot(w_h)
			plt.subplot(7, 1, 3)
			plt.plot(p_w)
			plt.subplot(7, 1, 4)
			plt.plot(p_r)
			plt.subplot(7, 1, 5)
			plt.plot(z_p)
			plt.subplot(7, 1, 6)
			plt.plot(r_h)
			
			plt.subplot(7, 1, 7)
			plt.imshow(s, origin='lower', aspect= 'auto', interpolation= 'nearest')

			plt.show()

		frame_feature.append(np.stack([w_h, p_w, p_r, z_p, r_h], axis= 1))
		frame_feature.append(s.T)

	frame_feature = np.concatenate(frame_feature, axis= 1)
	return frame_feature.astype(np.float32)	

#source 
'''
Digital voicing...
https://scientificallysound.org/2016/08/22/python-analysing-emg-signals-part-4/
'''