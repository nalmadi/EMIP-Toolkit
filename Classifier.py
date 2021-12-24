import numpy as np
from scipy.spatial import distance_matrix
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
import copy, math

import math
import statistics

		
class IDT:
		def classify(self,raw_fixations, minimum_duration, sample_duration, threshold):
			"""I-DT classifier based on page 296 of eye tracker manual:
				https://psychologie.unibas.ch/fileadmin/user_upload/psychologie/Forschung/N-Lab/SMI_iView_X_Manual.pdf

				Notes:
					remember that some data is MSG for mouse clicks.
					some records are invalid with value -1.
					read right eye data only.

			Parameters
			----------
			raw_fixations : list
				a list of fixations information containing timestamp, x_cord, and y_cord

			minimum_duration : int, optional
				minimum duration for a fixation in milliseconds, less than minimum is considered noise.
				set to 50 milliseconds by default

			sample_duration : int, optional
				Sample duration in milliseconds, this is 4 milliseconds based on this eye tracker

			threshold : int, optional
				maximum distance from a group of samples to be considered a single fixation.
				Set to 25 pixels by default

			Returns
			-------
			list
				a list where each element is a list of timestamp, duration, x_cord, and y_cord
			"""

			# Create moving window based on minimum_duration
			window_size = int(math.ceil(minimum_duration / sample_duration))

			window_x = []
			window_y = []

			filter_fixation = []

			# Go over all SMPs in trial data
			for timestamp, x_cord, y_cord in raw_fixations:

				# Filter (skip) coordinates outside of the screen 1920×1080 px
				if x_cord < 0 or y_cord < 0 or x_cord > 1920 or y_cord > 1080:
					continue

				# Add sample if it appears to be valid
				window_x.append(x_cord)
				window_y.append(y_cord)

				# Calculate dispersion = [max(x) - min(x)] + [max(y) - min(y)]
				dispersion = (max(window_x) - min(window_x)) + (max(window_y) - min(window_y))

				# If dispersion is above maximum_dispersion
				if dispersion > threshold:

					# Then the window does not represent a fixation
					# Pop last item in window
					window_x.pop()
					window_y.pop()

					# Add fixation to fixations if window is not empty (size >= window_size)
					if len(window_x) == len(window_y) and len(window_x) > window_size:
						# The fixation is registered at the centroid of the window points
						filter_fixation.append(
							[timestamp, len(window_x) * 4, statistics.mean(window_x), statistics.mean(window_y)])

					window_x = []
					window_y = []

			return filter_fixation


class IMST:
	
	def minimum_spanning_tree(self,X, copy_X=True):
		"""X are edge weights of fully connected graph"""
		if copy_X:
			X = X.copy()

		if X.shape[0] != X.shape[1]:
			raise ValueError("X needs to be square matrix of edge weights")
		n_vertices = X.shape[0]
		spanning_edges = []

		# initialize with node 0:
		visited_vertices = [0]
		num_visited = 1
		# exclude self connections:
		diag_indices = np.arange(n_vertices)
		X[diag_indices, diag_indices] = np.inf

		while num_visited != n_vertices:
			new_edge = np.argmin(X[visited_vertices], axis=None)
			# 2d encoding of new_edge from flat, get correct indices
			new_edge = divmod(new_edge, n_vertices)
			new_edge = [visited_vertices[new_edge[0]], new_edge[1]]
			# add edge to tree
			spanning_edges.append(new_edge)
			visited_vertices.append(new_edge[1])
			# remove all edges inside current tree
			X[visited_vertices, new_edge[1]] = np.inf
			X[new_edge[1], visited_vertices] = np.inf
			num_visited += 1
		return np.vstack(spanning_edges)


	def classify(self,raw_fixations, sample_duration=4, minimum_duration=50,threshold=.6):
		"""I-MST classifier based on page 10 of the following benchmarking paper:
			https://digital.library.txstate.edu/bitstream/handle/10877/2577/fulltext.pdf?sequence=1&isAllowed=y
			Notes:
				remember that some data is MSG for mouse clicks.
				some records are invalid with value -1.
				read right eye data only.

		Parameters
		----------
		raw_fixations : list
			a list of fixations information containing timestamp, x_cord, and y_cord

		minimum_duration : int, optional
			minimum duration for a fixation in milliseconds, less than minimum is considered noise.
			set to 50 milliseconds by default

		sample_duration : int, optional
			Sample duration in milliseconds, this is 4 milliseconds based on this eye tracker

		maximum_dispersion : int, optional
			maximum distance from a group of samples to be considered a single fixation.
			Set to 25 pixels by default

		Returns
		-------
		list
			a list where each element is a list of timestamp, duration, x_cord, and y_cord
		"""

		window_size = int(math.ceil(minimum_duration / sample_duration))

		filter_fixation=[]
		raw_fixations_np=np.array(raw_fixations)
		# Go over each window:
		timestamp = np.array(raw_fixations_np[:,0])
		x_cord = np.array(raw_fixations_np[:,1])
		y_cord = np.array(raw_fixations_np[:,2])
		# now loop through each time window:
		for time_frame in range(0,math.ceil(len(x_cord)/window_size)):
			timestamp_now = timestamp[time_frame*window_size:(time_frame+1)*window_size-1]
			x_cord_now=x_cord[(time_frame*window_size):((time_frame+1)*window_size-1)]
			y_cord_now=y_cord[time_frame*window_size:(time_frame+1)*window_size-1]
			remove_coordinates= np.any(np.vstack((x_cord_now <= 0,y_cord_now <= 0,x_cord_now >= 1920,y_cord_now >= 1080)),axis=0)
			timestamp_now = timestamp_now[np.logical_not(remove_coordinates)]
			x_cord_now = x_cord_now[np.logical_not(remove_coordinates)]
			y_cord_now = y_cord_now[np.logical_not(remove_coordinates)]
			coord = np.vstack((x_cord_now, y_cord_now))
			# Pairwise matrix using Euclidean distance, if smaller than 5 points are used, we skip this window
			if coord.shape[1]<5:
				continue
			coord_pairwise = distance_matrix(coord.T, coord.T)
			# construct MST using the just calculated pairwise matrix and Prim's algorithm
			edge_list = self.minimum_spanning_tree(coord_pairwise)
			edge_list_flat = np.array(edge_list).T.tolist()
			corresponding_dist = coord_pairwise[tuple(edge_list_flat)]
			#print(corresponding_dist)
			# on the matrix, get the entries that is under the threshold:
			corresponding_dist=np.array(corresponding_dist,dtype=np.float64)
			fixation_which = np.where(corresponding_dist < threshold)
			fixation_which = np.sort(np.unique(edge_list[fixation_which].flatten()))
			filter_fixation.append([timestamp_now[-1],4*len(fixation_which),np.mean(x_cord_now[fixation_which]),np.mean(y_cord_now[fixation_which])])
		

		return filter_fixation


class IVT:
	def classify(self,raw_fixations,minimum_duration=50,sample_duration=4,threshold=0.6):
		"""I-VT velocity algorithm from Salvucci & Goldberg (2000). 
		
	
		For reference see:
		
		---
		Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations 
		and saccades in eye-tracking protocols. In Proceedings of the 
		2000 symposium on Eye tracking research & applications
		---
		
		Parameters
		----------
		raw_fixations : list
			a list of fixations information containing timestamp, x_cord, and y_cord

		minimum_duration : int, optional
			minimum duration for a fixation in milliseconds, less than minimum is considered noise.
			set to 50 milliseconds by default

		sample_duration : int, optional
			Sample duration in milliseconds, this is 4 milliseconds based on this eye tracker

		velocity_threshold : int, optional
			maximum distance from a group of samples to be considered a single fixation.
			Set to 25 pixels by default

		Returns
		-------
		list
			a list where each element is a list of timestamp, duration, x_cord, and y_cord
		"""
		# process time argument and calculate sample threshold
		raw_fixations_np=np.array(raw_fixations)
		times=raw_fixations_np[:,0]
		sfreq = 1 / np.mean(times[1:] - times[:-1])

		sample_thresh = sfreq * threshold / 1000
		
		# calculate movement velocities
		x_cord=raw_fixations_np[:,1]
		y_cord=raw_fixations_np[:,2]
		coord = np.stack([x_cord, y_cord])
		vels = np.linalg.norm(coord[:, 1:] - coord[:, :-1], axis=0)
		vels = np.concatenate([[0.], vels])
		
		# define classes by threshold
		is_fixation = np.empty(len(x_cord), dtype=object)
		is_fixation[:] = True
		is_fixation[vels > sample_thresh] = False
		
		# group consecutive classes to one segment
		segments = np.zeros(len(x_cord), dtype=int)
		for i in range(1, len(is_fixation)):
			if is_fixation[i] == is_fixation[i - 1]:
				segments[i] = segments[i - 1]
			else:
				segments[i] = segments[i - 1] + 1
		filter_fixation=[]
		for i in range (1,np.max(segments)+1):
			segment_where=np.where(segments==i)
			if len(segment_where)<50:
				continue
			time_now=times[segment_where[-1]]
			duration_now=len(segment_where)*4
			x_cord_now=np.mean(x_cord[segment_where])
			y_cord_now=np.mean(y_cord[segment_where])
			filter_fixation.append([time_now,duration_now,x_cord_now,y_cord_now])


		
	
		
		
		filter_fixation=np.array(filter_fixation)		
		
		return filter_fixation.T.tolist()


class I2MC:
	def twoClusterWeighting(xpos, ypos, missing, downsamples, downsampFilter, chebyOrder, windowtime, steptime, freq, maxerrors, dev=False):
		"""
		Description
		
		Parameters
		----------
		xpos : type
			Description
		ypos : type
			Description
		missing : type
			Description
		downsamples : type
			Description
		downsampFilter : type
			Description
		chebyOrder : type
			Description
		windowtime : type
			Description
		steptime : type
			Description
		freq : type
			Description
		maxerrors : type
			Description
		Returns
		-------
		finalweights : np.array
			Vector of 2-means clustering weights (one weight for each sample), the higher, the more likely a saccade happened        
			
		Examples
		--------
		>>> 
		"""   
		# calculate number of samples of the moving window
		nrsamples = int(windowtime/(1./freq))
		stepsize  = np.max([1,int(steptime/(1./freq))])
		
		# create empty weights vector
		totalweights = np.zeros(len(xpos))
		totalweights[missing] = np.nan
		nrtests = np.zeros(len(xpos))
		
		# stopped is always zero, unless maxiterations is exceeded. this
		# indicates that file could not be analysed after trying for x iterations
		stopped = False
		counterrors = 0
		
		# Number of downsamples
		nd = len(downsamples)
		
		# Downsample 
		if downsampFilter:
			# filter signal. Follow the lead of decimate(), which first runs a
			# Chebychev filter as specified below
			rp = .05 # passband ripple in dB
			b = [[] for i in range(nd)]
			a = [[] for i in range(nd)]
			for p in range(nd):
				b[p],a[p] = scipy.signal.cheby1(chebyOrder, rp, .8/downsamples[p]) 
		
		
		# idx for downsamples
		idxs = []
		for i in range(nd):
			idxs.append(np.arange(nrsamples,0,-downsamples[i],dtype=int)[::-1] - 1)
			
		# see where are missing in this data, for better running over the data
		# below.
		on,off = bool2bounds(missing)
		if on.size > 0:
			#  merge intervals smaller than nrsamples long 
			merge = np.argwhere((on[1:] - off[:-1])-1 < nrsamples).flatten()
			for p in merge[::-1]:
				off[p] = off[p+1]
				off = np.delete(off, p+1)
				on = np.delete(on, p+1)

			# check if intervals at data start and end are large enough
			if on[0]<nrsamples+1:
				# not enough data point before first missing, so exclude them all
				on[0]=0

			if off[-1]>(len(xpos)-1-nrsamples):
				# not enough data points after last missing, so exclude them all
				off[-1]=len(xpos)-1

			# start at first non-missing sample if trial starts with missing (or
			# excluded because too short) data
			if on[0]==0:
				i=off[0]+1 # start at first non-missing
			else:
				i=0
		else:
			i=0

		eind = i+nrsamples
		while eind<=(len(xpos)-1):
			# check if max errors is crossed
			if counterrors > maxerrors:
				print('Too many empty clusters encountered, aborting file. \n')
				stopped = True
				finalweights = np.nan
				return finalweights, stopped
			
			# select data portion of nrsamples
			idx = range(i,eind)
			ll_d = [[] for p in range(nd+1)]
			IDL_d = [[] for p in range(nd+1)]
			ll_d[0] = np.vstack([xpos[idx], ypos[idx]])
					
			# Filter the bit of data we're about to downsample. Then we simply need
			# to select each nth sample where n is the integer factor by which
			# number of samples is reduced. select samples such that they are till
			# end of window
			for p in range(nd):
				if downsampFilter:
					ll_d[p+1] = scipy.signal.filtfilt(b[p],a[p],ll_d[0])
					ll_d[p+1] = ll_d[p+1][:,idxs[p]]
				else:
					ll_d[p+1] = ll_d[0][:,idxs[p]]
			
			# do 2-means clustering
			try:
				for p in range(nd+1):
					IDL_d[p] = kmeans2(ll_d[p].T)[0]
			except Exception as e:
				print('Unknown error encountered at sample {}.\n'.format(i))
				raise e
			
			# detect switches and weight of switch (= 1/number of switches in
			# portion)
			switches = [[] for p in range(nd+1)]
			switchesw = [[] for p in range(nd+1)]
			for p in range(nd+1):
				switches[p] = np.abs(np.diff(IDL_d[p]))
				switchesw[p]  = 1./np.sum(switches[p])
			
			# get nearest samples of switch and add weight
			weighted = np.hstack([switches[0]*switchesw[0],0])
			for p in range(nd):
				j = np.array((np.argwhere(switches[p+1]).flatten()+1)*downsamples[p],dtype=int)-1
				for o in range(int(downsamples[p])):
					weighted[j+o] = weighted[j+o] + switchesw[p+1]
			
			# add to totalweights
			totalweights[idx] = totalweights[idx] + weighted
			# record how many times each sample was tested
			nrtests[idx] = nrtests[idx] + 1
			
			# update i
			i += stepsize
			eind += stepsize
			missingOn = np.logical_and(on>=i, on<=eind)
			missingOff = np.logical_and(off>=i, off<=eind)
			qWhichMiss = np.logical_or(missingOn, missingOff) 
			if np.sum(qWhichMiss) > 0:
				# we have some missing in this window. we don't process windows
				# with missing. Move back if we just skipped some samples, or else
				# skip whole missing and place start of window and first next
				# non-missing.
				if on[qWhichMiss][0] == (eind-stepsize):
					# continue at first non-missing
					i = off[qWhichMiss][0]+1
				else:
					# we skipped some points, move window back so that we analyze
					# up to first next missing point
					i = on[qWhichMiss][0]-nrsamples
				eind = i+nrsamples
				
			if eind>len(xpos)-1 and eind-stepsize<len(xpos)-1:
				# we just exceeded data bound, but previous eind was before end of
				# data: we have some unprocessed samples. retreat just enough so we
				# process those end samples once
				d = eind-len(xpos)+1
				eind = eind-d
				i = i-d
				
		# create final weights
		finalweights = totalweights/nrtests
		return finalweights, stopped

	def getFixations(finalweights, timestamp, xpos, ypos, missing, params):
		"""
		Algorithm: identification by 2 means clustering
		
		Parameters
		----------
		finalweights : type
			2-means clustering weighting
		timestamp : np.array
			Timestamp from Eyetracker (should be in ms!)
		xpos : np.array
			Horizontal coordinates from Eyetracker
		ypos : np.array
			Vertical coordinates from Eyetracker
		missing : np.array
			Vector containing the booleans for mising values
		par : Dictionary containing the following keys and values
			cutoffstd : float
				Number of std above mean clustering-weight to use as fixation cutoff
			onoffsetThresh : float
				Threshold (x*MAD of fixation) for walking forward/back for saccade off- and onsets
			maxMergeDist : float
				Maximum Euclidean distance in pixels between fixations for merging
			maxMergeTime : float
				Maximum time in ms between fixations for merging
			minFixDur : Float
				Minimum duration allowed for fiation
		Returns
		-------
		fix : Dictionary containing the following keys and values
			cutoff : float
				Cutoff used for fixation detection
			start : np.array
				Vector with fixation start indices
			end : np.array
				Vector with fixation end indices
			startT : np.array
				Vector with fixation start times
			endT : np.array
				Vector with fixation end times
			dur : type
				Vector with fixation durations
			xpos : np.array
				Vector with fixation median horizontal position (one value for each fixation in trial)
			ypos : np.array
				Vector with fixation median vertical position (one value for each fixation in trial)

		
		Examples
		--------
		>>> fix = getFixations(finalweights,data['time'],xpos,ypos,missing,par)
		>>> fix
			{'cutoff': 0.1355980099309374,
			'dur': array([366.599, 773.2  , 239.964, 236.608, 299.877, 126.637]),
			'end': array([111, 349, 433, 508, 600, 643]),
			'endT': array([ 369.919, 1163.169, 1443.106, 1693.062, 1999.738, 2142.977]),
			'flankdataloss': array([1., 0., 0., 0., 0., 0.]),
			'fracinterped': array([0.06363636, 0.        , 0.        , 0.        , 0.        ,
					0.        ]),
			'start': array([  2, 118, 362, 438, 511, 606]),
			'startT': array([   6.685,  393.325, 1206.498, 1459.79 , 1703.116, 2019.669]),
			'xpos': array([ 945.936,  781.056, 1349.184, 1243.92 , 1290.048, 1522.176]),
			'ypos': array([486.216, 404.838, 416.664, 373.005, 383.562, 311.904])}
		"""    
		### Extract the required parameters 
		assert ['cutoffstd','onoffsetThresh','maxMergeDist','maxMergeTime','minFixDur'] in par.keys()
		cutoffstd = par['cutoffstd']
		onoffsetThresh = par['onoffsetThresh']
		maxMergeDist = par['maxMergeDist']
		maxMergeTime = par['maxMergeTime']
		minFixDur = par['minFixDur']
			
		### first determine cutoff for finalweights
		cutoff = np.nanmean(finalweights) + cutoffstd*np.nanstd(finalweights,ddof=1)

		### get boolean of fixations
		fixbool = finalweights < cutoff
		
		### get indices of where fixations start and end
		fixstart, fixend = bool2bounds(fixbool)
		
		### for each fixation start, walk forward until recorded position is below 
		# a threshold of lambda*MAD away from median fixation position.
		# same for each fixation end, but walk backward
		for p in range(len(fixstart)):
			xFix = xpos[fixstart[p]:fixend[p]+1]
			yFix = ypos[fixstart[p]:fixend[p]+1]
			xmedThis = np.nanmedian(xFix)
			ymedThis = np.nanmedian(yFix)
			
			# MAD = median(abs(x_i-median({x}))). For the 2D version, I'm using
			# median 2D distance of a point from the median fixation position. Not
			# exactly MAD, but makes more sense to me for 2D than city block,
			# especially given that we use 2D distance in our walk here
			MAD = np.nanmedian(np.hypot(xFix-xmedThis, yFix-ymedThis))
			thresh = MAD*onoffsetThresh

			# walk until distance less than threshold away from median fixation
			# position. No walking occurs when we're already below threshold.
			i = fixstart[p]
			if i>0:  # don't walk when fixation starting at start of data 
				while np.hypot(xpos[i]-xmedThis,ypos[i]-ymedThis)>thresh:
					i = i+1
				fixstart[p] = i
				
			# and now fixation end.
			i = fixend[p]
			if i<len(xpos): # don't walk when fixation ending at end of data
				while np.hypot(xpos[i]-xmedThis,ypos[i]-ymedThis)>thresh:
					i = i-1
				fixend[p] = i

		### get start time, end time,
		starttime = timestamp[fixstart]
		endtime = timestamp[fixend]
		
		### loop over all fixation candidates in trial, see if should be merged
		for p in range(1,len(starttime))[::-1]:
			# get median coordinates of fixation
			xmedThis = np.median(xpos[fixstart[p]:fixend[p]+1])
			ymedThis = np.median(ypos[fixstart[p]:fixend[p]+1])
			xmedPrev = np.median(xpos[fixstart[p-1]:fixend[p-1]+1]);
			ymedPrev = np.median(ypos[fixstart[p-1]:fixend[p-1]+1]);
			
			# check if fixations close enough in time and space and thus qualify
			# for merging
			# The interval between the two fixations is calculated correctly (see
			# notes about fixation duration below), i checked this carefully. (Both
			# start and end of the interval are shifted by one sample in time, but
			# assuming practicalyl constant sample interval, thats not an issue.)
			if starttime[p]-endtime[p-1] < maxMergeTime and \
				np.hypot(xmedThis-xmedPrev,ymedThis-ymedPrev) < maxMergeDist:
				# merge
				fixend[p-1] = fixend[p];
				endtime[p-1]= endtime[p];
				# delete merged fixation
				fixstart = np.delete(fixstart, p)
				fixend = np.delete(fixend, p)
				starttime = np.delete(starttime, p)
				endtime = np.delete(endtime, p)
				
		### beginning and end of fixation must be real data, not interpolated.
		# If interpolated, those bit(s) at the edge(s) are excluded from the
		# fixation. First throw out fixations that are all missing/interpolated
		for p in range(len(starttime))[::-1]:
			miss = missing[fixstart[p]:fixend[p]+1]
			if np.sum(miss) == len(miss):
				fixstart = np.delete(fixstart, p)
				fixend = np.delete(fixend, p)
				starttime = np.delete(starttime, p)
				endtime = np.delete(endtime, p)
		
		# then check edges and shrink if needed
		for p in range(len(starttime)):
			if missing[fixstart[p]]:
				fixstart[p] = fixstart[p] + np.argmax(np.invert(missing[fixstart[p]:fixend[p]+1]))
				starttime[p]= timestamp[fixstart[p]]
			if missing[fixend[p]]:
				fixend[p] = fixend[p] - (np.argmax(np.invert(missing[fixstart[p]:fixend[p]+1][::-1]))+1)
				endtime[p] = timestamp[fixend[p]]
		
		### calculate fixation duration
		# if you calculate fixation duration by means of time of last sample during
		# fixation minus time of first sample during fixation (our fixation markers
		# are inclusive), then you always underestimate fixation duration by one
		# sample because you're in practice counting to the beginning of the
		# sample, not the end of it. To solve this, as end time we need to take the
		# timestamp of the sample that is one past the last sample of the fixation.
		# so, first calculate fixation duration by simple timestamp subtraction.
		fixdur = endtime-starttime
		
		# then determine what duration of this last sample was
		nextSamp = np.min(np.vstack([fixend+1,np.zeros(len(fixend),dtype=int)+len(timestamp)-1]),axis=0) # make sure we don't run off the end of the data
		extratime = timestamp[nextSamp]-timestamp[fixend] 
		
		# if last fixation ends at end of data, we need to determine how long that
		# sample is and add that to the end time. Here we simply guess it as the
		# duration of previous sample
		if not len(fixend)==0 and fixend[-1]==len(timestamp): # first check if there are fixations in the first place, or we'll index into non-existing data
			extratime[-1] = np.diff(timestamp[-3:-1])
		
		# now add the duration of the end sample to fixation durations, so we have
		# correct fixation durations
		fixdur = fixdur+extratime

		### check if any fixations are too short
		qTooShort = np.argwhere(fixdur<minFixDur)
		if len(qTooShort) > 0:
			fixstart = np.delete(fixstart, qTooShort)
			fixend = np.delete(fixend, qTooShort)
			starttime = np.delete(starttime, qTooShort)
			endtime = np.delete(endtime, qTooShort)
			fixdur = np.delete(fixdur, qTooShort)
			
		### process fixations, get other info about them
		xmedian = np.zeros(fixstart.shape) # vector for median
		ymedian = np.zeros(fixstart.shape)  # vector for median
		flankdataloss = np.zeros(fixstart.shape) # vector for whether fixation is flanked by data loss
		fracinterped = np.zeros(fixstart.shape) # vector for fraction interpolated
		for a in range(len(fixstart)):
			idxs = range(fixstart[a],fixend[a]+1)
			# get data during fixation
			xposf = xpos[idxs]
			yposf = ypos[idxs]
			# for all calculations below we'll only use data that is not
			# interpolated, so only real data
			qMiss = missing[idxs]
			
			# get median coordinates of fixation
			xmedian[a] = np.median(xposf[np.invert(qMiss)])
			ymedian[a] = np.median(yposf[np.invert(qMiss)])
			
			# determine whether fixation is flanked by period of data loss
			flankdataloss[a] = (fixstart[a]>0 and missing[fixstart[a]-1]) or (fixend[a]<len(xpos)-1 and missing[fixend[a]+1])
			
			# fraction of data loss during fixation that has been (does not count
			# data that is still lost)
			fracinterped[a]  = np.sum(np.invert(np.isnan(xposf[qMiss])))/(fixend[a]-fixstart[a]+1)

		# store all the results in an appropriate form 
		fix = {}
		fix['endT'] = endtime
		fix['dur'] = fixdur
		fix['xpos'] = xmedian
		fix['ypos'] = ymedian

		fix_list = list(fix.items())
		fix_arr = np.array(fix_list)

		return fix_arr.T
	
	def kmeans2(data):
		# n points in p dimensional space
		n = data.shape[0]
		maxit = 100

		## initialize using kmeans++ method.
		# code taken and slightly edited from scipy.cluster.vq
		dims = data.shape[1] if len(data.shape) > 1 else 1
		C = np.ndarray((2, dims))
		
		# first cluster
		C[0, :] = data[np.random.randint(data.shape[0])]

		# second cluster
		D = cdist(C[:1,:], data, metric='sqeuclidean').min(axis=0)
		probs = D/D.sum()
		cumprobs = probs.cumsum()
		r = np.random.rand()
		C[1, :] = data[np.searchsorted(cumprobs, r)]

		# Compute the distance from every point to each cluster centroid and the
		# initial assignment of points to clusters
		D = cdist(C, data, metric='sqeuclidean')
		# Compute the nearest neighbor for each obs using the current code book
		label = vq(data, C)[0]
		# Update the code book by computing centroids
		C = _vq.update_cluster_means(data, label, 2)[0]
		m = np.bincount(label)

		## Begin phase one:  batch reassignments
		#-----------------------------------------------------
		# Every point moved, every cluster will need an update
		prevtotsumD = math.inf
		iter = 0
		while True:
			iter += 1
			# Calculate the new cluster centroids and counts, and update the
			# distance from every point to those new cluster centroids
			Clast = C
			mlast = m
			D = cdist(C, data, metric='sqeuclidean')

			# Deal with clusters that have just lost all their members
			if np.any(m==0):
				i = np.argwhere(m==0)
				d = D[[label],[range(n)]]   # use newly updated distances
			
				# Find the point furthest away from its current cluster.
				# Take that point out of its cluster and use it to create
				# a new singleton cluster to replace the empty one.
				lonely = np.argmax(d)
				cFrom = label[lonely]    # taking from this cluster
				if m[cFrom] < 2:
					# In the very unusual event that the cluster had only
					# one member, pick any other non-singleton point.
					cFrom = np.argwhere(m>1)[0]
					lonely = np.argwhere(label==cFrom)[0]
				label[lonely] = i
			
				# Update clusters from which points are taken
				C = _vq.update_cluster_means(data, label, 2)[0]
				m = np.bincount(label)
				D = cdist(C, data, metric='sqeuclidean')
		
			# Compute the total sum of distances for the current configuration.
			totsumD = np.sum(D[[label],[range(n)]])
			# Test for a cycle: if objective is not decreased, back out
			# the last step and move on to the single update phase
			if prevtotsumD <= totsumD:
				label = prevlabel
				C = Clast
				m = mlast
				iter -= 1
				break
			if iter >= maxit:
				break
		
			# Determine closest cluster for each point and reassign points to clusters
			prevlabel = label
			prevtotsumD = totsumD
			newlabel = vq(data, C)[0]
		
			# Determine which points moved
			moved = newlabel != prevlabel
			if np.any(moved):
				# Resolve ties in favor of not moving
				moved[np.bitwise_and(moved, D[0,:]==D[1,:])] = False
			if not np.any(moved):
				break
			label = newlabel
			# update centers
			C = _vq.update_cluster_means(data, label, 2)[0]
			m = np.bincount(label)

	def bool2bounds(b):
		"""
		Finds all contiguous sections of true in a boolean

		Parameters
		----------
		data : np.array
			A 1d np.array containing True, False values.
		
		Returns
		-------
		on : np.array
			The array contains the indexes of the first value = True
		off : np.array
			The array contains the indexes of the last value = True in a sequence
		
		Example
		--------
		>>> import numpy as np
		>>> b = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0])
		>>> on, off = bool2bounds(b)
		>>> print(on)
		[0 4 8]
		>>> print(off)
		[0 6 9]
		"""
		b = np.array(np.array(b, dtype = np.bool), dtype=int)
		b = np.pad(b, (1, 1), 'constant', constant_values=(0, 0))
		D = np.diff(b)
		on  = np.array(np.where(D == 1)[0], dtype=int)
		off = np.array(np.where(D == -1)[0] -1, dtype=int)
		return on, off

	def classify(raw_fixations,par):
		assert ['downsamples','downsampFilter','chebyOrder','windowtime','steptime','freq','maxerrors','dev_cluster','maxerrors','dev_cluster'] in params.keys()
		raw_fixations_np=np.array(raw_fixations)
		xpos=raw_fixations[:,1]
		ypos=raw_fixations[:,2]
		time=raw_fixations[:,0]

		# we assume that no data are missing: 
		data['finalweights'], stopped = twoClusterWeighting(xpos, ypos, np.ones((xpos.shape[0],1),dtype=bool), par['downsamples'], par['downsampFilter'], par['chebyOrder'],par['windowtime'], par['steptime'],par['freq'],par['maxerrors'],par['dev_cluster'])
		if not stopped:
			filter_fixation = getFixations(data['finalweights'],data['time'],xpos,ypos,np.ones((xpos.shape[0],1),dtype=bool),par)
			return filter_fixation
		
class HMM:
	def forward(V, a, b, initial_distribution):
		'''
		carries out the long computations in HMM models
		'''
		alpha = np.zeros((V.shape[0], a.shape[0]))
		alpha[0, :] = initial_distribution * b[:, V[0]]

		for t in range(1, V.shape[0]):
			# matrix computation
			for j in range(a.shape[0]):
				alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]

		return alpha


	def backward(V, a, b):
		'''
		
		'''
		beta = np.zeros((V.shape[0], a.shape[0]))

		# set beta(T) = 1
		beta[V.shape[0] - 1] = np.ones((a.shape[0]))

		# Loop in backward way from T-1 to 1
		# python indexing so loop will be T-2 to 0
		for t in range(V.shape[0] - 2, -1, -1):
			for j in range(a.shape[0]):
				beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])

		return beta


	def baum_welch(V, a, b, initial_distribution, n_iter=4):
		'''
		finds unknown parameters of a hidden Markov model
		'''
		M = a.shape[0]
		T = len(V)

		for n in range(n_iter):
			alpha = HMM_classifier.forward(V, a, b, initial_distribution)
			beta = HMM_classifier.backward(V, a, b)

			xi = np.zeros((M, M, T - 1))
			for t in range(T - 1):
				denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
				for i in range(M):
					numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
					xi[i, :, t] = numerator / denominator

			gamma = np.sum(xi, axis=1)
			a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

			# Add additional T'th element in gamma
			gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

			K = b.shape[1]
			denominator = np.sum(gamma, axis=1)
			for l in range(K):
				b[:, l] = np.sum(gamma[:, V == l], axis=1)

			b = np.divide(b, denominator.reshape((-1, 1)))

		return (a, b)


	def viterbi(V, a, b, initial_distribution):
		'''
		computes the the most likely state sequence in a Hidden Markov model
		'''
		T = V.shape[0]
		M = a.shape[0]

		omega = np.zeros((T, M))
		omega[0, :] = np.log(initial_distribution * b[:, V[0]])

		prev = np.zeros((T - 1, M))

		for t in range(1, T):
			for j in range(M):
				# Same as Forward Probability
				probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])

				# This is our most probable state given previous state at time t (1)
				prev[t - 1, j] = np.argmax(probability)

				# This is the probability of the most probable state (2)
				omega[t, j] = np.max(probability)

		# Path Array
		S = np.zeros(T)

		# Find the most probable last hidden state
		last_state = np.argmax(omega[T - 1, :])

		S[0] = last_state

		backtrack_index = 1
		for i in range(T - 2, -1, -1):
			S[backtrack_index] = prev[i, int(last_state)]
			last_state = prev[i, int(last_state)]
			backtrack_index += 1

		# Flip the path array since we were backtracking
		S = np.flip(S, axis=0)

		# Convert numeric values to actual hidden states
		result = []
		for s in S:
			if s == 0:
				result.append("A")
			else:
				result.append("B")

		return result


	def classify(raw_fixations, velocity_threshold=50, sample_duration=4, maximum_dispersion=25):

		'''Hidden Markov Model Identification algorithm from Salvucci & Goldberg (2000)
		https://digital.library.txstate.edu/bitstream/handle/10877/2577/fulltext.pdf?sequence=1&isAllowed=y
		Input: array of eye position points, velocity threshold, initial,
		transitional, observation probabilities
		
		Output: array of fixations and saccades
		'''

		# Calculate point­to­point velocities for each point in the eye position array
		timestamp=raw_fixations['timestamp']
		sample_freq = 1 / np.mean(timestamp[1:] - timestamp[:-1]) 

		sample_threshold = sample_freq * velocity_threshold / 1000
		
		x_coord = raw_fixations['x_cord']
		y_coord=raw_fixations['y_cord']
		gaze = np.stack([x_coord, y_coord])
		velocities = np.linalg.norm(gaze[:, 1:] - gaze[:, :-1], axis=0)
		velocities = np.concatenate([[0.], velocities])
		
		# Mark points below velocity threshold as fixations and the points above the threshold as saccades
		is_fixation = np.empty(len(x_coord), dtype=object)
		is_fixation[:] = True
		is_fixation[velocities > sample_threshold] = False

		# Define Viterbi sampler of the HMM and re­estimate fixation, saccade assignment for every eye position point
		
		# Use Baum­welch algorithm to re­estimate initial, transition,	and observation probabilities for the defined sampler
		# Merge Function(array of pre classified fixation and saccades)
		# Return saccades and fixations




import scipy.stats as scpst
class KF:
	def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):
		if(F is None or H is None):
			raise ValueError("|--INVALID PARAMETERS--|")

		self.n = F.shape[1]
		self.m = H.shape[1]

		# Next State Function
		self.F = F
		# Measurement Function
		self.H = H
		# External Motion (dotted with mu in prediction step)
		self.B = 0 if B is None else B
		# Identity Matrix
		self.Q = np.eye(self.n) if Q is None else Q
		self.R = np.eye(self.n) if R is None else R
		self.P = np.eye(self.n) if P is None else P
		# Initial State Matrix (Position & Velocity)
		self.x = np.zeros((self.n, 1)) if x0 is None else x0

	def predict(self, u = 0):
		self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
		self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
		return self.x

	def update(self, z):
		y = z - np.dot(self.H, self.x)
		S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
		K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
		self.x = self.x + np.dot(K, y)
		I = np.eye(self.n)
		self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
			(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

	@staticmethod
	def classify(raw_fixations, sample_duration=4, minimum_duration=50, threshold=1):
		filter_fixation = []
		raw_fixations_np = np.array(raw_fixations)
		times = raw_fixations_np[:,0]
		dt = sample_duration
		sfreq = 1 / np.mean(times[1:] - times[:-1])
		sample_thresh = sfreq * threshold / 1000
		
		# calculate movement velocities
		x_coord = raw_fixations_np[:,1]
		y_coord = raw_fixations_np[:,2]
		coord = np.stack([x_coord, y_coord])
		
		F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
		H = np.array([1, 0, 0]).reshape(1, 3)
		Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
		R = np.array([0.5]).reshape(1, 1)

		for time_frame in range(0,math.ceil(len(x_coord)/window_size)):
			timestamp_now = timestamp[time_frame*window_size:(time_frame+1)*window_size-1]
			x_coord_now = x_coord[(time_frame*window_size):((time_frame+1)*window_size-1)]
			y_coord_now = y_coord[time_frame*window_size:(time_frame+1)*window_size-1]
			remove_coordinates = np.any(np.vstack((x_coord_now <= 0, y_coord_now <= 0, x_coord_now >= 1920, y_coord_now >= 1080)), axis=0)
			timestamp_now = timestamp_now[np.logical_not(remove_coordinates)]
			x_coord_now = x_coord_now[np.logical_not(remove_coordinates)]
			y_coord_now = y_coord_now[np.logical_not(remove_coordinates)]
			coord_now = np.vstack((x_coord_now, y_coord_now))
			if coord_now.shape[1] < 5:
				continue

			sfreq = 1 / np.mean(timestamp_now[1:] - timestamp_now[:-1])
			vels_now = np.linalg.norm(coord_now[:, 1:] - coord_now[:, :-1], axis=0)
			vels_now = np.concatenate([[0.], vels_now])
			vels_now_pred = []
			kf = KF(F = F, H = H, Q = Q, R = R)

			for z in vels_now:
				vels_now_pred.append(np.dot(H, kf.predict())[0])
				kf.update(z)
				
			chi_square_now,_ = scpst.chisquare(vels_now, vels_now_pred)
			if chi_square_now < threshold:
				# a fixation:
				filter_fixation.append[timestamp_now[-1], len(x_coord_now)*4, np.mean(x_coord_now), np.mean(y_coord_now)]
		return filter_fixation
