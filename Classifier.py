import numpy as np
from scipy.spatial import distance_matrix
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
import copy, math

import math
import os
import statistics

		
class IDT_classifier: 
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
class IMST_classifier:
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
	def velocity(self,raw_fixations,sample_duration=4,minimum_duration=50):

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
		print(len(x_cord))
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
			# Pairwise matrix using Euclidean distance
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



class IVT_classifier:
	def classify(raw_fixations,minimum_duration=50,sample_duration=4,threshold=0.6):
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
		is_fixation = np.empty(len(x), dtype=object)
		is_fixation[:] = True
		is_fixation[vels > sample_thresh] = False
		
		# group consecutive classes to one segment
		segments = np.zeros(len(x), dtype=int)
		for idx in range(1, len(is_fixation)):
			if is_fixation[idx] == is_fixation[idx - 1]:
				segments[idx] = segments[idx - 1]
			else:
				segments[idx] = segments[idx - 1] + 1
		

		
		filter_fixation_x=x_cord[np.mean(x_cord[segments[i]:segments[i+1]]) for i in range(len(segments))]
		filter_fixation_y=x_cord[np.mean(y_cord[segments[i]:segments[i+1]]) for i in range(len(segments))]
		
		filter_fixation_timestamp=times[segments]
		duration=
		return [duration]


class I2MC:
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
		cutoffstd = params['cutoffstd']
		onoffsetThresh = params['onoffsetThresh']
		maxMergeDist = params['maxMergeDist']
		maxMergeTime = params['maxMergeTime']
		minFixDur = params['minFixDur']
			
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
		xpos_out=[xpos[fixstart[i]:fixend[i]] for i in range(len(fixstart))]
		ypos_out=[ypos[fixstart[i]:fixend[i]] for i in range(len(fixstart))]
		time_out=[ypos[fixstart[i]:fixend[i]] for i in range(len(fixstart))]

		return [time_out,xpos_out,ypos_out]


